#!/usr/bin/env python3
"""
analyze_volume_curves.py

基于 ACDC 视频分割结果（adapter/baseline × lv/myo/rv）执行：
  1. 体积计算：逐时间帧计算每位受试者各解剖结构的物理体积（mL）
  2. 体积波动曲线绘制：按疾病分组着色，展示心脏在心动周期内的体积变化

输出目录：<project_root>/analysis_outputs/
  - volume_tables/   ── CSV 体积数值表格
  - figures/         ── 体积曲线 PNG/PDF（300 dpi）
  - disease_map.csv  ── 受试者 ID ↔ 疾病类型映射

用法：
  python scripts/analyze_volume_curves.py [--project_root /path/to/CardioSAM]
                                          [--database_root /path/to/acdc/testing]
                                          [--outputs_root /path/to/outputs]
                                          [--analysis_root /path/to/analysis_outputs]
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from PIL import Image

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 路径配置（从脚本位置推导，可通过 CLI 参数覆盖）
# ─────────────────────────────────────────────────────────────────────────────
# 脚本在 <PROJECT_ROOT>/scripts/ 下，PROJECT_ROOT 为其父目录
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_PROJECT_ROOT = _SCRIPT_DIR.parent

STRUCTURES = ["lv", "myo", "rv"]
# METHODS 由 main() 根据 --adapter_suffix 动态设置
METHODS = ["adapter", "baseline"]

# 疾病名称、颜色、顺序
DISEASE_INFO = {
    "NOR":  {"label": "Normal (NOR)",                   "color": "#2ca02c"},  # 绿色
    "DCM":  {"label": "Dilated Cardiomyopathy (DCM)",   "color": "#1f77b4"},  # 蓝色
    "HCM":  {"label": "Hypertrophic Cardiomyopathy (HCM)", "color": "#ff7f0e"},  # 橙色
    "MINF": {"label": "Myocardial Infarction (MINF)",   "color": "#d62728"},  # 红色
    "RV":   {"label": "Right Ventricular (RV)",         "color": "#9467bd"},  # 紫色
}

STRUCTURE_LABELS = {
    "lv":  "Left Ventricle (LV)",
    "myo": "Myocardium (MYO)",
    "rv":  "Right Ventricle (RV)",
}

# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def load_disease_map(database_root: Path) -> dict:
    """从 ACDC testing 数据库读取 patient_id → disease_group 映射。"""
    disease_map = {}
    for patient_dir in sorted(database_root.iterdir()):
        cfg = patient_dir / "Info.cfg"
        if not cfg.exists():
            continue
        pid = patient_dir.name
        for line in cfg.read_text().splitlines():
            if line.startswith("Group:"):
                disease_map[pid] = line.split(":")[1].strip()
                break
    return disease_map


def compute_patient_volumes(
    method: str, structure: str, patient_id: str, infer_dirs: dict
) -> pd.DataFrame:
    """
    计算单个受试者某解剖结构在每个时间帧的体积（mL）。

    返回 DataFrame，列：frame_idx, volume_mm3, volume_ml, frame_pct, num_frames
    """
    infer_dir = infer_dirs[(method, structure)]
    patient_dir = infer_dir / "test" / patient_id

    if not patient_dir.exists():
        return pd.DataFrame()

    slices = sorted([d for d in patient_dir.iterdir() if d.is_dir()])
    if not slices:
        return pd.DataFrame()

    # 读取第一个有效 meta 确定 num_frames
    num_frames = None
    for sl in slices:
        meta_path = sl / "video_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            num_frames = meta["video_num_frames"]
            break

    if num_frames is None:
        return pd.DataFrame()

    # 按帧累加所有切片面积（mm²）
    frame_area_mm2 = {fi: 0.0 for fi in range(num_frames)}

    for sl in slices:
        meta_path = sl / "video_meta.json"
        masks_dir = sl / "masks"
        if not meta_path.exists() or not masks_dir.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        ps = meta["pixel_spacing_mm"]  # [row_spacing, col_spacing]
        pixel_area_mm2 = ps[0] * ps[1]
        slice_thickness_mm = meta["slice_thickness_mm"]
        voxel_vol_mm3 = pixel_area_mm2 * slice_thickness_mm

        for mask_file in masks_dir.iterdir():
            if not mask_file.suffix == ".png":
                continue
            # 文件名 frame_XXXX.png → 帧索引
            try:
                fi = int(mask_file.stem.replace("frame_", ""))
            except ValueError:
                continue
            if fi >= num_frames:
                continue

            try:
                img = Image.open(mask_file)
                img.load()
                mask = np.array(img)
            except Exception as e:
                print(f"    [WARN] Skipping corrupted mask {mask_file}: {e}")
                continue
            if mask.ndim == 3:
                mask = mask[..., 0]
            pixel_count = int((mask > 127).sum())
            frame_area_mm2[fi] += pixel_count * voxel_vol_mm3

    rows = []
    for fi in range(num_frames):
        vol_mm3 = frame_area_mm2[fi]
        rows.append({
            "frame_idx": fi,
            "volume_mm3": vol_mm3,
            "volume_ml": vol_mm3 / 1000.0,
            "frame_pct": fi / (num_frames - 1) * 100.0 if num_frames > 1 else 0.0,
            "num_frames": num_frames,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 体积计算主流程
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_volumes(disease_map: dict, infer_dirs: dict) -> pd.DataFrame:
    """计算所有 method × structure × patient × frame 的体积，返回总表。"""
    all_rows = []
    patients = sorted(disease_map.keys())

    for method in METHODS:
        for structure in STRUCTURES:
            print(f"  Computing volumes: method={method}, structure={structure} ...")
            for pid in patients:
                df = compute_patient_volumes(method, structure, pid, infer_dirs)
                if df.empty:
                    print(f"    [WARN] No data for {method}/{structure}/{pid}")
                    continue
                df["patient_id"] = pid
                df["disease"] = disease_map[pid]
                df["method"] = method
                df["structure"] = structure
                all_rows.append(df)

    if not all_rows:
        raise RuntimeError("No volume data found. Check infer directories.")

    return pd.concat(all_rows, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# 绘图函数
# ─────────────────────────────────────────────────────────────────────────────

def plot_volume_curves(
    df_method: pd.DataFrame,
    method: str,
    figures_dir: Path,
):
    """
    为单个 method 绘制体积波动曲线图（3 个解剖结构各一个子图，共 1 张图）。
    每条曲线 = 1 名受试者，颜色 = 疾病类型。
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=False)
    fig.suptitle(
        f"Cardiac Volume Fluctuation Curves  ·  {method.capitalize()} Model",
        fontsize=16, fontweight="bold", y=1.01,
    )

    for ax, structure in zip(axes, STRUCTURES):
        df_s = df_method[df_method["structure"] == structure].copy()
        if df_s.empty:
            ax.set_title(STRUCTURE_LABELS[structure])
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        # 按患者绘制曲线
        for pid, grp in df_s.groupby("patient_id"):
            disease = grp["disease"].iloc[0]
            color = DISEASE_INFO.get(disease, {}).get("color", "gray")
            grp_sorted = grp.sort_values("frame_pct")

            # 仅绘制有体积（>0）的帧；若全为 0 则用虚线提示
            nonzero = grp_sorted[grp_sorted["volume_ml"] > 0]
            if nonzero.empty:
                ax.plot(
                    grp_sorted["frame_pct"], grp_sorted["volume_ml"],
                    color=color, lw=0.8, alpha=0.4, ls=":",
                )
            else:
                ax.plot(
                    grp_sorted["frame_pct"], grp_sorted["volume_ml"],
                    color=color, lw=1.2, alpha=0.75,
                )

        ax.set_xlabel("Cardiac Cycle (%)", fontsize=11)
        ax.set_ylabel("Volume (mL)", fontsize=11)
        ax.set_title(STRUCTURE_LABELS[structure], fontsize=13, fontweight="bold")
        ax.set_xlim(0, 100)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.tick_params(axis="both", labelsize=10)

    # 统一图例
    legend_handles = [
        mlines.Line2D([], [], color=info["color"], lw=2, label=info["label"])
        for info in DISEASE_INFO.values()
    ]
    fig.legend(
        handles=legend_handles,
        title="Disease Group",
        title_fontsize=11,
        fontsize=10,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.06),
        frameon=True,
    )

    plt.tight_layout()

    for ext in ["png", "pdf"]:
        save_path = figures_dir / f"volume_curves_{method}.{ext}"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    plt.close(fig)


def plot_per_structure_curves(
    df_method: pd.DataFrame,
    method: str,
    figures_dir: Path,
):
    """为每个解剖结构单独绘制一张图（更大分辨率，便于查看）。"""
    for structure in STRUCTURES:
        df_s = df_method[df_method["structure"] == structure].copy()
        if df_s.empty:
            continue

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.set_title(
            f"{STRUCTURE_LABELS[structure]}  ·  {method.capitalize()} Model\n"
            f"Volume Fluctuation over Cardiac Cycle",
            fontsize=14, fontweight="bold",
        )

        for pid, grp in df_s.groupby("patient_id"):
            disease = grp["disease"].iloc[0]
            color = DISEASE_INFO.get(disease, {}).get("color", "gray")
            grp_sorted = grp.sort_values("frame_pct")
            nonzero = grp_sorted[grp_sorted["volume_ml"] > 0]
            if nonzero.empty:
                ax.plot(grp_sorted["frame_pct"], grp_sorted["volume_ml"],
                        color=color, lw=0.8, alpha=0.35, ls=":")
            else:
                ax.plot(grp_sorted["frame_pct"], grp_sorted["volume_ml"],
                        color=color, lw=1.3, alpha=0.8,
                        label=f"{pid}({disease})")

        ax.set_xlabel("Cardiac Cycle (%)", fontsize=12)
        ax.set_ylabel("Volume (mL)", fontsize=12)
        ax.set_xlim(0, 100)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.tick_params(axis="both", labelsize=11)

        legend_handles = [
            mlines.Line2D([], [], color=info["color"], lw=2.5, label=info["label"])
            for info in DISEASE_INFO.values()
        ]
        ax.legend(
            handles=legend_handles,
            title="Disease Group",
            title_fontsize=11,
            fontsize=10,
            loc="upper right",
            framealpha=0.9,
        )

        plt.tight_layout()
        for ext in ["png", "pdf"]:
            save_path = figures_dir / f"volume_curves_{method}_{structure}.{ext}"
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"  Saved: {save_path}")
        plt.close(fig)


def plot_per_disease_curves(
    df_method: pd.DataFrame,
    method: str,
    figures_dir: Path,
):
    """
    按疾病类型分组绘图。
    每张图 = 1 种疾病，3 个子图（LV / MYO / RV 并排）。
    每条曲线 = 1 名患者，颜色区分患者（colormap 顺序色）。
    """
    diseases = [d for d in DISEASE_INFO.keys() if d in df_method["disease"].unique()]

    for disease in diseases:
        df_d = df_method[df_method["disease"] == disease].copy()
        if df_d.empty:
            continue

        patients = sorted(df_d["patient_id"].unique())
        # 为每位患者分配一个颜色（colormap）
        cmap = plt.get_cmap("tab10")
        pid_colors = {pid: cmap(i % 10) for i, pid in enumerate(patients)}

        disease_label = DISEASE_INFO[disease]["label"]
        fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=False)
        fig.suptitle(
            f"Cardiac Volume Fluctuation  ·  {method.capitalize()} Model  ·  {disease_label}",
            fontsize=15, fontweight="bold", y=1.02,
        )

        for ax, structure in zip(axes, STRUCTURES):
            df_s = df_d[df_d["structure"] == structure].copy()
            if df_s.empty:
                ax.set_title(STRUCTURE_LABELS[structure])
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=12, color="gray")
                continue

            for pid, grp in df_s.groupby("patient_id"):
                color = pid_colors[pid]
                grp_sorted = grp.sort_values("frame_pct")
                nonzero = grp_sorted[grp_sorted["volume_ml"] > 0]
                ls = ":" if nonzero.empty else "-"
                alpha = 0.4 if nonzero.empty else 0.85
                ax.plot(
                    grp_sorted["frame_pct"], grp_sorted["volume_ml"],
                    color=color, lw=1.6, alpha=alpha, ls=ls, label=pid,
                )

            ax.set_xlabel("Cardiac Cycle (%)", fontsize=11)
            ax.set_ylabel("Volume (mL)", fontsize=11)
            ax.set_title(STRUCTURE_LABELS[structure], fontsize=13, fontweight="bold")
            ax.set_xlim(0, 100)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.tick_params(axis="both", labelsize=10)

        # 患者图例放在图右侧
        handles = [
            mlines.Line2D([], [], color=pid_colors[pid], lw=2, label=pid)
            for pid in patients
        ]
        fig.legend(
            handles=handles,
            title="Patient ID",
            title_fontsize=10,
            fontsize=9,
            loc="center right",
            bbox_to_anchor=(1.08, 0.5),
            frameon=True,
        )

        plt.tight_layout()
        disease_key = disease.lower()
        for ext in ["png", "pdf"]:
            save_path = figures_dir / f"volume_curves_{method}_by_disease_{disease_key}.{ext}"
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"  Saved: {save_path}")
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="ACDC Volume Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--project_root",
        type=Path,
        default=_DEFAULT_PROJECT_ROOT,
        help="Project root directory (default: parent of this script's directory).",
    )
    parser.add_argument(
        "--database_root",
        type=Path,
        default=None,
        help="ACDC testing database root containing patient dirs with Info.cfg. "
             "Default: <project_root>/database/testing",
    )
    parser.add_argument(
        "--outputs_root",
        type=Path,
        default=None,
        help="Root of inference outputs directory. "
             "Default: <project_root>/outputs",
    )
    parser.add_argument(
        "--analysis_root",
        type=Path,
        default=None,
        help="Root directory for analysis outputs. "
             "Default: <project_root>/analysis_outputs",
    )
    parser.add_argument(
        "--adapter_suffix",
        type=str,
        default="",
        help="Suffix appended to adapter inference directory names. "
             "e.g. '_280px' → acdc_video_infer_adapter_lv_280px. "
             "Default: '' (uses acdc_video_infer_adapter_lv etc.)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    project_root = args.project_root.resolve()
    database_root = (args.database_root or project_root / "database" / "testing").resolve()
    outputs_root  = (args.outputs_root  or project_root / "outputs").resolve()
    sfx = args.adapter_suffix
    default_analysis_root = project_root / f"analysis_outputs{sfx}"
    analysis_root = (args.analysis_root or default_analysis_root).resolve()

    sfx = args.adapter_suffix  # e.g. "" or "_280px"
    adapter_label = f"adapter{sfx}" if sfx else "adapter"

    infer_dirs = {
        (adapter_label, "lv"):  outputs_root / f"acdc_video_infer_adapter_lv{sfx}",
        (adapter_label, "myo"): outputs_root / f"acdc_video_infer_adapter_myo{sfx}",
        (adapter_label, "rv"):  outputs_root / f"acdc_video_infer_adapter_rv{sfx}",
        ("baseline",    "lv"):  outputs_root / "acdc_video_infer_baseline_lv",
        ("baseline",    "myo"): outputs_root / "acdc_video_infer_baseline_myo",
        ("baseline",    "rv"):  outputs_root / "acdc_video_infer_baseline_rv",
    }
    # 动态设置 method 列表
    global METHODS
    METHODS = [adapter_label, "baseline"]

    print("=" * 60)
    print("ACDC Volume Analysis Pipeline")
    print("=" * 60)
    print(f"  Project root  : {project_root}")
    print(f"  Database root : {database_root}")
    print(f"  Outputs root  : {outputs_root}")
    print(f"  Analysis root : {analysis_root}")

    # 创建输出目录
    volume_dir = analysis_root / "volume_tables"
    figures_dir = analysis_root / "figures"
    for d in [volume_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. 疾病映射
    print("\n[1] Loading disease map ...")
    disease_map = load_disease_map(database_root)
    df_disease = pd.DataFrame(
        [{"patient_id": k, "disease": v} for k, v in sorted(disease_map.items())]
    )
    disease_csv = analysis_root / "disease_map.csv"
    df_disease.to_csv(disease_csv, index=False)
    print(f"  Disease map saved: {disease_csv}")
    print(f"  Distribution:\n{df_disease['disease'].value_counts().to_string()}")

    # 2. 体积计算
    print("\n[2] Computing volumes for all method × structure × patient × frame ...")
    df_all = compute_all_volumes(disease_map, infer_dirs)

    # 保存总表
    all_csv = volume_dir / "all_volumes.csv"
    df_all.to_csv(all_csv, index=False)
    print(f"\n  Total records: {len(df_all)}")
    print(f"  All volumes saved: {all_csv}")

    # 保存分表（按 method × structure）
    for (method, structure), grp in df_all.groupby(["method", "structure"]):
        out_csv = volume_dir / f"volumes_{method}_{structure}.csv"
        grp.sort_values(["patient_id", "frame_idx"]).to_csv(out_csv, index=False)
        print(f"  Saved: {out_csv}")

    # 保存绘图原始数据（宽格式，便于进一步处理）
    for method in METHODS:
        df_m = df_all[df_all["method"] == method]
        # Pivot: rows=patient×frame_pct, cols=structure
        pivot = df_m.pivot_table(
            index=["patient_id", "disease", "frame_idx", "frame_pct", "num_frames"],
            columns="structure",
            values="volume_ml",
        ).reset_index()
        pivot.columns.name = None
        raw_csv = volume_dir / f"plot_data_{method}.csv"
        pivot.to_csv(raw_csv, index=False)
        print(f"  Plot raw data saved: {raw_csv}")

    # 3. 绘图
    print("\n[3] Plotting volume fluctuation curves ...")
    for method in METHODS:
        df_m = df_all[df_all["method"] == method].copy()
        print(f"\n  --- {method} ---")
        # 3a. 合并三结构子图（1张图，所有疾病混合）
        plot_volume_curves(df_m, method, figures_dir)
        # 3b. 每结构单独大图
        plot_per_structure_curves(df_m, method, figures_dir)
        # 3c. 按疾病分组：每疾病1张图，3结构并排
        plot_per_disease_curves(df_m, method, figures_dir)

    # 4. 统计摘要
    print("\n[4] Summary statistics ...")
    summary_rows = []
    for (method, structure, disease), grp in df_all.groupby(["method", "structure", "disease"]):
        max_vol = grp.groupby("patient_id")["volume_ml"].max()
        min_vol = grp.groupby("patient_id")["volume_ml"].min()
        summary_rows.append({
            "method": method,
            "structure": structure,
            "disease": disease,
            "n_patients": len(max_vol),
            "mean_max_vol_ml": max_vol.mean(),
            "std_max_vol_ml": max_vol.std(),
            "mean_min_vol_ml": min_vol.mean(),
            "std_min_vol_ml": min_vol.std(),
        })
    df_summary = pd.DataFrame(summary_rows)
    summary_csv = analysis_root / "volume_summary.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"  Summary saved: {summary_csv}")
    print(df_summary.to_string(index=False))

    print("\n" + "=" * 60)
    print("Analysis complete! All outputs in:", analysis_root)
    print("=" * 60)


if __name__ == "__main__":
    main()
