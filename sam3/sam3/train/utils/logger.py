# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import atexit
import fnmatch
import functools
import logging
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr
from numpy import ndarray
from sam3.train.utils.train_utils import get_machine_local_and_dist_rank, makedir
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

Scalar = Union[Tensor, ndarray, int, float]


def make_tensorboard_logger(log_dir: str, **writer_kwargs: Any):
    makedir(log_dir)
    summary_writer_method = SummaryWriter
    return TensorBoardLogger(
        path=log_dir, summary_writer_method=summary_writer_method, **writer_kwargs
    )


def make_wandb_logger(
    project: str,
    name: Optional[str] = None,
    dir: Optional[str] = None,
    entity: Optional[str] = None,
    tags: Optional[list[str]] = None,
    mode: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    group: Optional[str] = None,
    job_type: Optional[str] = None,
    resume: Optional[str] = None,
    run_id: Optional[str] = None,
    segm_vis: Optional[Dict[str, Any]] = None,
    # ==================== 自定义添加：W&B 指标过滤（仅保留关键指标） ====================
    log_keys: Optional[list[str]] = None,
    log_key_patterns: Optional[list[str]] = None,
    # ==================== 自定义添加结束 ====================
    **kwargs: Any,
):
    if dir:
        makedir(dir)
    return WandbLogger(
        project=project,
        name=name,
        dir=dir,
        entity=entity,
        tags=tags,
        mode=mode,
        config=config,
        group=group,
        job_type=job_type,
        resume=resume,
        run_id=run_id,
        segm_vis=segm_vis,
        log_keys=log_keys,
        log_key_patterns=log_key_patterns,
        **kwargs,
    )


class TensorBoardWriterWrapper:
    """
    A wrapper around a SummaryWriter object.
    """

    def __init__(
        self,
        path: str,
        *args: Any,
        filename_suffix: str = None,
        summary_writer_method: Any = SummaryWriter,
        **kwargs: Any,
    ) -> None:
        """Create a new TensorBoard logger.
        On construction, the logger creates a new events file that logs
        will be written to.  If the environment variable `RANK` is defined,
        logger will only log if RANK = 0.

        NOTE: If using the logger with distributed training:
        - This logger can call collective operations
        - Logs will be written on rank 0 only
        - Logger must be constructed synchronously *after* initializing distributed process group.

        Args:
            path (str): path to write logs to
            *args, **kwargs: Extra arguments to pass to SummaryWriter
        """
        self._writer: Optional[SummaryWriter] = None
        _, self._rank = get_machine_local_and_dist_rank()
        self._path: str = path
        if self._rank == 0:
            logging.info(
                f"TensorBoard SummaryWriter instantiated. Files will be stored in: {path}"
            )
            self._writer = summary_writer_method(
                log_dir=path,
                *args,
                filename_suffix=filename_suffix or str(uuid.uuid4()),
                **kwargs,
            )
        else:
            logging.debug(
                f"Not logging meters on this host because env RANK: {self._rank} != 0"
            )
        atexit.register(self.close)

    @property
    def writer(self) -> Optional[SummaryWriter]:
        return self._writer

    @property
    def path(self) -> str:
        return self._path

    def flush(self) -> None:
        """Writes pending logs to disk."""

        if not self._writer:
            return

        self._writer.flush()

    def close(self) -> None:
        """Close writer, flushing pending logs to disk.
        Logs cannot be written after `close` is called.
        """

        if not self._writer:
            return

        self._writer.close()
        self._writer = None


class TensorBoardLogger(TensorBoardWriterWrapper):
    """
    A simple logger for TensorBoard.
    """

    def log_dict(self, payload: Dict[str, Scalar], step: int) -> None:
        """Add multiple scalar values to TensorBoard.

        Args:
            payload (dict): dictionary of tag name and scalar value
            step (int, Optional): step value to record
        """
        if not self._writer:
            return
        for k, v in payload.items():
            self.log(k, v, step)

    def log(self, name: str, data: Scalar, step: int) -> None:
        """Add scalar data to TensorBoard.

        Args:
            name (string): tag name used to group scalars
            data (float/int/Tensor): scalar data to log
            step (int, optional): step value to record
        """
        if not self._writer:
            return
        self._writer.add_scalar(name, data, global_step=step, new_style=True)

    def log_hparams(
        self, hparams: Dict[str, Scalar], meters: Dict[str, Scalar]
    ) -> None:
        """Add hyperparameter data to TensorBoard.

        Args:
            hparams (dict): dictionary of hyperparameter names and corresponding values
            meters (dict): dictionary of name of meter and corersponding values
        """
        if not self._writer:
            return
        self._writer.add_hparams(hparams, meters)


class Logger:
    """
    A logger class that can interface with multiple loggers. It now supports tensorboard only for simplicity, but you can extend it with your own logger.
    """

    def __init__(self, logging_conf):
        # allow turning off TensorBoard with "should_log: false" in config
        tb_config = logging_conf.tensorboard_writer
        tb_should_log = tb_config and tb_config.pop("should_log", True)
        self.tb_logger = instantiate(tb_config) if tb_should_log else None
        wb_config = logging_conf.wandb_writer
        wb_should_log = wb_config and wb_config.pop("should_log", True)
        self.wb_logger = instantiate(wb_config) if wb_should_log else None

    def log_dict(self, payload: Dict[str, Scalar], step: int) -> None:
        if self.tb_logger:
            self.tb_logger.log_dict(payload, step)
        if self.wb_logger:
            self.wb_logger.log_dict(payload, step)

    def log(self, name: str, data: Scalar, step: int) -> None:
        if self.tb_logger:
            self.tb_logger.log(name, data, step)
        if self.wb_logger:
            self.wb_logger.log(name, data, step)

    def log_hparams(
        self, hparams: Dict[str, Scalar], meters: Dict[str, Scalar]
    ) -> None:
        if self.tb_logger:
            self.tb_logger.log_hparams(hparams, meters)
        if self.wb_logger:
            self.wb_logger.log_hparams(hparams, meters)


class WandbLogger:
    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        dir: Optional[str] = None,
        entity: Optional[str] = None,
        tags: Optional[list[str]] = None,
        mode: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        group: Optional[str] = None,
        job_type: Optional[str] = None,
        resume: Optional[str] = None,
        run_id: Optional[str] = None,
        segm_vis: Optional[Dict[str, Any]] = None,
        # ==================== 自定义添加：W&B 指标过滤（仅保留关键指标） ====================
        log_keys: Optional[list[str]] = None,
        log_key_patterns: Optional[list[str]] = None,
        # ==================== 自定义添加结束 ====================
        **kwargs: Any,
    ) -> None:
        _, self._rank = get_machine_local_and_dist_rank()
        self._enabled = self._rank == 0
        self._wandb = None
        self.segm_vis = segm_vis or {}
        # ==================== 自定义添加：W&B 指标过滤（仅保留关键指标） ====================
        self._log_keys = set(log_keys or [])
        self._log_key_patterns = list(log_key_patterns or [])
        # ==================== 自定义添加结束 ====================
        if not self._enabled:
            return

        import wandb

        self._wandb = wandb
        init_kwargs = {
            "project": project,
            "name": name,
            "dir": dir,
            "entity": entity,
            "tags": tags,
            "mode": mode,
            "config": config,
            "group": group,
            "job_type": job_type,
            "resume": resume,
            "id": run_id,
        }
        init_kwargs.update(kwargs)
        self._run = self._wandb.init(**{k: v for k, v in init_kwargs.items() if v is not None})
        atexit.register(self.close)

    def log_dict(self, payload: Dict[str, Scalar], step: int) -> None:
        if not self._enabled or self._wandb is None:
            return
        filtered = self._filter_payload(payload)
        if not filtered:
            return
        self._wandb.log(filtered, step=step)

    def log(self, name: str, data: Scalar, step: int) -> None:
        if not self._enabled or self._wandb is None:
            return
        if not self._should_log_key(name):
            return
        self._wandb.log({name: data}, step=step)

    def log_hparams(self, hparams: Dict[str, Scalar], meters: Dict[str, Scalar]) -> None:
        if not self._enabled or self._wandb is None:
            return
        self._wandb.config.update(hparams, allow_val_change=True)
        if meters:
            self._wandb.log(meters)

    def close(self) -> None:
        if not self._enabled or self._wandb is None:
            return
        self._wandb.finish()

    # ==================== 自定义添加：W&B 指标过滤（仅保留关键指标） ====================
    def _should_log_key(self, key: str) -> bool:
        if not self._log_keys and not self._log_key_patterns:
            return True
        if key in self._log_keys:
            return True
        for pattern in self._log_key_patterns:
            if fnmatch.fnmatch(key, pattern):
                return True
        return False

    def _filter_payload(self, payload: Dict[str, Scalar]) -> Dict[str, Scalar]:
        if not self._log_keys and not self._log_key_patterns:
            return payload
        return {k: v for k, v in payload.items() if self._should_log_key(k)}
    # ==================== 自定义添加结束 ====================

    # ==================== 自定义添加：验证分割可视化（按 epoch 追加，不覆盖） ====================
    def log_coco_segm_samples(
        self,
        *,
        pred_json_path: str,
        gt_json_path: str,
        img_root: str,
        epoch: int,
        step: int,
    ) -> None:
        """
        从 COCO segm 预测文件中，取固定 N 个验证样本渲染 overlay 并上传到 W&B。
        通过每次使用不同的 key（包含 epoch）来避免覆盖之前结果。

        注意：W&B 的 `step` 必须全局单调递增，因此这里的 `step` 用于 W&B 的 x 轴，
        `epoch` 仅用于 key/caption 标识当前 epoch。
        """
        if not self._enabled or self._wandb is None:
            return
        cfg = self.segm_vis or {}
        if not cfg.get("enabled", False):
            return

        num_samples = int(cfg.get("num_samples", 10))
        alpha = float(cfg.get("alpha", 0.5))
        gt_alpha = float(cfg.get("gt_alpha", 0.35))
        score_thresh = float(cfg.get("score_thresh", 0.0))
        show_gt = bool(cfg.get("show_gt", True))
        key_prefix = str(cfg.get("key_prefix", "val/segm_samples"))

        # 固定调色板（最多 10 类，超出则循环）
        palette: List[Tuple[int, int, int]] = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 128, 255),
            (255, 128, 0),
            (128, 0, 255),
            (0, 255, 255),
            (255, 0, 255),
            (128, 128, 0),
            (0, 128, 128),
            (128, 0, 128),
        ]

        try:
            import json

            import numpy as np
            from PIL import Image
            from pycocotools import mask as mask_utils
        except Exception as e:
            logging.warning(f"W&B segm vis: missing deps, skip. err={e}")
            return

        pred_path = Path(pred_json_path)
        gt_path = Path(gt_json_path)
        img_root_path = Path(img_root)

        if not pred_path.exists() or not gt_path.exists():
            logging.warning(
                f"W&B segm vis: pred/gt json not found. pred={pred_path} gt={gt_path}"
            )
            return

        try:
            pred = json.loads(pred_path.read_text())
            gt = json.loads(gt_path.read_text())
        except Exception as e:
            logging.warning(f"W&B segm vis: failed to read json. err={e}")
            return

        img_id_to_info = {img["id"]: img for img in gt.get("images", [])}
        if not img_id_to_info:
            logging.warning("W&B segm vis: empty gt images, skip.")
            return

        # 固定选择：在所有图片的 10 等位索引上取样（分布更均匀、跨 epoch 稳定）
        img_ids_sorted = sorted(img_id_to_info.keys())
        selected_img_ids: List[int] = []
        if img_ids_sorted and num_samples > 0:
            total = len(img_ids_sorted)
            if num_samples == 1:
                indices = [total // 2]
            else:
                indices = [
                    int(round(i * (total - 1) / (num_samples - 1)))
                    for i in range(num_samples)
                ]
            # 去重且保持顺序
            seen = set()
            for idx in indices:
                if idx < 0 or idx >= total:
                    continue
                img_id = img_ids_sorted[idx]
                if img_id in seen:
                    continue
                seen.add(img_id)
                selected_img_ids.append(img_id)
        else:
            selected_img_ids = []

        # predictions by image_id
        pred_by_img: Dict[int, List[Dict[str, Any]]] = {}
        for ann in pred:
            try:
                pred_by_img.setdefault(int(ann["image_id"]), []).append(ann)
            except Exception:
                continue

        gt_by_img: Dict[int, List[Dict[str, Any]]] = {}
        for ann in gt.get("annotations", []) or []:
            try:
                gt_by_img.setdefault(int(ann["image_id"]), []).append(ann)
            except Exception:
                continue

        def to_rgb_gray(pil_img: "Image.Image") -> "np.ndarray":
            arr = np.array(pil_img.convert("L"), dtype=np.uint8)
            return np.stack([arr, arr, arr], axis=-1)

        def overlay_mask(base_rgb: "np.ndarray", mask: "np.ndarray", color: Tuple[int, int, int], a: float) -> "np.ndarray":
            if mask is None:
                return base_rgb
            mask_bool = mask.astype(bool)
            out = base_rgb.copy()
            for c in range(3):
                out[:, :, c] = np.where(
                    mask_bool,
                    (1 - a) * out[:, :, c] + a * color[c],
                    out[:, :, c],
                )
            return out.astype(np.uint8)

        def decode_segmentation(segm, height: int, width: int) -> Optional["np.ndarray"]:
            if segm is None:
                return None
            if isinstance(segm, dict):
                if isinstance(segm.get("counts"), list):
                    rle = mask_utils.frPyObjects(segm, height, width)
                    return mask_utils.decode(rle)
                return mask_utils.decode(segm)
            if isinstance(segm, list):
                rle = mask_utils.frPyObjects(segm, height, width)
                return mask_utils.decode(rle)
            return None

        def collapse_mask(mask: Optional["np.ndarray"]) -> Optional["np.ndarray"]:
            if mask is None:
                return None
            if mask.ndim == 3:
                return (mask.sum(axis=2) > 0).astype(np.uint8)
            return mask.astype(np.uint8)

        def render_overlays(img_path: Path, img_info: Dict[str, Any], anns: List[Dict[str, Any]], a: float) -> "np.ndarray":
            pil_img = Image.open(img_path)
            canvas = to_rgb_gray(pil_img)
            h, w = int(img_info["height"]), int(img_info["width"])
            anns_sorted = sorted(anns, key=lambda x: x.get("score", 0.0), reverse=True)
            for ann in anns_sorted:
                if float(ann.get("score", 0.0)) < score_thresh:
                    continue
                segm = ann.get("segmentation")
                mask = collapse_mask(decode_segmentation(segm, h, w))
                if mask is None:
                    continue
                cat_id = int(ann.get("category_id", 0))
                color = palette[(cat_id - 1) % len(palette)]
                canvas = overlay_mask(canvas, mask, color=color, a=a)
            return canvas

        def render_gt(img_path: Path, img_info: Dict[str, Any], anns: List[Dict[str, Any]], a: float) -> "np.ndarray":
            pil_img = Image.open(img_path)
            canvas = to_rgb_gray(pil_img)
            h, w = int(img_info["height"]), int(img_info["width"])
            for ann in anns:
                segm = ann.get("segmentation")
                mask = collapse_mask(decode_segmentation(segm, h, w))
                if mask is None:
                    continue
                cat_id = int(ann.get("category_id", 0))
                color = palette[(cat_id - 1) % len(palette)]
                canvas = overlay_mask(canvas, mask, color=color, a=a)
            return canvas

        wandb_images: List[Any] = []
        for img_id in selected_img_ids:
            img_info = img_id_to_info.get(img_id)
            if not img_info:
                continue
            img_path = img_root_path / str(img_info["file_name"])
            if not img_path.exists():
                continue

            pred_canvas = render_overlays(img_path, img_info, pred_by_img.get(img_id, []), alpha)
            if show_gt:
                gt_canvas = render_gt(img_path, img_info, gt_by_img.get(img_id, []), gt_alpha)
                try:
                    pred_canvas = np.concatenate([pred_canvas, gt_canvas], axis=1)
                except Exception:
                    pass

            caption = f"epoch={int(epoch)} | {img_info.get('file_name', img_id)}"
            wandb_images.append(self._wandb.Image(pred_canvas, caption=caption))

        if not wandb_images:
            return

        # 关键点：key 带 epoch，保证不覆盖历史结果
        key = f"{key_prefix}/epoch_{int(epoch)}"
        self._wandb.log({key: wandb_images}, step=int(step))
    # ==================== 自定义添加结束 ====================


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    # we tune the buffering value so that the logs are updated
    # frequently.
    log_buffer_kb = 10 * 1024  # 10KB
    io = g_pathmgr.open(filename, mode="a", buffering=log_buffer_kb)
    atexit.register(io.close)
    return io


def setup_logging(
    name,
    output_dir=None,
    rank=0,
    log_level_primary="INFO",
    log_level_secondary="ERROR",
):
    """
    Setup various logging streams: stdout and file handlers.
    For file handlers, we only setup for the master gpu.
    """
    # get the filename if we want to log to the file as well
    log_filename = None
    if output_dir:
        makedir(output_dir)
        if rank == 0:
            log_filename = f"{output_dir}/log.txt"

    logger = logging.getLogger(name)
    logger.setLevel(log_level_primary)

    # create formatter
    FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)4d: %(message)s"
    formatter = logging.Formatter(FORMAT)

    # Cleanup any existing handlers
    for h in logger.handlers:
        logger.removeHandler(h)
    logger.root.handlers = []

    # setup the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if rank == 0:
        console_handler.setLevel(log_level_primary)
    else:
        console_handler.setLevel(log_level_secondary)

    # we log to file as well if user wants
    if log_filename and rank == 0:
        file_handler = logging.StreamHandler(_cached_log_stream(log_filename))
        file_handler.setLevel(log_level_primary)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logging.root = logger


def shutdown_logging():
    """
    After training is done, we ensure to shut down all the logger streams.
    """
    logging.info("Shutting down loggers...")
    handlers = logging.root.handlers
    for handler in handlers:
        handler.close()
