# README: Adapting SAM3 to Cardiac MRI Segmentation

## Overview
This project aims to adapt the SAM3 model to the task of cardiac MRI segmentation, specifically using the ACDC dataset. We will follow these steps:

1. **Set up the environment and run SAM3 examples** to ensure everything works.
2. **Prepare cardiac MRI data** and organize it for input.
3. **Run a baseline inference** with SAM3 on the MRI data before any adaptation.
4. **Adapt SAM3 for the cardiac MRI task**, making necessary modifications and improvements.

## Step 1: Environment Setup and Running Examples

First, install the necessary dependencies and make sure the SAM3 code is working in your environment. Then run the official examples to confirm everything is set up properly.

- **Install Dependencies**: Follow the instructions in the SAM3 repository to set up the Python environment (e.g., `pip install -r requirements.txt`).
- **Local Install (Recommended)**: Install the local SAM3 package and notebook deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e /data/home/shujia/CardioSAM/sam3-main
pip install -e "/data/home/shujia/CardioSAM/sam3-main[notebooks]"
pip install nibabel Pillow
```

- **Hugging Face Checkpoint Access**:
  - Request access to the SAM3 checkpoints on Hugging Face.
  - Login once with `hf auth login` so the model can download weights.

- **Run Image Example**: Use the `sam3_image_predictor_example.ipynb` notebook to run an image-level example and verify single-frame inference works.

- **Run Video Example**: Use the `sam3_video_predictor_example.ipynb` notebook to run a video-level example and verify that multi-frame tracking works.

After this step, you should have a working baseline and know that the SAM3 model can run on your machine.

## Step 2: Prepare Cardiac MRI Data

Next, we need to prepare the input data. For cardiac MRI (using the ACDC dataset), we will convert the MRI volumes into a format that the SAM3 model can ingest.

- **Convert Data**: Write a script to load the ACDC dataset (NIfTI format) and export ED/ES frames as individual images (e.g., PNG/JPG).
- **Script (Done)**: Use `scripts/prepare_acdc.py` to export ED/ES slices and cine slice videos:

```bash
python /data/home/shujia/CardioSAM/scripts/prepare_acdc.py \
  --acdc_root /path/to/ACDC \
  --output_root /data/home/shujia/CardioSAM/data/acdc_prepared \
  --patients patient001 \
  --slice_stride 1 \
  --max_slices 5
```

This will generate:
- `data/acdc_prepared/<split>/<patient_id>/images/ED|ES/*.png`
- `data/acdc_prepared/<split>/<patient_id>/videos/slice_XX/frame_YY.png`
- `data/acdc_prepared/<split>/<patient_id>/meta.json`

- **Organize Folders**: Create a directory structure where each patient’s cine MRI sequence can be stored as a series of frames. We want to ensure that SAM3 can treat a single slice over time as a “video” input.

- **Baseline Input Preparation**: Prepare a few sample slices and frames so we can run the SAM3 model on them as a baseline test.

## Step 3: Run Baseline Inference

Now that we have the data prepared, we will run the SAM3 model on the cardiac MRI data without any adaptation. This will give us a baseline result.

- **Run Baseline**: Use the prepared MRI frames as input to the SAM3 video predictor. Run inference on a chosen slice’s full cine sequence and on ED/ES frames for static evaluation.
- **Image Baseline (ED/ES slices)**:

```bash
python /data/home/shujia/CardioSAM/scripts/run_baseline_image.py \
  --image_dir /data/home/shujia/CardioSAM/data/acdc_prepared/train/patient001/images/ED \
  --prompt "left ventricle" \
  --output_dir /data/home/shujia/CardioSAM/outputs/baseline_image
```

- **Video Baseline (cine slice)**:

```bash
python /data/home/shujia/CardioSAM/scripts/run_baseline_video.py \
  --video_path /data/home/shujia/CardioSAM/data/acdc_prepared/train/patient001/videos/slice_00 \
  --prompt "left ventricle" \
  --output_dir /data/home/shujia/CardioSAM/outputs/baseline_video \
  --max_frames 20
```

- **Generate Initial Results**: Record the initial segmentation results and note any obvious issues or domain gaps.

## Step 4: Adapt SAM3 for Cardiac MRI

In this final step, we will modify and adapt the SAM3 model to better handle cardiac MRI data. This may involve adding a lightweight adapter layer, fine-tuning on a small subset of the data, or introducing a domain-specific pre-processing step.

- **Implement Adaptation**: Add code to fine-tune or adapt the SAM3 model for cardiac MRI. This could be done using a small number of labeled examples from the ACDC dataset.

- **Run Adapted Model**: After adaptation, evaluate on a held-out subset and compare against the baseline outputs.
