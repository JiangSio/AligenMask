# AliGen

**AliGen: Benchmarking and Advancing Few-Shot Industrial Anomaly Generation under Object-Level Misalignment** (BMVC 2026)

> Official code for the paper. AliGen is a **plug-and-play framework** that adaptively generates *spatially plausible anomaly masks* for arbitrary normal images, so that existing few-shot anomaly generation (FSAG) methods work reliably under object-level misalignment (position / scale / quantity variations).

[Paper (BMVC 2026 camera-ready)](https://github.com/JiangSio/AligenMask)

## News / Highlight

- We release **MIAD**, the first **Misaligned Industrial Anomaly Dataset**, which jointly models **position**, **scale** and **quantity** misalignment.
- We release **AliGen**, which improves the generation / detection / localization performance of existing FSAG methods such as **AnomalyDiffusion (AAAI 2024)**, **AnoGen (ECCV 2024)** and **FAST (NeurIPS 2025)** in a plug-and-play manner.

---

## Overview

Most existing FSAG methods assume **well-aligned** objects (centered, uniformly scaled, consistently oriented) and rely on predefined masks or few-shot reference templates to place the anomaly.

In real industrial imaging, however, objects are frequently **misaligned** — their position, scale, and quantity vary from image to image. When such methods are applied to misaligned data, the anomaly is often generated in an **implausible location** (e.g., in the background or on the wrong object part), which hurts both generation realism and downstream anomaly detection.

AliGen addresses this by predicting an adaptive anomaly mask for each normal image:

1. **Anomaly Marker** — a texture-agnostic spatial representation that encodes *object structure + position* (via edge maps) together with *anomaly locations* (filled with a solid color). It is built from the edge map of the image and the anomaly region, i.e.

   ```
   I'_a = C(I_a; σ) ⊙ (1 - M_a) + Color(r, g, b) ⊙ M_a
   ```

2. **Structure-Aware Diffusion Model (SADM)** — a latent diffusion model (LDM) trained (with LoRA) on Anomaly Markers to relate anomaly locations with object spatial context. Training uses a generation loss + a **mask-supervised alignment loss** on cross-attention maps:

   ```
   L_train = L_gen + λ · L_align
   ```

3. **Latent Stochastic Differential Sampling (LSDS)** — during inference, the initial denoising latent is noised to a random forward step within a sampling interval `[T_low, T_high]` (e.g., 600–840). This strikes a balance between **spatial-structure consistency** with the input normal image and **stable, diverse anomaly-region generation**.

The generated Anomaly Marker is converted into a mask by simple color thresholding, and the mask is fed together with the normal image into an existing **inpainting-based FSAG method** (AnomalyDiffusion / AnoGen / FAST) to synthesize the final anomaly image.

---

## MIAD Dataset

**Misaligned Industrial Anomaly Dataset (MIAD)** is a challenging benchmark proposed alongside AliGen.

| Dataset | #Cat | Misaligned | Image Number | Position | Scale | Quantity |
|---------|------|------------|--------------|----------|-------|----------|
| MVTec AD | 15 | 1  | 4096 normal / 1258 anomaly / 5354 all | ✓ | ✗ | ✗ |
| VisA     | 12 | 2  | 9621 normal /  1200 anomaly / 10821 all | ✓ | ✗ | ✗ |
| **MIAD** | **20** | **20** | **10288 normal / 1905 anomaly / 12193 all** | **✓** | **✓** | **✓** |

- Built on top of **MVTec AD**, **VisA**, and **MPDD** (20 subsets, 53 anomaly types).
- Contains **4 real misaligned categories** (capsules, macaroni, screw_single, tubes) plus many categories **synthesized from real aligned data**.
- Construction: foreground objects are resized / rotated / translated and dropped onto a background extracted from normal samples (objects removed via *Gemini 2.5 Flash Image* + manual cleanup), then fused with **Poisson blending**; pixel-level masks are derived by applying the same foreground transform to the ground-truth masks.

### Download

The dataset is released on Hugging Face:

> **JiangSio/MisalignedIndustryAnomalyDataset**

Download it via `huggingface-cli` or `git-lfs`:

```bash
# Option 1: huggingface-cli
pip install -U huggingface_hub
huggingface-cli download JiangSio/MisalignedIndustryAnomalyDataset \
    --local-dir ./datasets

# Option 2: git-lfs
git lfs install
git clone https://huggingface.co/datasets/JiangSio/MisalignedIndustryAnomalyDataset ./datasets
```

After downloading, set `mvtec_path` in `run_mvtec.py` to the local dataset folder.

---

## Repository Structure

```
├── run_mvtec.py                    # End-to-end pipeline orchestration (train SADM → infer markers → extract masks → FSAG → localization)
├── requirements.txt                # Python / PyTorch dependencies
├── name-list.txt                  # Category list of MIAD
├── name-list-gen.py               # Helper to regenerate the category list from a data root
├── rename.py                      # Helper to rename output/log dirs
│
├── DualAnoDiff/                   # DualAnoDiff codebase (upstream)
│   └── dual-interrelated_diff/    # ── AliGen core components ──
│       ├── train_dreambooth_lora.py    # Train SADM (LoRA on SD-v1-5) on Anomaly Markers
│       ├── inference_mvtec_split.py    # Generate Anomaly Markers via LSDS
│       ├── process.py                  # Extract anomaly masks from generated markers
│       ├── remove_no_mask_samples.py   # Drop samples with no valid mask
│       ├── diffusers/                  # Local diffusers (modified SD pipeline, Canny support)
│       └── pil2canny/                  # PiDiNet-based edge detection (Anomaly Marker context)
│
├── anomalydiffusion/              # AnomalyDiffusion (AAAI 2024) codebase (inpainting-based FSAG)
│   ├── main.py                        # Train anomaly generation model (LDM + textual inversion)
│   ├── generate_mask.py               # Generate masks
│   ├── generate_with_mask.py          # Generate anomaly image-mask pairs from masks (+ adaptive attention reweighting)
│   ├── run-mvtecgeneratematching1.py  # Matching-generation entry used by run_mvtec.py
│   ├── train_mask.py                  # Train the mask generation model
│   ├── train-localization.py / train-classification.py
│   ├── test-localization.py  / test-classification.py
│   └── unet_utils/                    # U-Net localization model, losses, AU-PRO metrics
│
├── fast-foreground-aware-anomaly-synthesis/   # FAST (NeurIPS 2025) codebase
│   └── main.py, generate_with_mask_mvtec.py, train.sh, ...
│
├── anogen/                        # AnoGen (ECCV 2024) codebase
│
├── anomaly_metrics/               # Downstream anomaly localization training/eval (U-Net)
│   └── train-localization.py           # Train the localization U-Net on generated data
│
└── anomaly_gen_metrics/           # Generation quality metrics
    ├── fid.py                          # FID
    └── iclpips.py                      # IC-LPIPS (intra-cluster pairwise LPIPS)
```

---

## Installation

```bash
# Create a conda environment (Python 3.10)
conda create -n aligen python=3.10
conda activate aligen

# Install dependencies
pip install -r requirements.txt
```

> **Third-party checkpoints you may need:**
> - The base LDM checkpoint `models/ldm/text2img-large/model.ckpt` (from *ommer-lab*).
> - Pretrained Stable Diffusion (runwayml/stable-diffusion-v1-5) for the marker model (downloaded automatically via 🤗 Hub).
> - A PiDiNet (or similar) edge-detection checkpoint for building Anomaly Markers.

---

## Quick Start

### Option A — End-to-end pipeline (all categories of MIAD)

```python
# (1) Set the correct paths inside run_mvtec.py, e.g.:
#     mvtec_path        = <path>/mvtec+visa
#     dualanodiff_path  = <path>/AligenMask/DualAnoDiff/dual-interrelated_diff
#     anomalydiffusion_path = <path>/AligenMask/anomalydiffusion
#     testmodel_path    = <path>/AligenMask/anomaly_metrics
# (2) Run the orchestrator (it generates bash scripts under train_shells/ and executes them)
python run_mvtec.py
```

`run_mvtec.py` performs, for every category and every anomaly type:

1. **Train SADM** on Anomaly Markers:
   ```bash
   CUDA_VISIBLE_DEVICES={id} accelerate launch --main_process_port=30005 train_dreambooth_lora.py \
       --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
       --instance_data_dir {mvtec_path} \
       --output_dir generate_data/{name}/{anomaly} \
       --instance_prompt "a photo of hazelnut" --resolution 512 \
       --train_batch_size 2 --learning_rate 5e-5 --max_train_steps 2000 \
       --rank 32 --seed 32 --train_text_encoder --attn_loss_weight 0.1 \
       --mvtec_name {name} --mvtec_anamaly_name {anomaly}
   ```

2. **Generate Anomaly Markers via LSDS**:
   ```bash
   python inference_mvtec_split.py {name} {anomaly} {mvtec_path}
   ```

3. **Extract anomaly masks** from the markers:
   ```bash
   python process.py {name} {anomaly}
   ```

4. **Generate anomaly image–mask pairs** with the FSAG method:
   ```bash
   cd ../anomalydiffusion
   python run-mvtecgeneratematching1.py --gpu_id={id} \
       --data_path {dualanodiff_path}/generate_data \
       --clssname {name} --anomalyname {anomaly}
   ```
   > For *texture* anomalies (e.g., color on wood) keep `--adaptive_mask` (adaptive attention reweighting); for *structural* anomalies (e.g., squeeze on capsule) remove it.

5. **Train a localization U-Net** on the generated data:
   ```bash
   cd ../anomaly_metrics
   python train-localization.py \
       --generated_data_path {anomalydiffusion_path}/generated_matched_dataset \
       --mvtec_path {mvtec_path} --suffix=jpg \
       --name=Anomalydiffusion+aligenEdge --epochs=100 \
       --clss_name {name} --bs 16 --lr 0.0001
   ```

### Option B — Train / generate a single category manually

Follow the same steps above but adapt the `{name}` / `{anomaly}` arguments for the single category you care about (the paper uses one model per category).

---

## How to use AliGen as a plugin for a new FSAG method

1. Build **Anomaly Markers** from your few-shot anomaly image–mask pairs
   (edge detection for the context + solid-color fill for the anomaly region).
2. Train **SADM** with LoRA on the markers (see `train_dreambooth_lora.py`).
3. At inference, feed a normal image and run **LSDS** (`inference_mvtec_split.py`) to obtain a marker, then extract the mask (`process.py`).
4. Pass the extracted mask together with the normal image into any **inpainting-based FSAG** method.

---

## Evaluation

| Metric | Purpose | Script |
|--------|---------|--------|
| FID | Generation realism | `anomaly_gen_metrics/fid.py` |
| IC-LPIPS | Generation diversity | `anomaly_gen_metrics/iclpips.py` |
| AUROC (pixel / image) | Detection & localization | `anomaly_metrics/train-localization.py` + `unet_utils/au_pro_util.py` |
| Failure Rate / Mask Acc. | Mask stability & spatial consistency | (ablation, defined in the paper) |

---

## Main Results (MIAD)

Plugging AliGen into existing FSAG methods consistently improves both generation and detection:

| Method | FID ↓ | AUC-P ↑ | AUC-I ↑ |
|--------|-------|---------|---------|
| AnomalyDiffusion | 100.9 | 94.87 | 84.55 |
| **+ AliGen** | **94.38** | **96.24** | **86.76** |
| AnoGen | 74.93 | 93.79 | 84.96 |
| **+ AliGen** | **64.39** | **96.38** | **87.05** |
| FAST | 47.16 | 93.47 | 85.34 |
| **+ AliGen** | **41.75** | **95.41** | **86.18** |

AliGen also preserves or improves performance on texture / well-aligned subsets of MVTec AD, demonstrating its strong generalization.

---

## Notes

- Paths inside `run_mvtec.py` and the various scripts are **hard-coded** (Linux-style, e.g., `/data1/...`). Adjust them for your environment.
- The paper's final settings: Anomaly Marker color `(255, 0, 0)`, loss balance `λ = 0.1`, mask tolerance `τ = 0.5`, LSDS sampling interval `[600, 840]`, and **500 anomaly image–mask pairs per anomaly type**.
- Code originally developed for a Linux / CUDA environment (compile extensions such as `upfirdn2d`, `ninja`, custom `diffusers`).

---

## Disclaimer

This repository bundles several **third-party upstream codebases** (AnomalyDiffusion, DualAnoDiff, FAST, AnoGen, etc.) with modifications for the AliGen experiments. Respect the licenses of the respective upstream projects when redistributing.

## Citation

If you find the paper, dataset, or code useful, please consider citing:

```bibtex
@inproceedings{jiang2026aligen,
  title={AliGen: Benchmarking and Advancing Few-Shot Industrial Anomaly Generation under Object-Level Misalignment},
  author={Jiang, Tianjia and Tao, Guangpin and Yuan, Minglei and Tai, Ying and
          Zou, Shirong and Wang, Yixuan and Yan, Ziyang and Lu, Tong},
  booktitle={British Machine Vision Conference (BMVC)},
  year={2026}
}
```

## Contact

For issues / questions, please open a GitHub Issue or contact the authors at `522024330029@smail.nju.edu.cn` (Tianjia Jiang).