# [Online-InReaCh: Online Inter-Realization Channels for Unsupervised Image Anomaly Detection](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07868.pdf)


**Online-InReaCh** is the **first fully unsupervised, online anomaly detection and localization method** that dynamically adapts to **non-stationary image distributions**. Unlike prior methods, it does not require a fixed, curated, and purely nominal training set or supervision — it builds and updates its nominal model on-the-fly during inference.

---

## 🔍 Paper Summary

> *"We propose Online-InReaCh, the first fully unsupervised on-
line method for detecting and localizing anomalies on-the-fly in image sequences while following non-stationary distributions. Previous anomaly
detection methods are limited to supervised one-class classification or are unsupervised but still pre-compute their nominal model. Online-InReaCh can operate online by dynamically maintaining a nominal model of commonly occurring patches that associate well across image realizations of the underlying nominal distribution while removing stale previously nominal patches. Online-InReaCh, while competitive in previous offline benchmarks, also achieves 0.936 and 0.961 image- and pixel-wise AUROC when tested online on MVTecAD, where 23.8% of all randomly sampled images contain anomalies. Online-InReaCh’s performance did not correlate with anomaly proportion even to 33.5%. We also show that Online-InReaCh can integrate new nominal structures and distinguish anomalies after a single frame, even in the worst-case distribution shift from one training class to a new previously unseen testing class."*

📰 **Title:** Unsupervised, Online and On-The-Fly Anomaly Detection for Non-Stationary Image Distributions

📚 **Authors:** Declan McIntosh, Alexandra Branzan Albu

📄 **PDF:** [View Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07868.pdf)

---

## ✨ Key Features

* ✅ Fully **unsupervised** — no curated nominal training set needed
* 🔄 **Online updates** — adapts model in real time with no re-training
* 🌩️ Handles **non-stationary data** — robust to domain and distribution shifts
* 🔍 Supports **on-the-fly inference** — per-frame predictions in streamed data
* 🚀 Competitive with state-of-the-art methods like SoftPatch and PatchCore in offline setting

---

## 🏗️ Code Structure

```
├── OnlineInReaCh.py         # Main class for online anomaly detection
├── FeatureDescriptors.py    # Feature extraction using Wide-ResNet50 (required)
├── model.py, utils.py       # Supporting modules (required)
├── mvtec_loader.py          # MVTec dataset loader with corruption support
├── qual/                    # Output directory for qualitative results
├── Experiments/             # Saved results and config logs
```

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install torch torchvision scikit-learn numpy opencv-python faiss-gpu tqdm
```

> ✅ Requires **CUDA ≥ 10.2** for full determinism

### 2. Run Optimal Online Detection on MVTec

```bash
python OnlineInReaCh.py -ttl 40 -minl 2 -seed 0 -data_dir data/MVTecAD/ -n DEMO_RUN"
```

This runs Online-InReaCh with:

* `Time-to-Live = 40`
* `Min Channel Length = 2`
* `Seed = 0`
* Dataset path: `data/MVTecAD/`



### 3. Custom Run

```bash
python OnlineInReaCh.py \
    -ttl 20 \
    -minl 2 \
    -seed 0 \
    -data_dir data/MVTecAD/ \
    -n my_experiment_name
```

| Argument      | Type  | Description                                                                |
| ------------- | ----- | -------------------------------------------------------------------------- |
| `-ttl`        | `int` | Time-to-live for channels (e.g., `40`)                                     |
| `-minl`       | `int` | Minimum channel length (e.g., `2`)                                         |
| `-seed`       | `int` | Random seed (e.g., `112358`)                                               |
| `-data_dir`   | `str` | Path to the dataset root (e.g., `data/MVTecAD/`)                           |
| `-n`          | `str` | Name of the experiment for saved results and output (e.g., `my_run_name`)  |
| `-corr` (opt) | `int` | Number of corrupted images to inject (default: `0` or unused)              |
| `-tur`  (opt) | `int` | Test update rate — controls update frequency after training (default: `1`) |


---

## 📊 Results (Online Setting)

This code has been significantly refactored since the original publication. The results from this code-base exceeds the reported online results for MvTec AD, seen below.  

| Dataset  | Image AUROC | Pixel AUROC | 
| -------- | ----------- | ----------- | 
| MvTec AD | 0.942       | 0.964       | 


The above is gained with the following configuration. 

```bash
python OnlineInReaCh.py \
    -ttl 40 \
    -minl 2 \
    -seed 0 \
    -data_dir data/MVTecAD/ \
    -n my_experiment_name
```

> Full results and logs are saved in `Experiments/<commit_hash>/<seed>/<experiment_name>/`.

---

## 📸 Visualizations

Output confidence maps are saved to:

```
qual/<experiment_name>/<class_name>/
```

---

## 📁 Dataset Format

Each class directory should follow the MvTec AD format and contain:

```
data/MVTecAD/
└── class_name_here/
    ├── ground_truth/
    │   └── test/
    │       └── corruption_type/
    │           ├── 000.png        # Binary mask (0 = nominal, 255 = anomaly)
    │           └── ...
    ├── test/
    │   └── good/
    │       ├── 000.png            # Test image without anomaly or corresponding gt mask
    │       └── ...
    │   └── corruption_type/
    │       ├── 000.png            # Test image with anomaly
    │       └── ...
    ├── train/
        └── good/
            ├── 000.png            # Nominal training image
            └── ...
```

---

## 📘 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{mcintosh2024onlineinreach,
  title={Unsupervised, Online and On-The-Fly Anomaly Detection For Non-Stationary Image Distributions},
  author={Declan McIntosh and Alexandra Branzan Albu},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```

---

## 🛠️ TODO

* Nothing at the Moment!

---

