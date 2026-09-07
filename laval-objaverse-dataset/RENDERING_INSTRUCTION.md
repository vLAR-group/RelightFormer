<p align="center">
  <h1 align="center">Rendering Instructions for the Laval Objaverse Dataset</h1>
  <p align="center">
    <strong>vLAR Group</strong>
  </p>
</p>

Welcome to the rendering pipeline for the **Laval Objaverse Dataset**. We highly encourage secondary creation, adaptation, and further modification of this dataset. This document outlines the exact pipeline we used to generate the dataset and will guide you through reproducing it step-by-step.

> **Note:** Please ensure your terminal's working directory is set to this folder before executing the commands below.

---

### 📋 Prerequisites & Setup

#### 1. Laval Indoor/Outdoor Database
Before proceeding, you must fetch the source files for the Laval Indoor/Outdoor Database and preprocess them. Please follow the specific preprocessing instructions provided in that repository's `README.md`.

#### 2. Blender Installation
The dataset images are rendered using **Blender**. For exact reproducibility, we used **Blender 4.3.2 (Linux x64)**.  
🔗 [Download Blender 4.3.2](https://www.blender.org/download/release/Blender4.3/blender-4.3.2-linux-x64.tar.xz) *(or select your OS equivalent from the [official Blender website](https://www.blender.org/download/))*.

#### 3. Hugging Face CLI
Ensure you have the Hugging Face CLI installed to download the metadata. If you haven't installed it yet, you can do so via pip:
```bash
pip install -U "huggingface_hub[cli]"
```
*(Don't forget to authenticate with `huggingface-cli login` if the repository requires it).*

---

### 🚀 Reproduction Pipeline

#### Step 1: Download Metadata & Configuration
Fetch the relevant metadata and object-environment pairing information from our Hugging Face repository:

```bash
mkdir -p info
hf download vLAR/LavalObjaverseDataset "info/*" --local-dir ./info
```

#### Step 2: Launch the Rendering Process
Once the prerequisites are met and the environment is configured, launch the automated rendering pipeline by running in default setting (10 GPUs, 16 workers per GPU, all splits):

```bash
bash ./blender/render.sh --blender path/to/your/blender
```

If you would like to render solely the testing split with 2 specific GPUs (e.g. GPU-0, GPU-1), with 8 worker each, and your path to blender is `./blender/blender-4.3.2-linux-x64/blender`:

```bash
bash ./blender/render.sh \
  --blender ./blender-4.3.2-linux-x64/blender \
  --split testing \
  --gpus 0 1 \
  --workers 8
```

> **⚡ Parallel Rendering Optimization:**  
> The rendering script is designed for high-throughput, parallel execution. To significantly accelerate dataset generation, you can configure the pipeline to run across multiple devices (e.g., **10 GPUs**) while spawning multiple subprocesses per device (e.g., **16 subprocesses**).  
> *Please review the configuration variables at the top of `./blender/render.sh` to adjust GPU allocation and subprocess counts to match your specific hardware setup.*

---

### 💡 Support & Troubleshooting
If you encounter any issues, have questions about the pipeline, or wish to suggest improvements, please feel free to [open an issue](../../issues) in this repository. We are happy to help!