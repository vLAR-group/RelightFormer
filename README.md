<p align="center">
  <h1 align="center">RelightFormer: Feed-Forward Generative Transformer for Multi-View Object Relighting</h1>
  <p align="center">
    <strong>SIGGRAPH Asia 2026</strong>
  </p>
  <p align="center">
    <a href="https://arxiv.org/abs/2609.07414"><img src="https://img.shields.io/badge/arXiv-2604.07414-b31b1b.svg" alt="arXiv"></a>
    <a href="https://huggingface.co/datasets/vLAR/LavalObjaverseDataset"><img src="https://img.shields.io/badge/🤗-Dataset-yellow" alt="Dataset"></a>
    <a href="https://huggingface.co/vLAR/RelightFormer"><img src="https://img.shields.io/badge/🤗-Model-yellow" alt="Model"></a>
    <a href="#license"><img src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg" alt="License"></a>
  </p>
</p>

<p align="center">
  <img src="./demo/teaser.jpg" alt="RelightFormer Teaser" width="100%">
</p>

<p align="center">
  <video src="./demo/video.mp4" controls width="80%" style="border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
    Your browser does not support the video tag.
  </video>
</p>

<p align="center">
  <em>Can't see the video? <a href="./demo/video.mp4">Click here to download and watch</a>.</em>
</p>

---

## 🌟 Overview

**RelightFormer** revolutionizes image relighting by replacing traditional, computationally expensive inverse rendering with a feed-forward generative Transformer. By seamlessly injecting target lighting into spatial features and processing multiple views symmetrically, it delivers highly photorealistic results. Trained on our newly introduced, large-scale open-source multi-view relighting dataset, RelightFormer achieves state-of-the-art quality and remarkable generalization across diverse scenes.

- 🏹 **Feed-Forward Architecture**: No iterative optimization required, enabling rapid generation.
- 🌟 **Multi-View Consistency**: Coherent and physically plausible relighting across all viewpoints.
- ⚡ **Performant Inference**: Highly optimized and expeditious execution on modern GPUs.
- 🎨 **Competitive Quality**: State-of-the-art, photorealistic relighting results.

## 🛠️ Environment Setup

```bash
# clone this repo
git clone git@github.com:vLAR-group/RelightFormer.git
cd RelightFormer

# create and activate conda environment
conda env create -f environment.yaml
conda activate relightformer
```

## 📦 Dataset

We introduce the **Laval-Objaverse Dataset (LOD)**, which comprises **90,545 high-quality 3D assets** from Objaverse and **39,008 diverse illumination conditions** derived from the Laval Indoor and Outdoor HDR datasets. Each render includes synchronized multi-view images, depth maps, and complete lighting metadata.

### 📥 Downloading the Rendering Results

To download the dataset, run the provided script. The results will be saved directly into the `./laval-objaverse-dataset/` directory.

```bash
# Download the testing split (default)
bash ./laval-objaverse-dataset/download.sh testing
```

Alternatively, you can download other specific splits:
```bash
bash ./laval-objaverse-dataset/download.sh training      # Full training set
bash ./laval-objaverse-dataset/download.sh training subset_5 # Subset 5 of the training set
bash ./laval-objaverse-dataset/download.sh validation    # Validation set
bash ./laval-objaverse-dataset/download.sh all           # All splits (training + validation + testing)
```

### 💡 Obtaining Illumination Maps

Due to licensing restrictions, we cannot directly distribute the raw illumination maps. To access the Laval Indoor and Outdoor HDR databases, please follow these steps:

1. Visit the [Laval HDR Database project page](http://hdrdb.com/).
2. Select both the **Laval Indoor HDR database** and the **Laval Outdoor HDR database**.3. Sign the End User License Agreement (EULA) and contact Jean-François Lalonde via the provided email.
4. You will receive a download link for the source archives, namely:
   - `IndoorHDRDatasetReexposedNoRedDotsNoInpaintingOct18.tar`
   - `outdoorPanosExr.tgz`

Once downloaded, place these two files in the `./laval-objaverse-dataset/laval/src/` directory and extract them using the following commands:

```bash
# Extract Indoor dataset
tar -xvf ./laval-objaverse-dataset/laval/src/IndoorHDRDatasetReexposedNoRedDotsNoInpaintingOct18.tar -C ./laval-objaverse-dataset/laval/src/Indoor

# Extract Outdoor dataset (note: use -xzvf for .tgz files)
tar -xzvf ./laval-objaverse-dataset/laval/src/outdoorPanosExr.tgz -C ./laval-objaverse-dataset/laval/src/Outdoor
```

Finally, run the preprocessing script to generate illumination maps compatible with our dataset format:
```bash
python ./laval-objaverse-dataset/scripts/process_exr.py
```
*The processed maps will be saved in `./laval-objaverse-dataset/laval/preprocessed/`.*

### 🎨 Custom Rendering
For researchers who wish to customize the rendering schema, please refer to the detailed instructions in [`RENDERING_INSTRUCTION.md`](https://github.com/vLAR-group/RelightFormer/blob/main/laval-objaverse-dataset/RENDERING_INSTRUCTION.md).

## 🚀 Inference

We have released the pre-trained weights for both **RelightFormer** and **RelightFormer-Post** on the Hugging Face Hub.

You can load and call the model via Python:

```python
from diffsynth import RelightFormerPipeline

from_pretrained = 'vLAR/RelightFormer'
revision = 'main'  # Use 'post' if you would like to load RelightFormer-Post

pipe = RelightFormerPipeline.from_pretrained(
    from_pretrained, 
    revision=revision
)
```

You can also run inference and evaluation on the Laval-Objaverse Dataset via the command line:

**Single GPU:**
```bash
python inference.py \
    --from_pretrained vLAR/RelightFormer \
    --revision main \
    --dataset_path ./laval-objaverse-dataset \
    --output_dir ./output \
    --batch_size 1 \
    --mixed_precision bf16 \
    --skip_exist \
    --save_gt
```

**Multi-GPU (Distributed):**
```bash
accelerate launch --num_processes=4 inference.py \
    --from_pretrained vLAR/RelightFormer \
    --revision main \
    --dataset_path ./laval-objaverse-dataset \
    --output_dir ./output/my_eval \
    --batch_size 1 \
    --mixed_precision bf16 \
    --skip_exist
```
*(Note: Use `--revision post` in the commands above to evaluate the post-trained model).*

## 🏋️ Training

### 1. Download Base Model (Wan 2.1)
RelightFormer is fine-tuned from Wan 2.1. Please use the following script to download the base Wan 2.1 weights first:
```bash
python download_wan2.1.py
```

### 2. Training RelightFormer
Once you have downloaded the full training split of the LOD dataset into `./laval-objaverse-dataset`, you can launch the training on 4× H200 GPUs:
```bash
nohup accelerate launch --main_process_port 25523 \
    --config_file configs/accelerate/4_16fp.yaml \
    train_diffusion.py --config configs/main/config.yaml \
    > logs/main/training.log 2>&1 &
```
Upon completion, you will find the trained **RelightFormer** weights under `./models/main/relightformer/checkpoint-80000`.

### 3. Post-Training RelightFormer
To initialize post-training, copy the main checkpoint to the post-training directory:
```bash
cp -r ./models/main/relightformer/checkpoint-80000 ./models/post/relightformer/checkpoint-80000
```

Then, launch the post-training process to obtain **RelightFormer-Post**:
```bash
nohup accelerate launch --main_process_port 25523 \
    --config_file configs/accelerate/4_16fp.yaml \
    train_diffusion.py --config configs/main/config-post.yaml \
    > logs/main/post-training.log 2>&1 &
```

## 📜 License

This work, including the code and dataset, is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/) (CC BY-NC-SA 4.0).

## 🙏 Acknowledgements

This work was supported in part by the National Natural Science Foundation of China under Grant 62271431; in part by the Research Grants Council of Hong Kong under Grants 15219125, 15228626, and 15225522; in part by the Otto Poon Charitable Foundation Smart Cities Research Institute (8-CDCQ); in part by the Research Center for Unmanned Autonomous Systems (1-CE3D); and in part by the PolyU Kunpeng & Ascend Technology Innovation Incubation Center, The Hong Kong Polytechnic University.
