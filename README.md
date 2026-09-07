<p align="center">
  <h1 align="center">RelightFormer: Feed-forward Generative Transformer for Multiview Object Relighting</h1>
  <p align="center">
    <strong>Siggraph Asia 2026</strong>
  </p>
  <p align="center">
    <a href="https://huggingface.co/datasets/vLAR/LavalObjaverseDataset"><img src="https://img.shields.io/badge/🤗-Dataset-yellow" alt="Dataset"></a>
    <a href="#license"><img src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg" alt="License"></a>
  </p>
</p>

<p align="center">
  <img src="./demo/teaser.jpg" alt="RelightFormer Teaser" width="100%">
</p>

<!-- ## 🎥 Demo -->

<p align="center">
  <video src="./demo/video.mp4" controls width="80%" style="border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
    Your browser does not support the video tag.
  </video>
</p>

<p align="center">
  <em>Can't see the video? <a href="./demo/video.mp4">Click here to download and watch</a>.</em>
</p>

---

## Overview

**RelightFormer** achieves image relighting by replacing traditional inverse rendering with a feedforward generative Transformer. By injecting target lighting into spatial features and processing multiple views symmetrically, it delivers photorealistic results. Trained on **our new, largest open-source multi-view relighting dataset**, it achieves top quality and remarkable broad scene generalization. 

- 🏹 **Feed-forward Architecture**: No iterative optimization required
- 🌟 **Multiview Consistency**: Coherent relighting across all viewpoints  
- ⚡ **Perfomant Inference**: Expeditious execution on modern GPUs
- 🎨 **Competitive Quality**: Photorealistic relighting results

## Environment Setup

Comming Soon.

## 📦 Dataset

We provide the **Laval-Objaverse Dataset**, which comprises **90,545 high-quality 3D assets** from Objaverse and **39,008 diverse illumination conditions** from the Laval Indoor and Outdoor HDR datasets. Each render includes synchronized multi-view images, depth maps, and complete lighting metadata.

### 📥 Downloading the Rendering Results

To download the dataset, run the provided script. The results will be saved directly into the `./laval-objaverse-dataset/` directory.

```bash
chmod +x ./laval-objaverse-dataset/download.sh

# Download the testing split (default)
./laval-objaverse-dataset/download.sh testing

# Alternatively, download other splits:
# ./laval-objaverse-dataset/download.sh training   # for the training set
# ./laval-objaverse-dataset/download.sh validation # for the validation set
# ./laval-objaverse-dataset/download.sh all        # for all splits (training + validation + testing)
```

### 💡 Obtaining Illumination Maps

Due to licensing restrictions, we cannot directly distribute the raw illumination maps. To access the Laval Indoor and Outdoor HDR databases, please follow these steps:

1. Visit the [Laval HDR Database project page](http://hdrdb.com/).
2. Select both the **Laval Indoor HDR database** and **Laval Outdoor HDR database**.
3. Sign the End User License Agreement (EULA) and contact Jean-François Lalonde via the provided email.
4. You will promptly receive a download link for the source archives, namely:
   - `IndoorHDRDatasetReexposedNoRedDotsNoInpaintingOct18.tar`
   - `outdoorPanosExr.tgz`

Once downloaded, place these two files in the `./laval-objaverse-dataset/laval/src/` directory and extract them using the following commands:

```bash
# Extract Indoor dataset
tar -xvf ./laval-objaverse-dataset/laval/src/IndoorHDRDatasetReexposedNoRedDotsNoInpaintingOct18.tar -C ./laval/src/Indoor

# Extract Outdoor dataset (note: use -xzvf for .tgz files)
tar -xzvf ./laval-objaverse-dataset/laval/src/outdoorPanosExr.tgz -C ./laval/src/Outdoor
```

Finally, run the preprocessing script:
```bash
python ./laval-objaverse-dataset/scripts/process_exr.py
```
The processed illumination maps compatible with our dataset will be generated and saved in `./laval-objaverse-dataset/laval/preprocessed/`.

### Rendering

For those who would like to customize the rendering schema, it is encouraged to read the instruction in `./laval-objaverse-dataset/RENDERING_INSTRUCTION.md/`

## Code

**Code coming soon!** Stay tuned for updates.



## License

This work is licensed under CC BY-NC-SA 4.0.


## Acknowlegements

This work was supported in part by National Natural Science Foundation of China under Grant 62271431, in part by Research Grants Council of Hong Kong under Grants 15219125 & 15228626 & 15225522, in part by Otto Poon Charitable Foundation Smart Cities Research Institute (8-CDCQ), in part by Research Center for Unmanned Autonomous Systems (1-CE3D), and in part by PolyU Kunpeng & Ascend Technology Innovation Incubation Center, The Hong Kong Polytechnic University.
