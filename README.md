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

## 📦 Dataset & Code

**Code and dataset coming soon!** Stay tuned for updates.

## License

This work is licensed under CC BY-NC-SA 4.0.


## Acknowlegements

This work was supported in part by National Natural Science Foundation of China under Grant 62271431, in part by Research Grants Council of Hong Kong under Grants 15219125 & 15228626 & 15225522, in part by Otto Poon Charitable Foundation Smart Cities Research Institute (8-CDCQ), in part by Research Center for Unmanned Autonomous Systems (1-CE3D), and in part by PolyU Kunpeng & Ascend Technology Innovation Incubation Center, The Hong Kong Polytechnic University.
