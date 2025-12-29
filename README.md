# 📜 AncientVision-3T: A Hierarchical Benchmark for Ancient Chinese Document Image-Text Tasks

<div align="center">

[![Dataset](https://img.shields.io/badge/HuggingFace-Images-yellow)](https://huggingface.co/datasets/Myralala/AncientVision-3T)
[![Benchmark](https://img.shields.io/badge/Benchmark-Results-success)](#-leaderboard)
[![Interpretability](https://img.shields.io/badge/Method-Interpretability-blueviolet)](#-interpretability-ac-tcgn)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>







## 📖 Introduction

**AncientVision-3T** is a hierarchically structured benchmark that enables fine-grained analysis of VLM capabilities across OCR, image classification, and image understanding tasks of increasing abstraction. 

Despite the rapid evolution of VLMs, applying general visual models to ancient manuscripts presents a **"dual chasm"**:
1.  **Visual Perception:** Challenges include vertical text flow, complex layouts, and degradation noise caused by temporal erosion.
2.  **Linguistic Comprehension:** The abstruse grammar and archaic glyphs necessitate deep linguistic priors.

Distinct from existing general benchmarks, AncientVision-3T employs a **hierarchical task design** to decouple and analyze model capabilities across different levels of task complexity.

<div align="center">
  <img src="figure/figure1.jpg" width="80%" alt="AncientVision-3T Framework">
  <br>
  <em>Figure 1: Overview of the AncientVision-3T Benchmark and the Proposed Analysis Framework.</em>
</div>

## 📂 Dataset Statistics

The dataset is curated from public digital archives such as the *Siku Quanshu* and local gazetteers, covering the Song, Yuan, Ming, and Qing dynasties. It comprises **1,500 images** divided into two subsets:

| Subset | Count | Content | Target Task |
| :--- | :---: | :--- | :--- |
| **Textual Images** | 500 | Traditional Chinese text with vertical/left-to-right layouts | **OCR Perception** |
| **Illustrative Images** | 1,000 | Cultural subjects (e.g., ritual vessels, musical instruments, attire) Multiple-Type Illustrations| **Image Classification (IC)** & **Image Understanding (IU)** |

## 🎯 Tasks

We evaluate models across three progressive dimensions:

1.  **Visual Symbol Perception (OCR)**
    * **Metric:** Normalized Edit Distance (NED).
    * **Goal:** Accurate recognition of archaic character forms amidst visual noise.
2.  **Cross-Modal Image Classification (IC)**
    * **Metric:** Accuracy.
    * **Goal:** Identifying artifact categories requiring historical knowledge alignment.
3.  **Cross-Modal Image Understanding (IU)**
    * **Metric:** ROUGE-L.
    * **Goal:** Generating semantic descriptions that bridge visual features with deep linguistic priors.

## 🏆 Leaderboard

We evaluated representative VLMs on AncientVision-3T. Below is the baseline performance:

| Model | Size | OCR (Acc.Cha) | IC (Accuracy) | IU (Rouge-L) |
| :--- | :---: | :---: | :---: | :---: |
| **Xunzi-Qwen2-VL** (Domain-Specific) | 7B | **97.57** | 47.20 | **23.24** |
| **Qwen3-VL** (General) | 8B | 87.94 | **78.00** | 8.06 |
| **LLaVA-OneVision-1.5** | 8B | 6.59 | 25.60 | 4.81 |

*Note: The general model (Qwen3-VL) exhibits stronger separability across tasks, while the domain-specific model (Xunzi-Qwen2-VL) demonstrates higher redundancy but superior understanding in low-resource settings.*

## 🧠 Interpretability: AC-TCGN

Along with the dataset, we propose **AC-TCGN (Ancient Chinese Target-Conditioned Gradient-based Neuron Identification)**. This method allows researchers to:
* **Identify** neurons crucial for predictions across OCR, Classification, and Understanding.
* **Analyze** how domain adaptation affects neuronal topology and task separability.
* **Visualize** distinct neuronal activation patterns across different task settings.

### 🔍 Neuron Activation Visualization
The heatmaps below visualize the neuron activation patterns. The **x-axis** represents the model layer index, and the **y-axis** denotes the identified neurons associated with specific tasks.

| Model \ Task | OCR (Optical Character Recognition) | IC (Image Classification) | IU (Image Understanding) |
| :--- | :---: | :---: | :---: |
| **Qwen3-VL-8B** | ![OCR](figure/Qwen-OCR.png) | ![IC](figure/Qwen-IC.png) | ![IU](figure/Qwen-IU.png) |
| **XunziQwen2-VL-7B** | ![OCR](figure/Xunzi-OCR.png) | ![IC](figure/Xunzi-IC.png) | ![IU](figure/Xunzi-IU.png) |

### 📈 Key Observations
Based on our AC-TCGN analysis, we observed several key differences between the base model and the domain-adapted model:

* **OCR Task:** Neurons are primarily concentrated in **layers 4–7**. **XunziQwen2-VL-7B** exhibits denser activations than Qwen3-VL-8B, indicating a significantly enhanced capacity for ancient text recognition after domain adaptation.
* **Image Classification (IC):** While classification typically requires higher-layer involvement, XunziQwen2-VL-7B shows weaker activation in these layers. This suggests a potential training bias toward textual data, which explains its specialized performance.
* **Image Understanding (IU):** XunziQwen2-VL-7B demonstrates **broader and stronger activations** across multiple layers. This suggests more comprehensive visual-semantic grounding compared to the base model, which relies more on shallower layers.

## 🚀 Usage

The dataset is distributed across **Hugging Face** (Images) and **GitHub** (Annotations/Text).

### 1. Download Images (Hugging Face)
The high-resolution images are hosted on Hugging Face to optimize storage.

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/Myralala/AncientVision-3T)

You can download the images manually or use the `huggingface_hub` library.

### 2. Get Annotations (GitHub)
The textual data (ground truth labels, OCR text, and classification categories) are provided in this repository.

1.  Clone this repository:
    ```bash
    git clone [https://github.com/YourUsername/YourRepoName.git](https://github.com/YourUsername/YourRepoName.git)
    cd YourRepoName
    ```
2.  Align images with annotations:
    Ensure the image filenames in the Hugging Face dataset match the IDs provided in the JSON/Text files located in the `data/` folder of this repository.
