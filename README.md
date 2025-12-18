# 📜 AncientVision-3T: A Hierarchical Benchmark for Ancient Chinese Document Vision-Language Tasks

<div align="center">

[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/Myralala/AncientVision-3T)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

## 📖 Introduction

**AncientVision-3T** is a benchmark designed to systematically evaluate the cognitive capabilities of Vision Language Models (VLMs) in the context of Ancient Chinese Documents.

Despite the rapid evolution of VLMs, applying general visual models to ancient manuscripts presents a **"dual chasm"**:
1.  **Visual Perception:** Challenges include vertical text flow, complex layouts, and degradation noise caused by temporal erosion.
2.  **Linguistic Comprehension:** The abstruse grammar and archaic glyphs necessitate deep linguistic priors.

Unlike existing outcome-oriented benchmarks, AncientVision-3T employs a **hierarchical task design**—spanning from OCR perception to image understanding—to decouple and analyze model capabilities based on cognitive complexity.

<div align="center">
  <img src="./assets/figure1_framework.png" width="80%" alt="AncientVision-3T Framework">
  <br>
  <em>Figure 1: The hierarchical task design of AncientVision-3T.</em>
</div>

## 📂 Dataset Statistics

The dataset is curated from public digital archives such as the *Siku Quanshu* and local gazetteers, covering the Song, Yuan, Ming, and Qing dynasties. It comprises **1,500 images** divided into two subsets:

| Subset | Count | Content | Target Task |
| :--- | :---: | :--- | :--- |
| **Textual Images** | 500 | Traditional Chinese text with vertical/left-to-right layouts | **OCR Perception** |
| **Illustrative Images** | 1,000 | Cultural subjects (e.g., ritual vessels, musical instruments, attire) | **Image Classification (IC)** & **Image Understanding (IU)** |

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
* Identify neurons crucial for predictions across OCR, Classification, and Understanding.
* Analyze how domain adaptation affects neuronal topology and task separability.

<div align="center">
  <img src="./assets/figure3_neuron_vis.png" width="80%" alt="Neuron Visualization">
  <br>
  <em>Visualization of neuron activation patterns across different cognitive tasks.</em>
</div>

## 🚀 Usage

### 1. Download Data
The dataset and images are hosted on Hugging Face. You can load it directly using the `datasets` library:

```python
from datasets import load_dataset

# Load the AncientVision-3T dataset
dataset = load_dataset("Myralala/AncientVision-3T")

# Accessing the data
print(dataset)
