# <img src="logo/mdseval_logo.png" alt="Local Image" width="45"> MDSEval:  Meta-Evaluation Benchmark for Multimodal Dialogue Summarization
---
**Authors**: Yinhong Liu, Jianfeng He, Hang Su, Ruixue Lian, Yi Nian, Jake W. Vincent, Srikanth Vishnubhotla, Robinson Piramuthu, Saab Mansour.

**Updates: Our work has been accepted by EMNLP 2025 🎉**

This is the official repository for the **MDSEval** benchmark. It includes all human annotations, benchmark data, and the implementation of our newly proposed data filtering framework, **Mutually Exclusive Key Information (MEKI)**. MEKI is designed to filter high-quality multimodal data by ensuring that each modality contributes unique information.  

⚠️ **Note:** MDSEval is an **evaluation benchmark**. The data provided here should **not** be used for training NLP models.

## Introduction to MDSEval
---

Multimodal Dialogue Summarization (MDS) is an important task with wide-ranging applications. To develop effective MDS models, robust automatic evaluation methods are essential to reduce both costs and human effort. However, such methods require a strong **meta-evaluation benchmark** grounded in human annotations.  

**MDSEval** is the first meta-evaluation benchmark for MDS. It consists of:
- Image-sharing dialogues  
- Multiple corresponding summaries   
- Human judgments across eight well-defined quality dimensions  

To ensure data quality and diversity, we introduce a novel filtering framework, **Mutually Exclusive Key Information (MEKI)**, which leverages complementary information across modalities.  

Our contributions include:
- The first formalization of key evaluation dimensions specific to MDS  
- A high-quality benchmark dataset for robust evaluation  
- A comprehensive assessment of state-of-the-art evaluation methods, showing their limitations in distinguishing between summaries from advanced MLLMs and their vulnerability to various biases

## Dependencies
---
Besides the `requirements.txt`, we additionaly depends on:
* The [google-research](https://github.com/google-research/google-research) with install command in `prepare_dialog_data.sh`
* The external images provided in `MDSEval_annotations.json` with download script in `prepare_image_data.sh`
* The model checkpoint [ViT-H-14-378-quickgelu](https://huggingface.co/immich-app/ViT-H-14-378-quickgelu__dfn5b) loaded by `meki.py`

## Download the Dialogue and Image Data
---
We first download and merge the textual dialogues from their source (PhotoChat and DialogCC)
```bash
bash prepare_dialog_data.sh
```
Then download the images for **MDSEval**:
```bash
bash prepare_image_data.sh
```
Note that the original hosting website is not very stable, so you may need to run the script multiple times to ensure all images are successfully downloaded.

<!-- We provide two approaches to download the images for **MDSEval**:

1. [Download Link](https://drive.google.com/drive/folders/1BHoTvIccgk56Q9vLCmuz8KtE6UbwB1X2?usp=drive_link) – Download the images directly and save them in the current folder.  
2. Use `image_download.py` – This script downloads the images from the source. Note that the original hosting website is not very stable, so you may need to run the script multiple times to ensure all images are successfully downloaded.   -->

## MDSEval data
---
You can explore the MDSEval dataset using the provided notebook:  
`demonstrations.ipynb`, which contains functions to load and visualize the data.

The MDSEval dataset includes the following statistics:  

| Statistic                   | Value |
|-----------------------------|------:|
| Total number of dialogues   |   198 |
| Summaries per dialogue      |     5 |
| Avg. turns per dialogue     |  17.1 |
| Avg. tokens per dialogue    | 209.0 |
| Evaluation aspects          |     8 |
| Avg. annotators per summary |   2.9 |
| Avg. sentences per summary  |   4.5 |


Human annotations across **eight evaluation dimensions**:

| Evaluation Dimensions                  | Scale                                                                       | Note                                  |
|----------------------------------------|-----------------------------------------------------------------------------|---------------------------------------|
| Multimodal Coherence (COH)             | 1-5                                                                         |                                       |
| Conciseness (CON)                      | 1-5                                                                         |                                       |
| Multimodal Coverage (COV)              | 1-5                                                                         |                                       |
| Multimodal Information Balancing (BAL) | 1-7                                                                         | Bipolar                               |
| Topic Progression (PROG)               | 1-5                                                                         |                                       |
| Multimodal Faithfulness (FAI)          | Faithful, Not faithful to image, Not faithful to text, Not faithful to both | Both sentence level and summary level |
|                                        |                                                                             |                                       |


## MEKI
---
To ensure the dataset is sufficiently challenging for multimodal summarization, dialogues should contain **key information uniquely conveyed by a single modality** — meaning it cannot be inferred from the other. To quantify this, we introduce **Mutually Exclusive Key Information (MEKI)** as a selection metric.


We embed both the image and textual dialogue into a **shared semantic space**, e.g. using the CLIP model, denoted as vectors  $I\in \mathbb{R}^N$ and $T \in \mathbb{R}^N$. $N$ is the embedding dimension. Since CLIP embeddings are unit-normalized, we maintain this normalization for consistency.

To measure **Exclusive Information (EI)** in $I$ that is not present in $T$, we compute the orthogonal component of $I$ relative to $T$:
\[
   % \operatorname{EI}(I|T) = 
   I_T^\perp = I - \operatorname{Proj}_T(I) = I -  \frac{\langle I, T\rangle}{\langle T, T\rangle} T,
\]
where $\langle \cdot , \cdot \rangle$ denote the dot product.

Next, to identify **Exclusive Key Information (EKI)** — crucial content uniquely conveyed by one modality — we first generate a pseudo-summary $S$, which extracts essential dialogue and image details. This serves as a reference proxy rather than a precise summary, helping distinguish key information. We embed and normalize $S$ in the CLIP space and compute:
\[
  \operatorname{EKI}(I|T; S) =  
  % \| \operatorname{Proj}_S(I_T^\perp) \| = 
  \left\| \frac{\langle I_T^\perp, S\rangle}{\langle S, S\rangle} S \right\|
\]
which quantifies the extent of exclusive image-based key information. Similarly, we compute $\operatorname{EKI}(T|I; S)$ for textual exclusivity.

Finally, the MEKI score aggregates both components:
\[
\operatorname{MEKI}(I, T; S) = \lambda \operatorname{EKI}(I \mid T; S)  + (1-\lambda)\operatorname{EKI}(T \mid I; S)
\]
where $\lambda=0.3$, chosen to balance the typically higher magnitude of the exclusivity term in text-based information, ensuring that the average magnitudes of both terms are approximately equal. 


**The MEKI implementation is provided in `meki.py`. Please follow the instructions in the file to use it.**



## License
---

MDSEval is constructed using images and dialogues from the following sources:  

- [DialogCC](https://github.com/passing2961/DialogCC) – released under the MIT License.  
- [PhotoChat](https://github.com/google-research/google-research/tree/29f189e3141d53b17cb7415655165d4eb6fc2d04/multimodalchat) – released under the Apache 2.0 License. 

Accordingly, we release MDSEval under the Apache 2.0 License.  


## Citation
---
If you found the benchmark useful, please consider citing our work.

## Other
---
This is an intern project which has ended. Therefore, there will be no regular updates for this repository.




