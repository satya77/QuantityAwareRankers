# Quantity-aware Retrieval

This repository contains code for the paper "[Numbers Matter! Bringing Quantity-awareness to Retrieval Systems](https://arxiv.org/pdf/2407.10283)".
The paper introduces two types of quantity-aware models, joint quantity and term ranking and disjoint ranking.


The modules include data generation, data processing, and evaluation. For fine-tuning the neural models
of [SPLADE](https://github.com/naver/splade) and [ColBERT](https://github.com/stanford-futuredata/ColBERT), refer to their respective repositories.
For loading the models and creating the quantity-aware variants, some code snippets from the mentioned repositories have been used here.
To run the code, create an environment using the `requirment.txt` file.
<hr>

## Data and Model Checkpoints
Alongside the code, we also publish the training data, benchmark dataset for testing, and trained model checkpoints.
* **Benchmark data:** The benchmark data contains test data, namely FinQuant and MedQuant datasets, and can be downloaded [here](https://drive.google.com/file/d/1JD2a2BRU8-gf5arLufpZw-51nIM6_kJm/view?usp=sharing) alongside the annotation guidelines.
* **Training data:** Raw training data containing sentences with quantities from news articles and the Trec Clinical Trails can be downloaded [here](https://drive.google.com/file/d/1adINyo8FpSdWqzC3gug2C4ZxvrhVb_ws/view?usp=sharing).
* **Checkpoints:** The trained checkpoint for the fine-tuned models on quantity-centric data for SPLADE and ColBERT on finance and medical data can be downloaded [here](https://drive.google.com/file/d/1_e2dFKIiXMbrjJIdhmmCPZ_Kgh0da1h0/view?usp=sharing).

<hr>
Below we describe the content of each module, for more information and examples refer to the readme files inside each respective module.
### Data Generation
For the joint quantity and term ranking we need to generate fine-tuning data using templates and
numerical indices. The module `data_generation` contains code for concept expansion, unit, and value permutation.

### Dataset
Data loader and dataset classes for loading collections and queries for inference are in the `dataset` module.

### Models
The model architecture and interfaces are in `models`.
The models are divided into semantic and lexical models.

The lexical models include BM25 baselines (BM25 without change, BM25 with filtering) and QBM25(quantity-aware variant).

The semantic models include neural baselines (SPLADE and ColBERT) and quantity-aware variants (QSPLADE and QColBERT).

### Evaluate
The module `evaluate` contains scripts for the evaluation of the proposed models on benchmarks.
Here, we include scripts to run evaluation using the pytrec_eval library as well as significant testing.
