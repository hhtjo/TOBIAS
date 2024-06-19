# TOBIAS: Topic-Oriented Bias In Abstractive Summarization

This repository contains the code for the Master's Thesis "TOBIAS: Topic-Oriented Bias In Abstractive Summarization".

It contains the code to train the following models on the NEWTS dataset ([Bahrainian et al. (2022)](https://aclanthology.org/2022.findings-acl.42/)):

- **our proposed TOBIAS model**

- **the CONFORMER\* model**:
  Our replication of the CONFORMER model from the paper "Controllable Topic-Focused Abstractive Summarization" by [Bahrainian et al. (2023)](https://arxiv.org/abs/2311.06724).
  Our implementation precomputes the bias weights for each token in the input and stores it as a separate dataset.
  The model is trained using the same training script as TOBIAS.
  

- **BART-base baselines**

## Folder Structure

```
TOBIAS
│
├── conformer/               # CONFORMER replication code
│
├── tobias/                  # TOBIAS code base
│   ├── model_modifications/ # BART Model modifications for TOBIAS and CONFORMER*
│
├── training_script/         # Training scripts for TOBIAS, CONFORMER* and Basline models
```

## Training Scripts

The training scripts are located in the `training_script` folder. The scripts are named as follows:

### Prepare the NEWTS dataset

1. Download the NEWTS dataset from https://github.com/ali-bahrainian/NEWTS:
   ```bash
   wget https://raw.githubusercontent.com/ali-bahrainian/NEWTS/main/NEWTS_test_600.csv
   wget https://raw.githubusercontent.com/ali-bahrainian/NEWTS/main/NEWTS_train_2400.csv
   ```
2. Preprocess the dataset:
   ```bash
   python prepare_dataset.py
   ```
   This script will split each row into two and copy each article to both rows, one for each topical summary.
   The script also creates a cleaned version of the T-W prompt using DBSCAN and BART embeddings.
   The resulting dataset will be stored at `./dataset/newts-cleaned` as a Huggingface Dataset.


### Prepare the bias weights for CONFORMER*
Note: This step is only necessary if you want to train the CONFORMER* model.
This step can take several hours to complete and requires a lot of memory.

1. Run the jupyter notebook `conformer/create_bias_weights.ipynb` to create the bias weights for the CONFORMER* model.
   The notebook will create a new dataset with the bias weights for each token in the input.
   The dataset will be stored at `./dataset/NEWTS_with_tau` as a Huggingface Dataset.