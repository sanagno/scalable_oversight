## 	How Susceptible are LLMs to Influence in Prompts?

This repository contains the code for the paper "How Susceptible are LLMs to Influence in Prompts?".

### Requirements

First download data for the QuALITY dataset if you intend to run experiments on it.
Data can be found at [QuALITY dataset](https://github.com/nyu-mll/quality/tree/main/data/v1.0.1).

### Experiments

The main experiments are run in the `scripts` folder. The most important scripts are `generate_explanations.py` and `generate_predictions.py` to generate explanations using an advocate and to make predictions using a judge on the generated explanations respectively.

To launch explanations:

```python
DATASET=gpqa
CACHE_DIR=cache
BASE_DATA_FOLDER=./data
MODEL=Llama-2-70b-chat

python scripts/generate_explanations.py --dataset $DATASET --cache_dir $CACHE_DIR --base_data_folder $BASE_DATA_FOLDER --model $MODEL
```

To get predictions:

```python
DATASET=gpqa
CACHE_DIR=cache
BASE_DATA_FOLDER=./data
MODEL_JUDGE=Llama-2-70b-chat
MODEL_ADVOCATE=Llama-2-70b-chat

python scripts/generate_predictions.py --dataset $DATASET --cache_dir $CACHE_DIR --base_data_folder $BASE_DATA_FOLDER --model_judge $MODEL_JUDGE --model_advocate $MODEL_ADVOCATE
```

### Citation
TODO