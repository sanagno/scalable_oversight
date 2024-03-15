Download data from https://github.com/nyu-mll/quality/tree/main/data/v1.0.1.

To launch explanations

```python
DATASET=gpqa
CACHE_DIR=cache
BASE_DATA_FOLDER=./data
MODEL=Llama-2-70b-chat

python scripts/generate_explanations.py --dataset $DATASET --cache_dir $CACHE_DIR --base_data_folder $BASE_DATA_FOLDER --model $MODEL
```
To get predictions

```python
DATASET=gpqa
CACHE_DIR=cache
BASE_DATA_FOLDER=./data
MODEL_JUDGE=Llama-2-70b-chat
MODEL_ADVOCATE=Llama-2-70b-chat

python scripts/generate_predictions.py --dataset $DATASET --cache_dir $CACHE_DIR --base_data_folder $BASE_DATA_FOLDER --model_judge $MODEL_JUDGE --model_advocate $MODEL_ADVOCATE
```

Other models to try:
- TheBloke/Wizard-Vicuna-30B-Uncensored-GPTQ (33b?)
- TheBloke/guanaco-65B-GPTQ
- TheBloke/OpenHermes-2.5-Mistral-7B-GPTQ