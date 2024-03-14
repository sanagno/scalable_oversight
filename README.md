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