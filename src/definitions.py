HUGGIGNFACE_MODEL_PATHS = {
    "Llama-2-7b-chat": {
        "float32": "meta-llama/Llama-2-7b-chat-hf",
        "float16": "meta-llama/Llama-2-7b-chat-hf",
        "int8": "TheBloke/Llama-2-7B-Chat-GPTQ",
    },
    "Llama-2-13b-chat": {
        "float32": "meta-llama/Llama-2-13b-chat-hf",
        "float16": "meta-llama/Llama-2-13b-chat-hf",
        "int8": "TheBloke/Llama-2-13B-chat-GPTQ",
    },
    "Llama-2-70b-chat": {
        "float32": "meta-llama/Llama-2-70b-chat-hf",
        "float16": "meta-llama/Llama-2-70b-chat-hf",
        "int8": "TheBloke/Llama-2-70B-Chat-GPTQ",
    },
}

MODELS = list(HUGGIGNFACE_MODEL_PATHS.keys())

APPEND_TOKENS = {"Llama-2-7b-chat": [29871], "Llama-2-13b-chat": [29871], "Llama-2-70b-chat": [29871]}
MAX_MODEL_LEN = {"Llama-2-7b-chat": 4096, "Llama-2-13b-chat": 4096, "Llama-2-70b-chat": 4096}
TOKENIZER_NAME = {"Llama-2-7b-chat": "meta-llama/Llama-2-7b-chat-hf", "Llama-2-13b-chat": "meta-llama/Llama-2-13b-chat-hf", "Llama-2-70b-chat": "meta-llama/Llama-2-70b-chat-hf"}

FIELDS = {
    "gpqa": "a science field",
    "quality": "a science field",
}

HAS_DATASET_EXPLANATIONS = {
    "gpqa": True,
    "quality": False,
}

LEVELS = {
    "assistant": "a helpful assistant",
    "level_1": "a 6 years old child",
    "level_2": "a 12 years old child",
    "level_3": "an undergraduate student in {field}",
    "level_4": "a PhD student in {field}",
    "level_5": "an university professor in {field}",
}

SYSTEM_PROMPTS = {level: f"You are {LEVELS[level]}." for level in LEVELS}

POSSIBLE_ADVOCATES = [("None", False), ("dataset", False)] + [
    y for x in list(LEVELS.keys()) for y in [(x, False), (x, True)]
]
