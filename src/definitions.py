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
    "falcon-40b": {
        "float32": "tiiuae/falcon-40b-instruct",
        "float16": "tiiuae/falcon-40b-instruct",
        "int8": "TheBloke/falcon-40b-instruct-GPTQ",
    },
    "Mixtral": {
        "float32": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "float16": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "int8": "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ",
    },
}

MODELS = list(HUGGIGNFACE_MODEL_PATHS.keys())

MAX_MODEL_LEN = {"Llama-2-7b-chat": 4096, "Llama-2-13b-chat": 4096, "Llama-2-70b-chat": 4096, "falcon-40b": 2048, "Mixtral": 32000}
TOKENIZER_NAME = {"Llama-2-7b-chat": "meta-llama/Llama-2-7b-chat-hf", "Llama-2-13b-chat": "meta-llama/Llama-2-13b-chat-hf", "Llama-2-70b-chat": "meta-llama/Llama-2-70b-chat-hf", "falcon-40b": "tiiuae/falcon-7b-instruct", "Mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1"}

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

_falcon_chat_template = """
{% if messages[0]['role'] == 'system' %}
    {% set loop_messages = messages[1:] %}
    {% set system_message = messages[0]['content'] %}
{% else %}
    {% set loop_messages = messages %}
    {% set system_message = '' %}
{% endif %}
{% for message in loop_messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}
    {% if loop.index0 == 0 %}
        {{ system_message }}
    {% endif %}
    {% if message['role'] == 'user' %}
        {{ '\\n\\nUser: ' + message['content'].strip() }}
    {% elif message['role'] == 'assistant' %}
        {{ '\\n\\nAssistant: ' + message['content'].strip() }}
    {% endif %}
{% endfor %}
{% if add_generation_prompt %}
    {{ '\\n\\nAssistant:' }}{% endif %}"
"""

# Append system prompt at the beginning of the first user message
_mixtral_chat_template = """
{{ bos_token }}
{% if messages[0]['role'] == 'system' %}
    {% set loop_messages = messages[1:] %}
    {% set system_message = messages[0]['content'] %}
{% else %}
    {% set loop_messages = messages %}
    {% set system_message = '' %}
{% endif %}
{% for message in loop_messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}
    {% if loop.index0 == 0 %}
        {{ system_message }}
    {% endif %}
    {% if message['role'] == 'user' %}
        {{ '[INST] ' + message['content'] + ' [/INST]' }}
    {% elif message['role'] == 'assistant' %}
        {{ message['content'] + eos_token}}
    {% endif %}
{% endfor %}
"""

OVERRIDE_CHAT_TEMPLATES = {
    "falcon-40b": "".join(line.strip() for line in _falcon_chat_template.split("\n")),
    "Mixtral": "".join(line.strip() for line in _mixtral_chat_template.split("\n")),
}