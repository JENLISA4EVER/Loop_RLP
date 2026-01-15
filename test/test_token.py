from transformers import AutoModelForCausalLM, AutoTokenizer
from modeling_ouro import OuroForCausalLM, OuroConfig
from ouro_cache_fix import UniversalTransformerCache
model_name = "/share/home/sxjiang/myproject/self-learn/checkpoints/ouro_rlp_checkpoints_bf16_debug_v1/epoch_1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
messages = [{"role": "user", "content": "256 + \\frac{128}{8} \\times 3 - 64"}]
input_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_str, return_tensors="pt")
print(inputs)
pass