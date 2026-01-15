import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoModelForCausalLM, AutoTokenizer
from ouro.modeling_ouro import OuroForCausalLM, OuroConfig
from ouro_cache_fix import UniversalTransformerCache
model_name = "/share/home/sxjiang/myproject/self-learn/checkpoints/ouro_rlp_checkpoints_bf16_bz128_debug_v1/epoch_1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = OuroForCausalLM.from_pretrained(
    model_name, device_map="auto", dtype="auto"
)
# Create custom cache
cache = UniversalTransformerCache()
# Generate text
problem = "Predict the next token of the context and wrap it in \\boxed{}.\n\nContext: '1 + 1 ='\\nAnswer: \\boxed{ 2}\n\nContext: 'The capital of France is'\\nAnswer: \\boxed{ Paris}\n\nContext: 'Problem:What is the smallest positive integer $t$ such that there exist integers $x_1,x_2,\\ldots,x_t$ with  \\[x^3_1+x^3_2+\\,\\ldots\\,+x^3_t=2002^{2002}\\,?\\]\nSolution:\n\nTo determine the smallest positive integer \\( t \\) such that there exist integers \\( x_1, x_2, \\ldots, x_t \\) satisfying\n\n\\[\nx_1^3 + x_2^3 + \\cdots + x_t^3 = 2002^{2002},\n\\]\n\nwe will apply Fermat's Last Theorem and results regarding sums of cubes.\n\n### Step 1: Understanding the Sum of Cubes\nThe problem requires expressing a large number, \\( 2002^{2002} \\), as a sum of cubes. This can be directly related to a result in number theory: every integer can be expressed as the sum of four cubes. We need to determine if three cubes suffice or if four are necessary.\n\n### Step 2: Evaluating Cubes and Powers\nCalculate the properties of \\( 2002^{2002} \\), and recognize:\n\n- \\( 2002 \\equiv 2 \\pmod{9} \\Rightarrow 2002^2 \\equiv '\\nAnswer: \\boxed{"
#加<context>
problem = "Predict the next token of the context and wrap it in \\boxed{}.\n\nContext: My favorite color is blue and my favorite animal is \\nNext token: \\boxed{ a}\n\nContext: The capital of France is\\nNext token: \\boxed{ Paris}\n\nContext: I am a student, my name is John Doe. My favorite color is blue and my father is a engineer. And my \\nNext token:"
messages = [{"role": "user", "content": problem}]
# messages = [{"role": "user", "content": "1 + 1 ="}]
inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",  # Use custom cache
).to(model.device)
print("========output2========")
outputs = model.generate(inputs, max_new_tokens=15, use_cache=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=False))

# 验证代码：尝试构造一个 Batch
prompts = [
    "1 + 1 =",
    "The capital of France is Paris, and the capital of Germany is"
]
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

# 这里很有可能会触发报错
outputs = model.generate(**inputs, max_new_tokens=5)
