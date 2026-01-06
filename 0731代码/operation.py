from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# 加载基座模型
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-1.8B", 
    device_map="auto", 
    torch_dtype=torch.float16, 
    trust_remote_code=True
)

# 加载 LoRA adapter（你刚刚训练得到的）
model = PeftModel.from_pretrained(base_model, "./lora-qwen1.5-adapter")

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B", trust_remote_code=True)

# 构造 prompt，注意格式：<|user|> 开头，<|assistant|> 提示模型生成
prompt = "<|user|>\n介绍一下深圳市\n<|assistant|>\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 模型生成
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
