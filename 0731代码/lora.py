from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
import torch

# ✅ Step 1: 模型 & tokenizer 加载
model_name = "Qwen/Qwen1.5-1.8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# ✅ Step 2: 应用 LoRA 配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen 支持这些模块
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ✅ Step 3: 构造微型中英文问答数据
examples = [
    {"question": "What is the capital of China?", "answer": "The capital of China is Beijing."},
    {"question": "2 + 2 等于多少？", "answer": "2 + 2 等于 4。"},
    {"question": "谁写了《百年孤独》？", "answer": "《百年孤独》的作者是加西亚·马尔克斯。"},
    {"question": "Who is the author of Hamlet?", "answer": "Hamlet was written by William Shakespeare."},
]
dataset = Dataset.from_list(examples)

# ✅ Step 4: Tokenize
def preprocess(example):
    prompt = f"<|user|>\n{example['question']}\n<|assistant|>\n{example['answer']}"
    encoding = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_attention_mask=True
    )
    encoding["labels"] = encoding["input_ids"].copy()
    return encoding

tokenized_dataset = dataset.map(preprocess)

# ✅ Step 5: 训练参数设置
training_args = TrainingArguments(
    output_dir="./lora-qwen-output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=5,
    logging_steps=1,
    save_strategy="no",
    fp16=True,
    report_to="none"
)

# ✅ Step 6: 使用 Causal LM DataCollator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ✅ Step 7: 启动训练
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()

# ✅ Step 8: 保存 LoRA adapter
model.save_pretrained("./lora-qwen1.5-adapter")
