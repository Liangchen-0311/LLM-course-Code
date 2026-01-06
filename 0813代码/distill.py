#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识蒸馏训练脚本
使用教师模型（GLM-4标注）蒸馏业务专属意图分类器
"""

import os
import json
import torch
import logging
import numpy as np
from typing import List, Dict, Any
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from langchain_openai import ChatOpenAI  # 添加这个导入
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntentDataset:
    """意图分类数据集处理类"""
    
    def __init__(self, data_path: str = None):
        self.data_path = data_path
        self.label2id = {}
        self.id2label = {}
        self.teacher_predictions = {}  # 存储教师模型预测
        
    def load_data(self) -> Dataset:
        if self.data_path and os.path.exists(self.data_path):
            logger.info(f"从本地路径加载数据: {self.data_path}")
            if self.data_path.endswith('.json'):
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                raise ValueError("支持的文件格式: .json")
        else:
            logger.info("使用示例意图分类数据")
            data = self._get_sample_data()
        
        self._build_label_mapping(data)
        self.dataset = Dataset.from_list(data)
        logger.info(f"数据集加载完成，共 {len(self.dataset)} 条数据")
        return self.dataset
    
    def _build_label_mapping(self, data: List[Dict[str, Any]]):
        labels = set()
        for item in data:
            if 'label' in item:
                labels.add(item['label'])
            elif 'intent' in item:
                labels.add(item['intent'])
        
        sorted_labels = sorted(list(labels))
        self.label2id = {label: idx for idx, label in enumerate(sorted_labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
    
    def _get_sample_data(self) -> List[Dict[str, Any]]:
        return [
            {"text": "我想查询我的账户余额", "label": "查询余额"},
            {"text": "帮我转账100元给张三", "label": "转账"},
            {"text": "我要申请信用卡", "label": "申请信用卡"},
            {"text": "怎么修改我的手机号码", "label": "修改信息"},
            {"text": "我的银行卡丢失了怎么办", "label": "挂失"},
            {"text": "我想了解理财产品", "label": "理财咨询"},
            {"text": "帮我预约银行服务", "label": "预约服务"},
            {"text": "我要投诉银行服务", "label": "投诉"},
            {"text": "怎么开通网银", "label": "开通服务"},
            {"text": "我想了解贷款利率", "label": "贷款咨询"}
        ]
    
    def preprocess_data(self, tokenizer, max_length: int = 128) -> Dataset:
        if self.dataset is None:
            self.load_data()
        
        def tokenize_function(examples):
            texts = examples.get("text", examples.get("instruction", examples.get("input", "")))
            labels = examples.get("label", examples.get("intent", examples.get("output", "")))
            
            # 确保labels是单个值而不是列表
            if isinstance(labels, list):
                labels = labels[0] if labels else ""
            
            label_ids = self.label2id.get(labels, 0)
            
            tokenized = tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=max_length,
                return_tensors=None
            )
            
            # 确保label_ids是单个整数
            tokenized["labels"] = label_ids
            return tokenized
        
        return self.dataset.map(
            tokenize_function,
            remove_columns=self.dataset.column_names,
            desc="分词处理中..."
        )

    def get_teacher_predictions(self, teacher_model):
        """获取教师模型对所有样本的预测"""
        logger.info("开始获取教师模型预测...")
        
        if self.dataset is None:
            self.load_data()
        
        all_texts = [item["text"] for item in self.dataset]
        all_labels = list(self.label2id.keys())
        
        teacher_logits = []
        for i, text in enumerate(tqdm(all_texts, desc="教师模型预测中")):
            try:
                # 调用教师模型获取预测
                intent, confidence = teacher_model.annotate_intent(text, all_labels)
                # 转换为one-hot向量
                logits = torch.zeros(len(all_labels))
                label_idx = self.label2id.get(intent, 0)
                logits[label_idx] = confidence
                teacher_logits.append(logits)
                
                # 存储预测结果
                self.teacher_predictions[i] = {
                    "intent": intent,
                    "confidence": confidence,
                    "logits": logits
                }
                
            except Exception as e:
                logger.warning(f"教师模型预测失败: {e}")
                # 使用默认预测
                default_logits = torch.zeros(len(all_labels))
                default_logits[0] = 0.5
                teacher_logits.append(default_logits)
                self.teacher_predictions[i] = {
                    "intent": all_labels[0],
                    "confidence": 0.5,
                    "logits": default_logits
                }
        
        logger.info(f"教师模型预测完成，共 {len(teacher_logits)} 个样本")
        return torch.stack(teacher_logits)

class GLM4Teacher:
    """GLM-4教师模型类"""
    
    def __init__(self):
        self.api_key = "sk-"
        self.api_base = "https://api.siliconflow.cn/v1"
        
        # 使用ChatOpenAI替代openai库
        self.llm = ChatOpenAI(
            temperature=0,
            model="THUDM/glm-4-9b-chat",  
            openai_api_key=self.api_key,
            openai_api_base=self.api_base,
            max_retries=0,
        )
        
        logger.info("GLM-4教师模型初始化成功")
    
    def annotate_intent(self, text: str, possible_intents: List[str]) -> tuple[str, float]:
        try:
            prompt = f"""
请分析以下文本的意图，从给定的意图列表中选择最合适的一个。

文本: {text}

可能的意图列表:
{chr(10).join([f"{i+1}. {intent}" for i, intent in enumerate(possible_intents)])}

请以JSON格式返回结果，包含以下字段:
- intent: 选择的意图名称
- confidence: 置信度 (0.0-1.0)
- reasoning: 选择理由

只返回JSON，不要其他内容。
"""
            
            # 使用ChatOpenAI调用
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            # 提取JSON内容，处理可能的markdown代码块
            if content.startswith('```json'):
                content = content[7:]  # 移除 ```json
            if content.endswith('```'):
                content = content[:-3]  # 移除 ```
            content = content.strip()
            
            try:
                result = json.loads(content)
                intent = result.get("intent", possible_intents[0])
                confidence = float(result.get("confidence", 0.8))
                return intent, confidence
            except json.JSONDecodeError:
                logger.warning(f"无法解析GLM-4响应: {content}")
                return possible_intents[0], 0.5
                
        except Exception as e:
            logger.error(f"GLM-4 API调用失败: {e}")
            return self._simulate_teacher_prediction(text, possible_intents)
    
    def _simulate_teacher_prediction(self, text: str, possible_intents: List[str]) -> tuple[str, float]:
        text_lower = text.lower()
        
        intent_keywords = {
            "查询余额": ["余额", "查询", "账户"],
            "转账": ["转账", "转钱", "汇款"],
            "申请信用卡": ["申请", "信用卡", "办卡"],
            "修改信息": ["修改", "更改", "更新"],
            "挂失": ["丢失", "挂失", "补办"],
            "理财咨询": ["理财", "投资", "收益"],
            "预约服务": ["预约", "预约", "服务"],
            "投诉": ["投诉", "不满", "问题"],
            "开通服务": ["开通", "激活", "启用"],
            "贷款咨询": ["贷款", "借款", "利率"]
        }
        
        for intent, keywords in intent_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return intent, 0.9
        
        return possible_intents[0], 0.5

class DistillationModel(torch.nn.Module):
    """知识蒸馏模型包装器"""
    
    def __init__(self, base_model, teacher_logits, temperature=2.0, alpha=0.7):
        super().__init__()
        self.base_model = base_model
        self.teacher_logits = teacher_logits  # 预计算的教师logits
        self.temperature = temperature
        self.alpha = alpha
        
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # 过滤掉不支持的参数
        model_kwargs = {k: v for k, v in kwargs.items() 
                       if k not in ['num_items_in_batch']}
        
        student_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **model_kwargs
        )
        student_logits = student_outputs.logits
        
        if labels is None:
            return student_outputs
        
        # 获取对应的教师logits
        batch_teacher_logits = self._get_batch_teacher_logits(input_ids, labels)
        
        total_loss, soft_loss, hard_loss = self._compute_distillation_loss(
            student_logits, batch_teacher_logits, labels
        )
        
        return {
            'loss': total_loss,
            'logits': student_logits,
            'soft_loss': soft_loss,
            'hard_loss': hard_loss
        }
    
    def _get_batch_teacher_logits(self, input_ids, labels):
        """获取当前批次的教师logits"""
        batch_size = input_ids.shape[0]
        # 这里需要根据input_ids找到对应的教师预测
        # 简化实现：返回预计算的教师logits
        device = input_ids.device
        return self.teacher_logits[:batch_size].to(device)
    
    """
    软标签损失：让学生模型学习教师模型的"软"知识（概率分布）
    硬标签损失：确保学生模型仍然能正确分类（硬分类）
    """
    def _compute_distillation_loss(self, student_logits, teacher_logits, labels):
        # 软标签损失（知识蒸馏）
        soft_targets = torch.softmax(teacher_logits / self.temperature, dim=-1)
        soft_prob = torch.log_softmax(student_logits / self.temperature, dim=-1)
        soft_targets_loss = torch.nn.functional.kl_div(
            soft_prob, soft_targets, reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # 硬标签损失（交叉熵）
        hard_targets_loss = torch.nn.functional.cross_entropy(student_logits, labels)
        
        # 总损失 = α * 软标签损失 + (1-α) * 硬标签损失
        total_loss = self.alpha * soft_targets_loss + (1 - self.alpha) * hard_targets_loss
        
        return total_loss, soft_targets_loss, hard_targets_loss

class DistillationTrainer:
    """知识蒸馏训练器"""
    
    def __init__(
        self,
        model_name: str = "",
        output_dir: str = "",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        temperature: float = 2.0,
        alpha: float = 0.7
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.temperature = temperature
        self.alpha = alpha
        self.model = None
        self.tokenizer = None
        self.teacher = None
        self.teacher_logits = None  # 存储教师模型预测
        
        os.makedirs(output_dir, exist_ok=True)
    
    def setup_model_and_tokenizer(self, num_labels: int, teacher_logits=None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # 针对BERT模型的LoRA配置
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=["query", "key", "value", "dense"]  # BERT的注意力层名称
        )
        
        self.base_model = get_peft_model(self.base_model, lora_config)
        
        self.teacher_logits = teacher_logits
        
        self.model = DistillationModel(
            self.base_model, 
            self.teacher_logits,  # 传入教师logits
            temperature=self.temperature, 
            alpha=self.alpha
        )
        
        logger.info("模型和分词器设置完成")
    
    def setup_teacher(self):
        self.teacher = GLM4Teacher()
        logger.info("教师模型设置完成")
    
    def train(
        self,
        dataset: Dataset,
        num_epochs: int = 5,
        batch_size: int = 8,
        learning_rate: float = 2e-4,
        warmup_steps: int = 100,
        save_steps: int = 200,
        logging_steps: int = 50,
        eval_steps: int = 200
    ):
        if self.model is None:
            raise ValueError("请先调用setup_model_and_tokenizer设置模型")
        
        if self.teacher is None:
            raise ValueError("请先调用setup_teacher设置教师模型")
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=2,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            # 修复参数名称兼容性问题
            eval_strategy="steps",  # 新版本用 eval_strategy
            save_strategy="steps",  # 新版本用 save_strategy
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            report_to=None,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            fp16=True,
        )
        
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=dataset.select(range(min(100, len(dataset)))),
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        logger.info("开始知识蒸馏训练...")
        trainer.train()
        
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        logger.info(f"模型已保存到: {self.output_dir}")
        
        eval_results = trainer.evaluate()
        logger.info(f"最终评估结果: {eval_results}")
    
    def _compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        
        # 确保predictions是正确的形状
        if isinstance(predictions, tuple):
            predictions = predictions[0]  # 取第一个元素
        
        # 处理不同形状的predictions
        if len(predictions.shape) == 3:
            # 如果形状是 (batch_size, sequence_length, num_labels)
            predictions = predictions[:, 0, :]  # 取第一个token的预测
        elif len(predictions.shape) == 2:
            # 如果形状是 (batch_size, num_labels)，直接使用
            pass
        else:
            # 其他情况，尝试重塑
            predictions = predictions.reshape(-1, predictions.shape[-1])
        
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        acc = accuracy_score(labels, predictions)
        
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

def main():
    config = {
        "model_name": "D:/developer/vscode_workspace/llm/models/TinyBERT_General_4L_312D",  # 本地TinyBERT模型
        "data_path": None,
        "output_dir": "./output/intent_classifier_distill",
        "max_length": 128,
        "num_epochs": 5,
        "batch_size": 32,  # 小模型可以更大batch
        "learning_rate": 3e-4,  # 小模型可以稍高学习率
        "lora_r": 4,  # 小模型LoRA参数可以更小
        "lora_alpha": 8,
        "lora_dropout": 0.1,
        "temperature": 2.0,
        "alpha": 0.7
    }
    
    dataset_loader = IntentDataset(data_path=config["data_path"])
    dataset = dataset_loader.load_data()
    
    trainer = DistillationTrainer(
        model_name=config["model_name"],
        output_dir=config["output_dir"],
        lora_r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        temperature=config["temperature"],
        alpha=config["alpha"]
    )
    
    # 先设置教师模型
    trainer.setup_teacher()
    
    # 获取教师模型预测
    teacher_logits = dataset_loader.get_teacher_predictions(trainer.teacher)
    
    # 设置模型和分词器，传入教师预测
    trainer.setup_model_and_tokenizer(
        num_labels=len(dataset_loader.label2id),
        teacher_logits=teacher_logits
    )
    
    processed_dataset = dataset_loader.preprocess_data(
        trainer.tokenizer,
        max_length=config["max_length"]
    )
    
    trainer.train(
        dataset=processed_dataset,
        num_epochs=config["num_epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"]
    )
    
    logger.info("知识蒸馏训练完成！")
    
    label_mapping = {
        "label2id": dataset_loader.label2id,
        "id2label": dataset_loader.id2label
    }
    
    with open(os.path.join(config["output_dir"], "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=2)
    
    logger.info(f"标签映射已保存到: {os.path.join(config['output_dir'], 'label_mapping.json')}")

if __name__ == "__main__":
    main()
