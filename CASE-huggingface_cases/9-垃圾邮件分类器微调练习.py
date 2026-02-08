"""
垃圾邮件分类器微调练习
使用BERT模型进行垃圾邮件分类，掌握Huggingface模型微调的完整流程

步骤：
1. 准备环境与数据
2. 数据预处理（Tokenization）
3. DataCollator 动态补齐
4. 模型加载
5. 评估指标定义
6. 训练配置与执行
7. 模型推理
"""

# ==================== Step 1: 准备环境与数据 ====================
# 安装依赖（如未安装请取消注释运行）
# !pip install transformers datasets evaluate accelerate scikit-learn

import os
import numpy as np
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import evaluate

# 设置随机种子，保证结果可复现
SEED = 42
np.random.seed(SEED)

# 加载SMS垃圾邮件数据集
# 数据集包含短信内容和标签（ham=正常邮件, spam=垃圾邮件）
print("=" * 50)
print("Step 1: 加载数据集")
print("=" * 50)

# 使用Huggingface的SMS Spam数据集
dataset = load_dataset("sms_spam")
print(f"数据集信息: {dataset}")
print(f"训练集样本数: {len(dataset['train'])}")

# 查看数据样例
print("\n数据样例:")
for i in range(3):
    sample = dataset['train'][i]
    print(f"  [{i}] 标签: {sample['label']} | 内容: {sample['sms'][:50]}...")

# 划分训练集和测试集
dataset = dataset['train'].train_test_split(test_size=0.2, seed=SEED)
print(f"\n划分后 - 训练集: {len(dataset['train'])} | 测试集: {len(dataset['test'])}")

# 定义标签映射
label2id = {"ham": 0, "spam": 1}
id2label = {0: "ham", 1: "spam"}


# ==================== Step 2: 数据预处理（Tokenization） ====================
print("\n" + "=" * 50)
print("Step 2: 数据预处理（Tokenization）")
print("=" * 50)

# 选择预训练模型（使用bert-base-uncased）
MODEL_NAME = "bert-base-uncased"
# 如果网络问题，可以使用国内镜像或本地模型
# MODEL_NAME = "hfl/chinese-bert-wwm-ext"  # 中文BERT

# 加载Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"Tokenizer加载完成: {MODEL_NAME}")

# 定义tokenize函数
def tokenize_function(examples):
    """
    对文本进行tokenization
    - truncation=True: 超过最大长度时截断
    - max_length: 设置最大序列长度
    """
    return tokenizer(
        examples["sms"],
        truncation=True,
        max_length=128,
        # padding在这里不做，由DataCollator动态补齐
    )

# 对整个数据集进行tokenization
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,  # 批量处理，提高效率
    remove_columns=["sms"],  # 移除原始文本列
)

print(f"Tokenization完成!")
print(f"处理后的特征: {tokenized_datasets['train'].features}")

# 查看tokenization结果
print("\nTokenization样例:")
sample = tokenized_datasets['train'][0]
print(f"  input_ids长度: {len(sample['input_ids'])}")
print(f"  input_ids前20个: {sample['input_ids'][:20]}")
print(f"  attention_mask前20个: {sample['attention_mask'][:20]}")


# ==================== Step 3: DataCollator 动态补齐 ====================
print("\n" + "=" * 50)
print("Step 3: DataCollator 动态补齐")
print("=" * 50)

# 创建DataCollator
# DataCollatorWithPadding会在每个batch中动态补齐到该batch的最大长度
# 这比预先padding到固定长度更高效
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

print("DataCollator创建完成!")
print("  - 类型: DataCollatorWithPadding")
print("  - 功能: 动态补齐每个batch到该batch的最大长度")
print("  - 优势: 减少不必要的padding，提高训练效率")


# ==================== Step 4: 模型加载 ====================
print("\n" + "=" * 50)
print("Step 4: 模型加载")
print("=" * 50)

# 加载预训练BERT模型用于序列分类
# num_labels=2 表示二分类任务（ham/spam）
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)

print(f"模型加载完成: {MODEL_NAME}")
print(f"  - 分类类别数: 2")
print(f"  - 标签映射: {id2label}")
print(f"  - 模型参数量: {model.num_parameters():,}")


# ==================== Step 5: 评估指标定义 ====================
print("\n" + "=" * 50)
print("Step 5: 评估指标定义")
print("=" * 50)

# 加载评估指标
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    """
    计算评估指标
    - accuracy: 准确率
    - precision: 精确率
    - recall: 召回率
    - f1: F1分数
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    precision = precision_metric.compute(predictions=predictions, references=labels, average="binary")
    recall = recall_metric.compute(predictions=predictions, references=labels, average="binary")
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="binary")
    
    return {
        "accuracy": accuracy["accuracy"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "f1": f1["f1"],
    }

print("评估指标定义完成!")
print("  - accuracy: 准确率 = 正确预测数 / 总预测数")
print("  - precision: 精确率 = TP / (TP + FP)")
print("  - recall: 召回率 = TP / (TP + FN)")
print("  - f1: F1分数 = 2 * precision * recall / (precision + recall)")


# ==================== Step 6: 训练配置与执行 ====================
print("\n" + "=" * 50)
print("Step 6: 训练配置与执行")
print("=" * 50)

# 定义输出目录
OUTPUT_DIR = "./spam_classifier_output"

# 配置训练参数
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,                    # 输出目录
    eval_strategy="epoch",                    # 每个epoch评估一次
    save_strategy="epoch",                    # 每个epoch保存一次
    learning_rate=2e-5,                       # 学习率
    per_device_train_batch_size=16,           # 训练batch大小
    per_device_eval_batch_size=16,            # 评估batch大小
    num_train_epochs=3,                       # 训练轮数
    weight_decay=0.01,                        # 权重衰减
    load_best_model_at_end=True,              # 训练结束时加载最佳模型
    metric_for_best_model="f1",               # 用于选择最佳模型的指标
    logging_dir=f"{OUTPUT_DIR}/logs",         # 日志目录
    logging_steps=50,                         # 每50步记录一次日志
    seed=SEED,                                # 随机种子
    # 如果有GPU，可以启用以下配置
    # fp16=True,                              # 混合精度训练
)

print("训练参数配置:")
print(f"  - 输出目录: {OUTPUT_DIR}")
print(f"  - 学习率: {training_args.learning_rate}")
print(f"  - 训练轮数: {training_args.num_train_epochs}")
print(f"  - Batch大小: {training_args.per_device_train_batch_size}")
print(f"  - 权重衰减: {training_args.weight_decay}")

# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("\nTrainer创建完成，开始训练...")
print("-" * 50)

# 开始训练
train_result = trainer.train()

# 保存模型
trainer.save_model()
print(f"\n模型已保存到: {OUTPUT_DIR}")

# 输出训练结果
print("\n训练结果:")
print(f"  - 训练损失: {train_result.training_loss:.4f}")
print(f"  - 训练时间: {train_result.metrics['train_runtime']:.2f}秒")

# 在测试集上评估
print("\n" + "-" * 50)
print("在测试集上评估...")
eval_results = trainer.evaluate()
print("\n评估结果:")
for key, value in eval_results.items():
    if isinstance(value, float):
        print(f"  - {key}: {value:.4f}")


# ==================== Step 7: 模型推理 ====================
print("\n" + "=" * 50)
print("Step 7: 模型推理")
print("=" * 50)

from transformers import pipeline

# 方法1: 使用pipeline进行推理（推荐，简单易用）
print("\n方法1: 使用Pipeline推理")
print("-" * 30)

# 创建分类pipeline
classifier = pipeline(
    "text-classification",
    model=trainer.model,
    tokenizer=tokenizer,
)

# 测试样例
test_texts = [
    "Congratulations! You've won a free iPhone! Click here to claim your prize!",
    "Hi, are we still meeting for lunch tomorrow at 12pm?",
    "URGENT: Your account has been compromised. Click here immediately!",
    "Hey, just wanted to check if you got my email about the project.",
    "FREE MONEY! Get $1000 cash now! Limited time offer!",
]

print("推理结果:")
for text in test_texts:
    result = classifier(text)[0]
    label = result['label']
    score = result['score']
    emoji = "🚫" if label == "spam" else "✅"
    print(f"  {emoji} [{label}] (置信度: {score:.4f})")
    print(f"     内容: {text[:50]}...")
    print()


# 方法2: 手动推理（更灵活）
print("\n方法2: 手动推理")
print("-" * 30)

import torch

def predict(text, model, tokenizer):
    """
    手动进行模型推理
    """
    # Tokenize输入
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    
    # 移动到模型所在设备
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 推理
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return {
        "label": id2label[predicted_class],
        "confidence": confidence,
        "probabilities": {
            "ham": probabilities[0][0].item(),
            "spam": probabilities[0][1].item(),
        }
    }

# 测试手动推理
test_text = "Win a FREE vacation to Hawaii! Reply YES now!"
result = predict(test_text, trainer.model, tokenizer)
print(f"测试文本: {test_text}")
print(f"预测结果: {result['label']}")
print(f"置信度: {result['confidence']:.4f}")
print(f"概率分布: ham={result['probabilities']['ham']:.4f}, spam={result['probabilities']['spam']:.4f}")


# ==================== 总结 ====================
print("\n" + "=" * 50)
print("🎉 垃圾邮件分类器微调完成！")
print("=" * 50)
print("""
本练习完成了以下步骤:

1. ✅ 准备环境与数据
   - 加载SMS Spam数据集
   - 划分训练集和测试集

2. ✅ 数据预处理（Tokenization）
   - 使用BERT tokenizer处理文本
   - 转换为模型可接受的格式

3. ✅ DataCollator 动态补齐
   - 使用DataCollatorWithPadding动态补齐

4. ✅ 模型加载
   - 加载预训练BERT模型
   - 配置二分类任务

5. ✅ 评估指标定义
   - 定义accuracy, precision, recall, f1

6. ✅ 训练配置与执行
   - 配置训练参数
   - 使用Trainer进行微调

7. ✅ 模型推理
   - Pipeline方式推理
   - 手动推理方式

模型已保存到: {OUTPUT_DIR}
""".format(OUTPUT_DIR=OUTPUT_DIR))
