2024年7月17日，星期三，作者：[adebisi_oluwatomiwa878](https://lablab.ai/u/@adebisi_oluwatomiwa878)

![Llama 3 微调：掌握 AI 项目的定制化](https://lablab.ai/_next/image?url=https%3A%2F%2Fimagedelivery.net%2FK11gkZF3xaVyYzFESMdWIQ%2Fabfb22d8-6222-4698-a302-0efcc88a1000%2Ffull&w=3840&q=80)

## 🚀 Llama 3 微调：掌握 AI 项目的定制化

欢迎阅读本教程！在这里，我将指导你如何使用真实世界的数据集对Llama 3模型进行微调。通过本教程的学习，你将能够在AI黑客马拉松及其他令人兴奋的项目中应用所学知识。

### 目标 📋

本教程将涵盖以下内容：

- 使用可定制的数据集对Llama 3进行任务微调的过程。
- 利用Unsloth实现的Llama 3进行高效训练。
- 利用Hugging Face的工具进行模型处理和数据集管理。
- 根据你的具体需求调整微调过程，使Llama 3可以适应任何任务。

### 先决条件 🛠️

- 基础的Transformer知识
- 熟悉Python编程
- 访问Google Colab
- 基本的模型微调知识

## 设置环境 🖥️

### Google Colab ⚙️

首先，打开[Google Colab](https://colab.research.google.com/)并创建一个新的notebook。确保启用GPU支持以加快训练速度。你可以通过导航至 `Edit > Notebook settings` 并选择 `T4 GPU` 作为硬件加速器来实现这一点。确保选择T4 GPU以获得最佳性能。

### 安装依赖 📦

在你的Colab notebook中运行以下命令来安装必要的库：

```python
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
```

### 加载预训练模型 📚

我们将使用Unsloth实现的Llama 3，该实现针对更快的训练和推理进行了优化。

> **注意:** 如果你使用的是来自Hugging Face的受限模型，需要在 `FastLanguageModel.from_pretrained` 中添加字段 "token"，并填写你的Hugging Face访问令牌。

```python
from unsloth import FastLanguageModel
import torch

max_seq_length = 2048  # 对于Llama 3，你可以选择任意值，最高可达8000
dtype = None  # 自动检测。Tesla T4, V100 选择Float16, Ampere+选择Bfloat16
load_in_4bit = True  # 使用4bit量化以减少内存使用量。可以选择False。

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    # token="YOUR_HUGGINGFACE_ACCESS_TOKEN"  # 使用受限模型时添加此行
)
```

### 准备数据集 📊

首先，将你的 `dataset.json` 文件上传到Google Colab。以下是用于训练情感分析模型的数据集示例：

```json
[
  {
    "instruction": "分类以下文本的情感。",
    "input": "我喜欢这个产品的新功能！",
    "output": "积极"
  },
  {
    "instruction": "分类以下文本的情感。",
    "input": "天气还可以，没什么特别的。",
    "output": "中性"
  },
  {
    "instruction": "分类以下文本的情感。",
    "input": "我对这个服务感到非常失望。",
    "output": "消极"
  },
  {
    "instruction": "分类以下文本的情感。",
    "input": "这部电影很棒，很刺激！",
    "output": "积极"
  },
  {
    "instruction": "分类以下文本的情感。",
    "input": "我不介意等待，这没什么大不了的。",
    "output": "中性"
  },
  {
    "instruction": "分类以下文本的情感。",
    "input": "食物很糟糕，毫无味道。",
    "output": "消极"
  },
  {
    "instruction": "分类以下文本的情感。",
    "input": "今天在公园里玩得很开心！",
    "output": "积极"
  },
  {
    "instruction": "分类以下文本的情感。",
    "input": "这本书很无聊，节奏很慢。",
    "output": "消极"
  },
  {
    "instruction": "分类以下文本的情感。",
    "input": "这只是普通的一天，没什么特别的。",
    "output": "中性"
  },
  {
    "instruction": "分类以下文本的情感。",
    "input": "客服非常有帮助。",
    "output": "积极"
  }
]
```

接下来，定义与数据集结合使用的提示模板，然后从上传的 `dataset.json` 文件中加载数据集：

```python
from datasets import load_dataset

fine_tuned_prompt = """以下是描述任务的指令，搭配的输入提供了进一步的上下文。请编写一个适当完成请求的响应。

### 指令:
{}

### 输入:
{}

### 响应:
{}"""

EOS_TOKEN = tokenizer.eos_token  # 必须添加EOS_TOKEN
def formatting_prompts_func(prompt_dict):
    instructions = prompt_dict["instruction"]
    inputs       = prompt_dict["input"]
    outputs      = prompt_dict["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # 必须添加EOS_TOKEN，否则生成过程将永远不会停止！
        text = fine_tuned_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

# 从本地JSON文件加载数据集
dataset = load_dataset('json', data_files='dataset.json', split='train')
dataset = dataset.map(formatting_prompts_func, batched = True)
```

### 模型微调 🔧

我们将使用 **LoRA (低秩适配)** 来高效地 **微调** 模型。LoRA通过在Transformer架构的每一层中插入可训练的低秩矩阵来适应大型模型。

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # 选择任意大于0的数！建议值为8, 16, 32, 64, 128
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,  # 支持任意值，但=0为优化设置
    bias="none",  # 支持任意值，但="none"为优化设置
    use_gradient_checkpointing="unsloth",  # True或“unsloth”可减少30%的VRAM使用量
)
```

### 参数说明 📝

- **r:** 低秩近似的秩，设置为16在性能和内存使用之间取得良好平衡。
- **target_modules:** 指定LoRA应用的模块，重点关注模型的关键部分。
- **lora_alpha:** LoRA权重的缩放因子，设置为16以确保训练稳定性。
- **lora_dropout:** 应用于LoRA层的dropout率，设置为0表示无dropout。
- **bias:** 指定如何处理偏置项，设置为“none”表示不训练偏置项。
- **use_gradient_checkpointing:** 通过存储中间激活值来减少内存使用量。

### 训练 🏋️

我们将使用Hugging Face的SFTTrainer进行模型训练。

```python
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import is_bfloat16_supported

training_args = TrainingArguments(
    output_dir="./results",
    # num_train_epochs=1,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    gradient_accumulation_steps = 4,
    warmup_steps = 5,
    max_steps = 60,
    learning_rate = 2e-4,
    fp16 = not is_bfloat16_supported(),
    bf16 = is_bfloat16_supported(),
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length = max_seq_length,
)

trainstats =

 trainer.train()
```

`TrainingArguments` 使用的参数：

- **output_dir:** 保存训练模型和检查点的目录。这对于恢复训练和共享模型至关重要。
- **per_device_train_batch_size:** 在每个设备上使用的训练批量大小。这会影响内存使用量和训练速度。
- **save_steps:** 每隔多少步保存一次模型。这有助于在训练中断的情况下从最后一个检查点恢复训练。
- **save_total_limit:** 保留的最大检查点数量。旧的检查点将被删除，这有助于管理磁盘空间。
- **gradient_accumulation_steps:** 在执行反向传播之前累积梯度的步数。这对无法容纳较大批量大小的大型模型非常有用。
- **warmup_steps:** 执行学习率预热的步数。这有助于稳定训练过程。
- **max_steps:** 总训练步数。达到此步数后训练将停止。
- **learning_rate:** 用于训练的学习率。这控制了模型权重更新的大小。
- **fp16:** 是否在训练过程中使用16位（半精度）浮点数，这可以减少内存使用量并加速支持该功能的GPU上的训练。
- **bf16:** 是否使用bfloat16（脑浮点）精度，这对某些硬件如TPU有益。

`SFTTrainer` 使用的参数：

- **model:** 要训练的模型。
- **args:** 定义训练配置的TrainingArguments。
- **train_dataset:** 用于训练的数据集。
- **tokenizer:** 用于处理数据的分词器。它对于将文本转换为输入张量至关重要。
- **dataset_text_field:** 数据集中包含用于训练的文本的字段名称。
- **max_seq_length:** 输入模型的序列最大长度。超过此长度的序列将被截断。

### 使用微调模型 🧠

现在模型已经训练完毕，我们可以尝试一些样本输入来测试情感分析任务：

- 推理是使用训练模型对新数据进行预测的过程。

```python
FastLanguageModel.for_inference(model) # 启用原生2倍速推理
inputs = tokenizer(
[
    fine_tuned_prompt.format(
        "分类以下文本的情感。", # 指令
        "我不喜欢在雨中踢足球", # 输入
        "", # 输出 - 保持空白以进行生成！
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
outputs = tokenizer.decode(outputs[0])
print(outputs)
```

### 保存和共享模型 💾

保存微调模型有两种方法：

#### 本地保存模型

```python
model.save_pretrained("path/to/save")
tokenizer.save_pretrained("path/to/save")
```

#### 将模型保存到Hugging Face Hub（在线）

```python
model.push_to_hub("your_username/your_model_name", token = "YOUR_HUGGINGFACE_ACCESS_TOKEN")
tokenizer.push_to_hub("your_username/your_model_name", token = "YOUR_HUGGINGFACE_ACCESS_TOKEN")
```

## 结论 🎉

通过这些步骤，你应该已经掌握了如何为各种任务微调Llama 3模型。掌握这些技术后，你将能够根据自己的需求调整模型，使你能够更加高效和精确地处理AI项目。祝你的微调和AI项目顺利！🚀