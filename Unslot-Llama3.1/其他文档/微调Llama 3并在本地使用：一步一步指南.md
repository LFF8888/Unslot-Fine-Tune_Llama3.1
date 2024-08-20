我们将在一个包含患者和医生对话的数据集上微调Llama 3，创建一个专门用于医疗对话的模型。完成合并、转换和量化后，该模型将通过Jan应用程序可以在本地私密使用。

在本教程中，我们将学习如何在医疗数据集上微调Llama 3，并将模型转换为可以通过Jan应用程序在本地使用的格式。

更具体地，我们将：

- 了解Llama 3模型。
- 在医疗数据集上微调Llama 3模型。
- 将适配器与基础模型合并并将完整模型推送到Hugging Face Hub。
- 将模型文件转换为Llama.cpp GGUF格式。
- 量化GGUF模型并将文件推送到Hugging Face Hub。
- 通过Jan应用程序在本地使用微调后的模型。

如果您在寻找学习AI的精选课程，可以查看[AI基础](https://www.datacamp.com/tracks/ai-fundamentals)的六门课程。

## 了解Llama 3 {#understanding-llama-3-metah}

Meta发布了一系列新的大型语言模型（LLMs），称为Llama 3，这是预训练和指令调优的文本到文本模型的集合。

Llama 3是一个自回归语言模型，使用优化的transformer架构。预训练和指令调优模型都具有8B和70B参数，支持8K的上下文长度。

Llama 3 8B是Hugging Face上最受欢迎的LLM。其指令调优版本在各种性能指标上优于谷歌的Gemma 7B-It和Mistral 7B Instruct。而70B的指令调优版本在大多数性能指标上超过了Gemini Pro 1.5和Claude Sonnet：

![Meta Llama 3 Instruct模型性能对比](./Fine-Tuning%20Llama%203%20and%20Using%20It%20Locally_%20A%20Step-by-Step%20Guide%20_%20DataCamp_files/image_4027d14730.png)

来源：[Meta Llama 3](https://llama.meta.com/llama3/)

Meta在一个包含超过15万亿标记的公开在线数据混合数据集上训练了Llama 3。8B模型的知识截止日期为2023年3月，而70B模型的截止日期为2023年12月。模型使用了Grouped-Query Attention（GQA），它减少了内存带宽并提高了效率。

Llama 3模型已根据自定义商业许可证发布。要访问模型，您需要填写表格并接受条款和条件。如果您在不同平台（如Kaggle和Hugging Face）使用不同的电子邮件地址，可能需要多次填写表格。

您可以通过这篇文章了解更多关于Llama 3的信息：[什么是Llama 3？](https://www.datacamp.com/blog/meta-announces-llama-3-the-next-generation-of-open-source-llms)。

## 1. 微调Llama 3 {#1.-fine-tuning-llama-3-forth}

在本教程中，我们将使用[ruslanmv/ai-medical-chatbot](https://huggingface.co/datasets/ruslanmv/ai-medical-chatbot)数据集来微调Llama 3 8B-Chat模型。该数据集包含25万条患者和医生之间的对话。我们将使用Kaggle Notebook来访问该模型和免费的GPU。

### 设置 {#setting-up-befor}

在启动Kaggle Notebook之前，先用您的Kaggle电子邮件地址填写[Meta下载表格](https://llama.meta.com/llama-downloads/)，然后进入Kaggle上的[Llama 3](https://www.kaggle.com/models/metaresearch/llama-3)模型页面并接受协议。批准过程可能需要一到两天。

接下来，执行以下步骤：

1. 启动新的Kaggle Notebook，点击 *+ Add Input* 按钮，选择 *Models* 选项，然后点击 *Llama 3* 模型旁边的加号 *+* 按钮。之后，选择正确的框架、变体和版本，并添加模型。

![将LLama 3模型添加到Kaggle notebook中](./Fine-Tuning%20Llama%203%20and%20Using%20It%20Locally_%20A%20Step-by-Step%20Guide%20_%20DataCamp_files/image_d44e8e9de5.png)

2. 进入 *Session options* 并选择 *GPU P100* 作为加速器。

![在Kaggle中将加速器更改为GPU P100](./Fine-Tuning%20Llama%203%20and%20Using%20It%20Locally_%20A%20Step-by-Step%20Guide%20_%20DataCamp_files/image_716d68d4d1.png)

3. 生成Hugging Face和Weights & Biases令牌，并创建Kaggle Secrets。可以通过 *Add-ons \> Secrets \> Add* 创建并激活Kaggle Secrets。

![设置secrets（环境变量）](./Fine-Tuning%20Llama%203%20and%20Using%20It%20Locally_%20A%20Step-by-Step%20Guide%20_%20DataCamp_files/image_0be6ada487.png)

4. 通过安装所有必要的Python包来启动Kaggle会话。

```javascript
%%capture
%pip install -U transformers 
%pip install -U datasets 
%pip install -U accelerate 
%pip install -U peft 
%pip install -U trl 
%pip install -U bitsandbytes 
%pip install -U wandb
```

5. 导入必要的Python包以加载数据集、模型和分词器并进行微调。

```javascript
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
import os, torch, wandb
from datasets import load_dataset
from trl import SFTTrainer, setup_chat_format
```

6. 我们将使用Weights & Biases跟踪训练过程，然后将微调后的模型保存到Hugging Face，为此我们需要使用API密钥登录到Hugging Face Hub和Weights & Biases。

```javascript
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()

hf_token = user_secrets.get_secret("HUGGINGFACE_TOKEN")

login(token = hf_token)

wb_token = user_secrets.get_secret("wandb")

wandb.login(key=wb_token)
run = wandb.init(
    project='Fine-tune Llama 3 8B on Medical Dataset', 
    job_type="training", 
    anonymous="allow"
)
```

7. 设置基础模型、数据集和新模型变量。我们将从Kaggle加载基础模型，从Hugging Face Hub加载数据集，然后保存新模型。

```javascript
base_model = "/kaggle/input/llama-3/transformers/8b-chat-hf/1"
dataset_name = "ruslanmv/ai-medical-chatbot"
new_model = "llama-3-8b-chat-doctor"
```

8. 设置数据类型和注意力实现。

```javascript
torch_dtype = torch.float16
attn_implementation = "eager"
```

### 加载模型和分词器 {#loading-the-model-and-tokenizer-inthi}

在这部分，我们将从Kaggle加载模型。然而由于内存限制，我们无法加载完整模型。因此，我们将使用4位精度加载模型。

我们的目标是减少内存使用并加快微调过程。

```javascript
# QLoRA 配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation=attn_implementation
)
```

加载分词器，然后设置用于对话AI任务的模型和分词器。默认情况下，它使用OpenAI的`chatml`模板，将输入文本转换为聊天格式。

```javascript
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(base_model)
model, tokenizer = setup_chat_format(model, tokenizer)
```

### 向层中添加适配器 {#adding-the-adapter-to-the-layer-fine-}

微调完整模型需要很多时间，因此为了提高训练时间，我们将附加具有少量参数的适配器层，使整个过程更快且更节省内存。

```javascript
# LoRA 配置
peft_config = LoraConfig

(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)
model = get_peft_model(model, peft_config)
```

### 加载数据集 {#loading-the-dataset-toloa}

要加载和预处理我们的数据集，我们需要：

1. 加载[ruslanmv/ai-medical-chatbot](https://huggingface.co/datasets/ruslanmv/ai-medical-chatbot?row=0)数据集，将其打乱，并仅选择前1000行。这将显著减少训练时间。

2. 格式化聊天模板以使其具有对话性。将患者问题和医生回复组合到一个“文本”列中。

3. 显示文本列中的一个样本（“文本”列具有带特殊标记的聊天格式）。

```javascript
# 导入数据集
dataset = load_dataset(dataset_name, split="all")
dataset = dataset.shuffle(seed=65).select(range(1000)) # 仅使用1000个样本进行快速演示

def format_chat_template(row):
    row_json = [{"role": "user", "content": row["Patient"]},
               {"role": "assistant", "content": row["Doctor"]}]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

dataset = dataset.map(
    format_chat_template,
    num_proc=4,
)

dataset['text'][3]
```

![医疗数据集格式化输出](./Fine-Tuning%20Llama%203%20and%20Using%20It%20Locally_%20A%20Step-by-Step%20Guide%20_%20DataCamp_files/image_289b3ed34b.png)

4. 将数据集拆分为训练集和验证集。

```javascript
dataset = dataset.train_test_split(test_size=0.1)
```

### 微调和训练模型 {#complaining-and-training-the-model-weare}

我们正在设置模型超参数，以便在Kaggle上运行。您可以通过阅读[Fine-Tuning Llama 2](https://www.datacamp.com/tutorial/fine-tuning-llama-2)教程了解每个超参数。

我们将对模型进行一个周期的微调，并使用Weights and Biases记录指标。

```javascript
training_arguments = TrainingArguments(
    output_dir=new_model,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    num_train_epochs=1,
    evaluation_strategy="steps",
    eval_steps=0.2,
    logging_steps=1,
    warmup_steps=10,
    logging_strategy="steps",
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    group_by_length=True,
    report_to="wandb"
)
```

现在，我们将设置一个监督微调（SFT）训练器，并提供训练和评估数据集、LoRA配置、训练参数、分词器和模型。我们将`max_seq_length`设置为`512`，以避免在训练过程中超过GPU内存。

```javascript
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    max_seq_length=512,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing= False,
)
```

我们将通过运行以下代码开始微调过程。

```javascript
trainer.train()
```

训练和验证损失均已下降。考虑对完整数据集进行三次周期的训练以获得更好的结果。

![模型训练中的训练损失和验证损失](./Fine-Tuning%20Llama%203%20and%20Using%20It%20Locally_%20A%20Step-by-Step%20Guide%20_%20DataCamp_files/image_7e33204db3.png)

### 模型评估 {#model-evaluation-wheny}

当您完成Weights & Biases会话时，它将生成运行历史和摘要。

```javascript
wandb.finish()
model.config.use_cache = True
```

![Weights&Biases生成的模型训练摘要](./Fine-Tuning%20Llama%203%20and%20Using%20It%20Locally_%20A%20Step-by-Step%20Guide%20_%20DataCamp_files/image_6c63bde0d0.png)

模型性能指标也存储在Weights & Biases账户下的特定项目名称下。

![Weights&Biases生成的训练图表](./Fine-Tuning%20Llama%203%20and%20Using%20It%20Locally_%20A%20Step-by-Step%20Guide%20_%20DataCamp_files/image_95ab8c4b6c.png)

让我们评估一下模型在一个患者查询样本上的表现，以检查其是否已正确微调。

要生成响应，我们需要将消息转换为聊天格式，通过分词器处理结果，将其输入到模型中，然后解码生成的标记以显示文本。

```javascript
messages = [
    {
        "role": "user",
        "content": "Hello doctor, I have bad acne. How do I get rid of it?"
    }
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, 
                                       add_generation_prompt=True)

inputs = tokenizer(prompt, return_tensors='pt', padding=True, 
                   truncation=True).to("cuda")

outputs = model.generate(**inputs, max_length=150, 
                         num_return_sequences=1)

text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(text.split("assistant")[1])
```

![微调模型的推理](./Fine-Tuning%20Llama%203%20and%20Using%20It%20Locally_%20A%20Step-by-Step%20Guide%20_%20DataCamp_files/image_6d3b42198f.png)

结果显示，即使只进行一个周期的训练，我们也能获得平均水平的结果。

### 保存模型文件 {#saving-the-model-file-we’ll}

我们现在将保存微调后的适配器并将其推送到Hugging Face Hub。Hub API将自动创建存储库并存储适配器文件。

```javascript
trainer.model.save_pretrained(new_model)
trainer.model.push_to_hub(new_model, use_temp_dir=False)
```

![保存微调后的适配器文件](./Fine-Tuning%20Llama%203%20and%20Using%20It%20Locally_%20A%20Step-by-Step%20Guide%20_%20DataCamp_files/image_fdda17d37b.png)

正如我们所见，保存的适配器文件比基础模型要小得多。

最终，我们将保存包含适配器文件的Notebook，以便在新Notebook中将其与基础模型合并。

要保存Kaggle Notebook，请点击右上角的*保存版本*按钮，选择版本类型为*快速保存*，打开高级设置，选择*创建快速保存时始终保存输出*，然后按*保存*按钮。

![Kaggle中的快速保存选项](./Fine-Tuning%20Llama%203%20and%20Using%20It%20Locally_%20A%20Step-by-Step%20Guide%20_%20DataCamp_files/image_552907fb6b.png)

如果在运行代码时遇到问题，请参考这个Kaggle Notebook：[Fine-tune Llama 3 8B on Medical Dataset](https://www.kaggle.com/code/kingabzpro/fine-tune-llama-3-8b-on-medical-dataset)。

我们已经使用GPU微调了模型。您也可以通过以下教程了解如何使用TPU微调LLMs：[使用TPU进行谷歌Gemma模型的微调和推理](https://www.datacamp.com/tutorial/combine-google-gemma-with-tpus-fine-tune-and-run-inference-with-enhanced-performance-and-speed)。

如果您想学习如何微调其他模型，请查看这个[Mistral 7B教程：使用和微调Mistral 7B的一步一步指南](https://www.datacamp.com/tutorial/mistral-7b-tutorial)。

## 2. 合并Llama 3 {#2.-merging-llama-3-touse}

为了在本地使用微调后的模型，我们首先需要将适配器与基础模型合并，然后保存完整模型。

### 设置 {#setting-up-let’s}

执行以下步骤：

1. 创建一个新的Kaggle Notebook并安装所有必要的Python包。确保您正在使用GPU作为加速器。

```javascript
%%capture
%pip install -U bitsandbytes
%pip install -U transformers
%pip install -U accelerate
%pip install -U peft
%pip install -U trl
```

2. 使用Kaggle Secrets登录到Hugging Face Hub。这将帮助我们轻松上传完整的微调模型。

```javascript
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()

hf_token = user_secrets.get_secret("HUGGINGFACE_TOKEN")
login(token = hf_token

)
```

3. 添加Llama 3 8B Chat模型和我们最近保存的微调Kaggle Notebook。我们可以像添加数据集和模型一样将Notebooks添加到当前会话中。

将Notebook添加到Kaggle会话中将允许我们访问输出文件。在我们的案例中，这是一个模型适配器文件。

![将Kaggle Notebook添加到工作区](./Fine-Tuning%20Llama%203%20and%20Using%20It%20Locally_%20A%20Step-by-Step%20Guide%20_%20DataCamp_files/image_cfacc5a214.png)

4. 设置基础模型和适配器的位置变量。

```javascript
base_model = "/kaggle/input/llama-3/transformers/8b-chat-hf/1"
new_model = "/kaggle/input/fine-tune-llama-3-8b-on-medical-dataset/llama-3-8b-chat-doctor/"
```

### 将基础模型与适配器合并 {#merging-the-base-model-with-the-adapter-we’ll}

我们将首先使用`transformers`库加载分词器和基础模型。然后，我们将使用`trl`库设置聊天格式。最后，我们将使用`PEFT`库加载并合并适配器与基础模型。

`merge_and_unload()`函数将帮助我们将适配器权重与基础模型合并并作为独立模型使用。

```javascript
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import torch
from trl import setup_chat_format

# 重新加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(base_model)

base_model_reload = AutoModelForCausalLM.from_pretrained(
        base_model,
        return_dict=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
)

base_model_reload, tokenizer = setup_chat_format(base_model_reload, tokenizer)

# 将适配器与基础模型合并
model = PeftModel.from_pretrained(base_model_reload, new_model)

model = model.merge_and_unload()
```

### 模型推理 {#model-inference-tover}

为了验证我们的模型是否已正确合并，我们将使用`transformers`库中的`pipeline`进行简单推理。我们将使用聊天模板转换消息，然后将提示提供给管道。管道使用模型、分词器和任务类型初始化。

此外，如果您希望使用多个GPU，可以将`device_map`设置为`"auto"`。

```javascript
messages = [{"role": "user", "content": "Hello doctor, I have bad acne. How do I get rid of it?"}]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)

outputs = pipe(prompt, max_new_tokens=120, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print(outputs[0]["generated_text"])
```

![微调模型推理输出](./Fine-Tuning%20Llama%203%20and%20Using%20It%20Locally_%20A%20Step-by-Step%20Guide%20_%20DataCamp_files/image_e04b9b2f4a.png)

我们合并后的微调模型正常工作。

### 保存并推送合并后的模型 {#saving-and-pushing-the-merged-model-we'll}

我们现在将使用`save_pretrained()`函数保存分词器和模型。

```javascript
model.save_pretrained("llama-3-8b-chat-doctor")
tokenizer.save_pretrained("llama-3-8b-chat-doctor")
```

模型文件以safetensors格式存储，模型总大小约为16 GB。

![保存完整的微调模型](./Fine-Tuning%20Llama%203%20and%20Using%20It%20Locally_%20A%20Step-by-Step%20Guide%20_%20DataCamp_files/image_e43a7baac3.png)

我们可以使用`push_to_hub()`函数将所有文件推送到Hugging Face Hub。

```javascript
model.push_to_hub("llama-3-8b-chat-doctor", use_temp_dir=False)
tokenizer.push_to_hub("llama-3-8b-chat-doctor", use_temp_dir=False)
```

最后，我们可以像之前一样保存Kaggle Notebook。

使用这个Kaggle Notebook：[微调适配器到完整模型](https://www.kaggle.com/code/kingabzpro/fine-tuned-adapter-to-full-model)将帮助您解决在自己运行代码时遇到的任何问题。

## 3. 将模型转换为Llama.cpp GGUF {#3.-converting-the-model-to-llama.cpp-gguf-wecan}

我们无法在本地使用safetensors文件，因为大多数本地AI聊天机器人不支持它们。相反，我们将其转换为*llama.cpp* GGUF文件格式。

### 设置 {#setting-up-start}

启动新的Kaggle Notebook会话并添加[微调适配器到完整模型](https://www.kaggle.com/code/kingabzpro/fine-tuned-adopter-to-full-model)Notebook。

克隆llama.cpp存储库并使用以下命令安装llama.cpp框架。

顺便提一下，以下命令仅适用于Kaggle Notebook。您可能需要更改一些内容以在其他平台或本地运行。

```javascript
%cd /kaggle/working
!git clone --depth=1 https://github.com/ggerganov/llama.cpp.git
%cd /kaggle/working/llama.cpp
!sed -i 's|MK_LDFLAGS   += -lcuda|MK_LDFLAGS   += -L/usr/local/nvidia/lib64 -lcuda|' Makefile
!LLAMA_CUDA=1 conda run -n base make -j > /dev/null
```

### 将Safetensors转换为GGUF模型格式 {#converting-safetensors-to-gguf-model-format-runth}

在Kaggle Notebook单元中运行以下命令，将模型转换为GGUF格式。

`convert-hf-to-gguf.py`需要输入模型目录、输出文件目录和输出类型。

```python
!python convert-hf-to-gguf.py /kaggle/input/fine-tuned-adapter-to-full-model/llama-3-8b-chat-doctor/ \
    --outfile /kaggle/working/llama-3-8b-chat-doctor.gguf \
    --outtype f16
```

几分钟内，模型就会转换并保存在本地。然后，我们可以保存Notebook以保存文件。

![将Hugging Face模型文件转换为GGUF模型格式](./Fine-Tuning%20Llama%203%20and%20Using%20It%20Locally_%20A%20Step-by-Step%20Guide%20_%20DataCamp_files/image_3636132aff.png)

如果在运行上述代码时遇到问题，请参考[HF LLM到GGUF](https://www.kaggle.com/code/kingabzpro/hf-llm-to-gguf)Kaggle Notebook。

## 4. 量化GGUF模型 {#4.-quantizing-the-gguf-model-regul}

普通笔记本电脑没有足够的RAM和GPU内存来加载整个模型，所以我们必须量化GGUF模型，将16 GB模型减少到大约4-5 GB。

### 设置 {#setting-up-start}

启动新的Kaggle Notebook会话并添加[HF LLM到GGUF](https://www.kaggle.com/code/kingabzpro/hf-llm-to-gguf?scriptVersionId=178707932)Notebook。

然后，在Kaggle Notebook单元中运行以下命令来安装llama.cpp。

```javascript
%cd /kaggle/working
!git clone --depth=1 https://github.com/ggerganov/llama.cpp.git
%cd /kaggle/working/llama.cpp
!sed -i 's|MK_LDFLAGS   += -lcuda|MK_LDFLAGS   += -L/usr/local/nvidia/lib64 -lcuda|' Makefile
!LLAMA_CUDA=1 conda run -n base make -j > /dev/null
```

### 量化 {#quantization-thequ}

量化脚本需要一个GGUF模型目录、输出文件目录和量化方法。我们使用`Q4_K_M`方法转换模型。

```python
%cd /kaggle/working/

!./llama.cpp/llama-quantize /kaggle/input/hf-llm-to-gguf/llama-3-8b-chat-doctor.gguf llama-3-8b-chat-doctor-Q4_K_M.gguf Q4_K_M
```

![使用Q4_K_M量化GGUF](./Fine-Tuning%20Llama%203%20and%20Using%20It%20Locally_%20A%20Step-by-Step%20Guide%20_%20DataCamp_files/image_abf37d5b8f.png)

我们的模型大小显著减少，从15317.05 MB

减少到4685.32 MB。

### 将模型文件推送到Hugging Face {#pushing-the-model-file-to-hugging-face-topus}

要将单个文件推送到Hugging Face Hub，我们需要：

1. 使用API密钥登录到Hugging Face Hub。
2. 创建API对象。
3. 提供本地路径、存储库路径、存储库ID和存储库类型，上传文件。

```javascript
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient
from huggingface_hub import HfApi
user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HUGGINGFACE_TOKEN")
login(token = hf_token)

api = HfApi()
api.upload_file(
    path_or_fileobj="/kaggle/working/llama-3-8b-chat-doctor-Q4_K_M.gguf",
    path_in_repo="llama-3-8b-chat-doctor-Q4_K_M.gguf",
    repo_id="kingabzpro/llama-3-8b-chat-doctor",
    repo_type="model",
)
```

我们的模型已成功推送到远程服务器，如下所示。

![将量化后的模型文件推送到Hugging Face](./Fine-Tuning%20Llama%203%20and%20Using%20It%20Locally_%20A%20Step-by-Step%20Guide%20_%20DataCamp_files/image_e40918d0e5.png)

如果您仍然遇到问题，请参考[GGUF到量化](https://www.kaggle.com/code/kingabzpro/gguf-to-quantize)Kaggle Notebook，其中包含所有代码和输出。

如果您在寻找更简单的方式转换和量化模型，请访问[这个Hugging Face Space](https://huggingface.co/spaces/ggml-org/gguf-my-repo)，并提供Hub模型ID。

## 5. 在本地使用微调后的模型 {#5.-using-the-fine-tuned-model-locally-touse}

要在本地使用GGUF模型，您需要下载并将其导入到Jan应用程序中。

### 从Hugging Face下载模型 {#downloading-the-model-from-hugging-face-todow}

要下载模型，我们需要：

1. 进入我们的Hugging Face[存储库](https://huggingface.co/kingabzpro/llama-3-8b-chat-doctor)。
2. 点击*文件*标签。
3. 点击带有GGUF扩展名的量化模型文件。

![选择量化后的微调模型](./Fine-Tuning%20Llama%203%20and%20Using%20It%20Locally_%20A%20Step-by-Step%20Guide%20_%20DataCamp_files/image_74007b0906.png)

4. 点击*下载*按钮。

![下载量化后的微调模型](./Fine-Tuning%20Llama%203%20and%20Using%20It%20Locally_%20A%20Step-by-Step%20Guide%20_%20DataCamp_files/image_98130d6958.png)

下载文件到本地需要几分钟。

### 安装Jan应用程序 {#installing-the-jan-application-downl}

从[Jan AI](https://jan.ai/)下载并安装Jan应用程序。

这是启动Jan Windows应用程序时的样子：

![Jan AI Windows应用程序](./Fine-Tuning%20Llama%203%20and%20Using%20It%20Locally_%20A%20Step-by-Step%20Guide%20_%20DataCamp_files/image_b543258d8d.png)

### 在Jan中加载微调后的模型 {#loading-the-fine-tuned-model-in-jan-toadd}

要将模型添加到Jan应用程序中，我们需要导入量化后的GGUF文件。

我们需要进入Hub菜单并点击*导入模型*，如图所示。提供最近下载的文件的位置，仅此而已。

![将微调后的模型导入到Jan AI](./Fine-Tuning%20Llama%203%20and%20Using%20It%20Locally_%20A%20Step-by-Step%20Guide%20_%20DataCamp_files/image_432e62351f.png)

我们进入*线程*菜单并选择微调后的模型。

![在Jan AI线程中选择微调后的模型](./Fine-Tuning%20Llama%203%20and%20Using%20It%20Locally_%20A%20Step-by-Step%20Guide%20_%20DataCamp_files/image_fa2cc0dc4b.png)

### 在Jan中使用微调后的模型 {#using-the-fine-tuned-model-in-jan-befor}

在使用模型之前，我们需要对其进行自定义以正确显示响应。首先，我们在*模型参数*部分修改*提示*模板。

```javascript
system
{system_message}
user
{prompt}
assistant
```

我们在推理参数中添加*停止*标记并将最大标记数更改为512。

```javascript
<endofstring>, Best, Regards, Thanks,-->
```

我们开始编写查询，医生将相应地做出回应。

我们的微调模型在本地完美运行。

![本地使用微调模型](./Fine-Tuning%20Llama%203%20and%20Using%20It%20Locally_%20A%20Step-by-Step%20Guide%20_%20DataCamp_files/image_003d92e286.png)

该模型适用于GPT4ALL、Llama.cpp、Ollama和许多其他本地AI应用程序。要了解如何使用每个应用程序，请查看[如何在本地运行LLMs的教程](https://www.datacamp.com/tutorial/run-llms-locally-tutorial)。

## 结论 {#conclusion-fine-}

在自定义数据集上微调Llama 3模型并在本地使用它，为构建创新应用程序开辟了许多可能性。潜在的使用案例范围从私人和定制的对话AI解决方案到特定领域的聊天机器人、文本分类、语言翻译、问答个性化推荐系统，甚至医疗和营销自动化应用。

使用Ollama和Langchain框架，只需几行代码即可构建自己的AI应用程序。要实现这一点，请参考[LlamaIndex：大型语言模型（LLMs）应用的数据框架](https://www.datacamp.com/tutorial/llama-index-adding-personal-data-to-llms)教程。