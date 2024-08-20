[How to Finetune and Inference Llama-3 - Inferless](https://docs.inferless.com/how-to-guides/how-to-finetune--and-inference-llama3)

Llama 3是一种自回归语言模型，利用改进的transformer架构。Llama 3模型在超过15万亿个标记上进行了8倍的数据训练。它的上下文长度为8K标记，并将分词器的词汇量从上一版本的32K标记增加到128,256。

在本笔记本和教程中，我们将探讨微调[Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)的过程。

你也可以通过提供的[colab笔记本](https://colab.research.google.com/drive/1Rw-zLEuKnnx-eE15HEqaF_ADq9NK1OGb?usp=sharing)直接访问本教程。

在本教程中，我们将使用QLoRA，这将对量化的LLM进行LoRA适配器的微调。

我们将使用[`HuggingFaceH4/ultrachat_200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)数据集，这是Huggingface的UltraChat数据集的过滤版本。

对于模型量化，我们将使用[bitsandbytes](https://github.com/TimDettmers/bitsandbytes)加载4-bit格式的模型。

最后，当在Inferless上部署模型时，你可以预期以下结果。

| 库   | 推理时间 | 冷启动时间 | 标记/秒 |
| ---- | -------- | --------- | ------- |
| vLLM | 1.63秒   | 13.30秒   | 78.65   |

## [为什么要进行微调？](https://docs.inferless.com/how-to-guides/how-to-finetune--and-inference-llama3#why-finetuning)

微调LLM是一种监督学习过程，我们将使用参数高效微调（PEFT），这是一种高效的指令微调形式。

## [开始吧：](https://docs.inferless.com/how-to-guides/how-to-finetune--and-inference-llama3#lets-get-started)

## [安装所需的库](https://docs.inferless.com/how-to-guides/how-to-finetune--and-inference-llama3#installing-the-required-libraries)

你需要以下库来进行微调。

```
!pip install -q -U bitsandbytes
!pip install -q -U transformers
!pip install -q -U peft
!pip install -q -U accelerate
!pip install -q -U datasets
!pip install -q -U trl
```

## [数据集预处理](https://docs.inferless.com/how-to-guides/how-to-finetune--and-inference-llama3#dataset-preprocessing)

从[`HuggingFaceH4/ultrachat_200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)数据集中，我们将采样10000个文本对话进行快速运行。我们使用ChatML格式化数据，因为我们希望模型遵循特定的聊天模板([ChatML](https://huggingface.co/docs/transformers/chat_templating))。

```python
dataset_name = "HuggingFaceH4/ultrachat_200k"
dataset = load_dataset(dataset_name, split="train_sft")
dataset = dataset.shuffle(seed=42).select(range(10000))

def format_chat_template(row):
    chat = tokenizer.apply_chat_template(row["messages"], tokenize=False)
    return {"text":chat}

processed_dataset = dataset.map(
    format_chat_template,
    num_proc= os.cpu_count(),
)

dataset = processed_dataset.train_test_split(test_size=0.01)
```

## [微调Llama-3](https://docs.inferless.com/how-to-guides/how-to-finetune--and-inference-llama3#finetuning-the-llama-3)

现在加载分词器和模型，然后使用[bitsandbytes](https://github.com/TimDettmers/bitsandbytes)将模型量化并准备进行4bit微调。使用Hugging Face Transformers的`AutoTokenizer`加载并初始化分词器。

为了支持`ChatML`，我们将在`trl`中使用`setup_chat_format()`函数。它将设置分词器的`chat_template`，添加特殊标记到`tokenizer`并调整模型的嵌入层以适应新标记。

使用`prepare_model_for_kbit_training()`为QLoRA训练准备模型。

```python
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(
          model_name, quantization_config=bnb_config, device_map={"": 0}, token=hf_token)

model, tokenizer = setup_chat_format(model, tokenizer)
model = prepare_model_for_kbit_training(model)
```

定义LoRA配置和微调模型所需的训练参数。我们将在TRL的`SFTTrainer`中使用这些参数。然后创建SFTTrainer并开始微调过程。

```python
# 定义LoRA配置
peft_config = LoraConfig(
        lora_alpha=64,
        lora_dropout=0.05,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules= ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",]
)

# 定义训练参数
training_arguments = TrainingArguments(
        output_dir="./results_llama3_sft/",
        evaluation_strategy="steps",
        do_eval=True,
        optim="paged_adamw_8bit",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=8,
        log_level="debug",
        save_steps=50,
        logging_steps=50,
        learning_rate=8e-6,
        eval_steps=10,
        num_train_epochs=1,
        warmup_steps=30,
        lr_scheduler_type="linear",
)

# 创建SFT Trainer
trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=2024,
        tokenizer=tokenizer,
        args=training_arguments,
)

# 开始训练过程
trainer.train()
```

在完成训练后，将适配器与原始模型结合并上传到huggingface hub。

```
# 保存适配器
trainer.model.save_pretrained("final_checkpoint")
tokenizer.save_pretrained("final_checkpoint")

# 加载基础模型
model = AutoPeftModelForCausalLM.from_pretrained("final_checkpoint", token=hf_token)
tokenizer = AutoTokenizer.from_pretrained("final_checkpoint", token=hf_token)

# 合并模型与适配器
model = model.merge_and_unload()

# 上传模型到huggingface hub
model.push_to_hub("inferless-llama-3-8B", token=hf_token)
tokenizer.push_to_hub("inferless-llama-3-8B", token=hf_token)
```

## [让我们在Inferless上部署微调后的模型](https://docs.inferless.com/how-to-guides/how-to-finetune--and-inference-llama3#lets-deploy-the-finetuned-model-on-inferless)

## [定义依赖项](https://docs.inferless.com/how-to-guides/how-to-finetune--and-inference-llama3#defining-dependencies)

我们使用的是[vLLM库](https://github.com/vllm-project/vllm)，它提升了LLM的推理速度。

## [构建GitHub/GitLab模板](https://docs.inferless.com/how-to-guides/how-to-finetune--and-inference-llama3#constructing-the-github-gitlab-template)

现在快速构建GitHub/GitLab模板，此过程是强制性的，并确保不要添加任何名为`model.py`的文件。

```
Llama-3/
├── app.py
├── inferless-runtime-config.yaml
├── inferless.yaml
└── input_schema.py
```

你还可以将其他文件添加到此目录。

## [创建推理类](https://docs.inferless.com/how-to-guides/how-to-finetune--and-inference-llama3#create-the-class-for-inference)

在[app.py](https://github.com/inferless/Llama-3/blob/main/app.py)中，我们将定义类并导入所有需要的函数。

1. `def initialize`: 在此函数中，你将初始化模型并定义在推理过程中要使用的任何变量。
2. `def infer`: 此函数在每个请求发送时调用。你可以在这里定义推理所需的所有步骤。你还可以传递自定义值进行推理，并通过`inputs(dict)`参数传递。
3. `def finalize`: 此函数清理所有分配

的内存。

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

class InferlessPythonModel:
    def initialize(self):
        model_id = "rbgo/inferless-llama-3-8B"  # 指定微调模型的仓库ID
        # 定义模型生成的采样参数
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=128)
        # 初始化LLM对象
        self.llm = LLM(model=model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
    def infer(self, inputs):
        prompts = inputs["prompt"]  # 从输入中提取提示
        chat_format = [{"role": "user", "content": prompts}]
        text = self.tokenizer.apply_chat_template(chat_format, tokenize=False, add_generation_prompt=True)
        result = self.llm.generate(text, self.sampling_params)
        # 从结果中提取生成的文本
        result_output = [output.outputs[0].text for output in result]

        # 返回包含结果的字典
        return {'generated_text': result_output[0]}

    def finalize(self):
        pass
```

## [创建输入模式](https://docs.inferless.com/how-to-guides/how-to-finetune--and-inference-llama3#create-the-input-schema)

我们必须在你的GitHub/Gitlab仓库中创建[`input_schema.py`](https://github.com/inferless/Llama-3/blob/main/input_schema.py)，这将帮助我们创建输入参数。你可以查看我们关于[输入/输出模式](https://docs.inferless.com/model-import/input-output-schema)的文档。

对于本教程，我们定义了在API调用期间所需的参数`prompt`。现在让我们创建`input_schema.py`。

```json
INPUT_SCHEMA = {
    "prompt": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["What is AI?"]
    }
}
```

## [创建自定义运行时](https://docs.inferless.com/how-to-guides/how-to-finetune--and-inference-llama3#creating-the-custom-runtime)

这是一个强制步骤，我们允许用户通过[inferless-runtime-config.yaml](https://github.com/inferless/Llama-3/blob/main/inferless-runtime-config.yaml)上传他们的自定义运行时。

```python
build:
  cuda_version: "12.1.1"
  system_packages:
    - "libssl-dev"
  python_packages:
    - "torch==2.2.1"
    - "vllm==0.4.1"
    - "transformers==4.40.1"
```

## [在Inferless上部署模型](https://docs.inferless.com/how-to-guides/how-to-finetune--and-inference-llama3#deploying-the-model-on-inferless)

Inferless支持多种[导入模型](https://docs.inferless.com/model-import/file-structure-req/file-structure-requirements)的方式。对于本教程，我们将使用GitHub。

### [通过GitHub导入模型](https://docs.inferless.com/how-to-guides/how-to-finetune--and-inference-llama3#import-the-model-through-github)

点击`Repo(Custom code)`，然后点击`Add provider`连接到你的GitHub账号。完成账号集成后，点击你的GitHub账号并继续。

![img](https://mintlify.s3-us-west-1.amazonaws.com/inferless-68/images/stable-cascade-model-import.png)

### [提供模型详细信息](https://docs.inferless.com/how-to-guides/how-to-finetune--and-inference-llama3#provide-the-model-details)

输入模型名称并传递GitHub仓库URL。

![img](https://mintlify.s3-us-west-1.amazonaws.com/inferless-68/images/llama-3-create-model.png)

## [配置机器](https://docs.inferless.com/how-to-guides/how-to-finetune--and-inference-llama3#configure-the-machine)

在这个第四步，用户需要配置推理设置。在Inferless平台上，我们支持所有GPU。对于本教程，我们建议使用A100 GPU。在GPU类型的下拉菜单中选择A100。

你还可以灵活选择不同的机器类型。选择`dedicated`机器类型将为你分配整个GPU，而选择`shared`选项将分配50%的显存。在本教程中，我们将选择dedicated机器类型。

选择你需要的模型的最小和最大副本：

- **Min replica:** 始终保持运行的推理工作者数量。
- **Max replica:** 任何时候允许的最大推理工作者数量。

如果你希望为模型设置启用自动重建功能，请切换开关。请注意，设置web-hook是此功能所必需的。点击[这里](https://docs.inferless.com/model-import/automatic-build/automatic-build-via-webhooks)了解更多详细信息。

在`Advance configuration`中，我们可以选择自定义运行时。首先，点击`Add runtime`上传[inferless-runtime-config.yaml](https://github.com/inferless/Llama-3/blob/main/inferless-runtime-config.yaml)文件，给它任意名称并保存。从下拉菜单中选择运行时，然后点击继续。

![img](https://mintlify.s3-us-west-1.amazonaws.com/inferless-68/images/musicgen-model-config.png)

### [审查和部署](https://docs.inferless.com/how-to-guides/how-to-finetune--and-inference-llama3#review-and-deploy)

在最后阶段，仔细审查所有修改。一旦你检查完所有更改，继续点击`Submit`按钮部署模型。

![img](https://mintlify.s3-us-west-1.amazonaws.com/inferless-68/images/llama-3-review.png)

瞧，你的模型现在已部署！

![img](https://mintlify.s3-us-west-1.amazonaws.com/inferless-68/gif/llama-3-demo.gif)

## [方法B：在Inferless CLI上部署模型](https://docs.inferless.com/how-to-guides/how-to-finetune--and-inference-llama3#method-b-deploying-the-model-on-inferless-cli)

Inferless允许你使用Inferless-CLI部署模型。按照以下步骤使用Inferless CLI进行部署。

### [初始化模型](https://docs.inferless.com/how-to-guides/how-to-finetune--and-inference-llama3#initialization-of-the-model)

创建[app.py](https://github.com/inferless/Llama-3/blob/main/app.py)和[inferless-runtime-config.yaml](https://github.com/inferless/Llama-3/blob/main/inferless-runtime-config.yaml)，将文件移动到工作目录。运行以下命令初始化你的模型：

```
inferless init
```

### [上传自定义运行时](https://docs.inferless.com/how-to-guides/how-to-finetune--and-inference-llama3#upload-the-custom-runtime)

创建完[inferless-runtime-config.yaml](https://github.com/inferless/Llama-3/blob/main/inferless-runtime-config.yaml)文件后，你可以运行以下命令：

```
inferless runtime upload
```

输入此命令后，你将被提示提供配置文件名称。输入名称并确保在[inferless.yaml](https://github.com/inferless/Llama-3/blob/main/inferless.yaml)文件中更新它。现在你已准备好进行部署。

![img](https://mintlify.s3-us-west-1.amazonaws.com/inferless-68/images/llama70b-6.png)

### [部署模型](https://docs.inferless.com/how-to-guides/how-to-finetune--and-inference-llama3#deploy-the-model)

执行以下命令部署你的模型。部署完成后，你可以在Inferless平台上跟踪构建日志：

```
inferless deploy
```