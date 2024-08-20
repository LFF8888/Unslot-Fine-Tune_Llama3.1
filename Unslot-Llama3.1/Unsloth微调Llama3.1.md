(https://huggingface.co/mlabonne)


[![](https://i.imgur.com/jUDo6ID.jpeg)](https://i.imgur.com/jUDo6ID.jpeg)

最近发布的Llama 3.1提供了令人难以置信的高性能，使得闭源模型和开源模型之间的差距不断缩小。与使用冻结的通用大型语言模型（如GPT-4o和Claude 3.5）相比，你可以针对特定的用例对Llama 3.1进行微调，从而以更低的成本获得更好的性能和定制化效果。

[![](https://i.imgur.com/u0rJPa6.png)](https://i.imgur.com/u0rJPa6.png)

本文将提供一个全面的监督微调概述。我们将其与提示工程进行比较，以理解何时使用它更为合理，详细介绍主要的技术及其优缺点，并介绍一些重要概念，如LoRA超参数、存储格式和聊天模板。最后，我们将在Google Colab中通过使用Unsloth进行最先进的优化，实际微调Llama 3.1 8B模型。

本文使用的所有代码均可在[Google Colab](https://colab.research.google.com/drive/164cg_O7SV7G8kZr_JXqLd6VC7pd86-1Z#scrollTo=PoPKQjga6obN)和[LLM课程](https://github.com/mlabonne/llm-course)中找到。特别感谢Daniel Han解答我的问题。

## [](https://huggingface.co/blog/mlabonne/sft-llama3#%F0%9F%94%A7-supervised-fine-tuning)🔧 监督微调

[![](https://i.imgur.com/0akg8cN.png)](https://i.imgur.com/0akg8cN.png)

监督微调（Supervised Fine-Tuning，SFT）是一种**改进和定制**预训练大型语言模型的方法。它涉及在一个小型指令和答案数据集上重新训练基础模型。其主要目标是将一个基本的文本预测模型转变为能够遵循指令并回答问题的助手。SFT还可以提高模型的整体性能，增加新知识，或使其适应特定任务和领域。经过微调的模型还可以通过可选的偏好对齐阶段（参见[我关于DPO的文章](https://mlabonne.github.io/blog/posts/Fine_tune_Mistral_7b_with_DPO.html)）来移除不需要的回应、修改其风格等。

下图展示了一个指令样本。它包括一个用于引导模型的系统提示，一个用于提供任务的用户提示，以及模型预期生成的输出。你可以在[💾 LLM数据集](https://github.com/mlabonne/llm-datasets)的GitHub仓库中找到一系列高质量的开源指令数据集。

[![](https://i.imgur.com/RqlJEtH.png)](https://i.imgur.com/RqlJEtH.png)

在考虑SFT之前，我建议先尝试**少样本提示**或**检索增强生成**（RAG）等提示工程技术。实际上，这些方法可以在不需要微调的情况下解决许多问题，无论是使用闭源还是开源模型（如Llama 3.1 Instruct）。如果这种方法无法满足你的目标（在质量、成本、延迟等方面），那么在有指令数据的情况下，SFT就成为了一个可行的选择。请注意，SFT还提供了额外的控制和定制化的优势，可以创建个性化的LLM。

然而，SFT也有其局限性。它在利用基础模型中已有的知识时效果最好。学习全新的信息（如一种未知的语言）可能会很困难，并导致更频繁的幻觉现象。对于基础模型未知的新领域，建议首先在原始数据集上连续进行预训练。

在另一端，指令模型（即已经微调的模型）可能已经非常接近你的要求。例如，一个模型可能表现得非常好，但却声明它是由OpenAI或Meta训练的，而不是你。这种情况下，你可能希望通过偏好对齐稍微引导指令模型的行为。通过为一小组指令（100到1000个样本之间）提供选定和拒绝的样本，你可以强迫LLM说出是你训练了它，而不是OpenAI。

## [](https://huggingface.co/blog/mlabonne/sft-llama3#%E2%9A%96%EF%B8%8F-sft-techniques)⚖️ SFT 技术

三种最流行的SFT技术是全量微调、LoRA和QLoRA。

[![](https://i.imgur.com/P6sLsxl.png)](https://i.imgur.com/P6sLsxl.png)

**全量微调**是最直接的SFT技术。它涉及在一个指令数据集上重新训练预训练模型的所有参数。此方法通常提供最佳的结果，但需要大量的计算资源（微调一个8B模型需要几块高端GPU）。由于它修改了整个模型，因此也是最具破坏性的方法，可能导致先前技能和知识的灾难性遗忘。

**低秩适应（LoRA）** 是一种流行的参数高效微调技术。它不是重新训练整个模型，而是在每个目标层引入小的适配器（低秩矩阵），同时冻结权重。这使得LoRA可以训练的参数数量显著低于全量微调（不到1%），从而减少了内存使用和训练时间。此方法是非破坏性的，因为原始参数被冻结，适配器可以随意切换或组合。

**QLoRA（量化感知低秩适应）** 是LoRA的扩展版本，提供了更大的内存节省。与标准LoRA相比，它最多可提供33%的额外内存减少，这在GPU内存受限的情况下尤为有用。这种效率的提高是以更长的训练时间为代价的，QLoRA的训练时间通常比常规LoRA长39%左右。

虽然QLoRA需要更多的训练时间，但其显著的内存节省可能使其成为在GPU内存受限情况下唯一可行的选择。因此，在下一节中，我们将在Google Colab中使用此技术微调Llama 3.1 8B模型。










## [](https://huggingface.co/blog/mlabonne/sft-llama3#%F0%9F%A6%99-fine-tune-llama-31-8b)🦙 微调Llama 3.1 8B

为了高效微调[Llama 3.1 8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)模型，我们将使用Daniel和Michael Han开发的[Unsloth](https://github.com/unslothai/unsloth)库。得益于其自定义内核，Unsloth在训练速度上比其他选项快2倍，内存使用减少60%，这使得它在Colab等资源受限的环境中尤为理想。不幸的是，Unsloth目前仅支持单GPU设置。如果需要多GPU设置，我推荐使用[TRL](https://huggingface.co/docs/trl/en/index)和[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)等流行替代方案（它们也包含Unsloth作为后端）。

在本例中，我们将使用QLoRA对[mlabonne/FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k)数据集进行微调。这是[arcee-ai/The-Tome](https://huggingface.co/datasets/arcee-ai/The-Tome)的一个子集（不包括[arcee-ai/qwen2-72b-magpie-en](https://huggingface.co/datasets/arcee-ai/qwen2-72b-magpie-en)），经过[HuggingFaceFW/fineweb-edu-classifier](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier)重新过滤。请注意，这个分类器并非专为评估指令数据质量设计，但我们可以将其用作粗略代理。最终的FineTome数据集是一个超高质量的数据集，包含对话、推理问题、函数调用等。

首先，安装所有必需的库。

```python
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
```

安装后，我们可以按如下方式导入它们。

```python
import torch
from trl import SFTTrainer
from datasets import load_dataset
from transformers import TrainingArguments, TextStreamer
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel, is_bfloat16_supported
```

现在加载模型。因为我们要使用QLoRA，我选择了预量化的[unsloth/Meta-Llama-3.1-8B-bnb-4bit](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-bnb-4bit)。相比原始的16位精度模型（16GB），这个4位精度版本（5.4GB）更小且下载速度更快。我们使用bitsandbytes库以NF4格式加载。

加载模型时，我们必须指定最大序列长度，以限制其上下文窗口。Llama 3.1支持最多128k的上下文长度，但在本例中我们将其设置为2,048，因为更大的上下文长度会消耗更多的计算和显存资源。最后，`dtype`参数会自动检测你的GPU是否支持[BF16格式](https://mlabonne.github.io/blog/posts/Introduction_to_Weight_Quantization.html#background-on-floating-point-representation)，以在训练期间提高稳定性（此功能仅限于Ampere及更新的GPU）。

```python
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=None,
)
```

现在我们的模型已经加载为4位精度，我们希望使用LoRA适配器对其进行参数高效微调。LoRA有三个重要参数：

- **Rank**（r）：决定LoRA矩阵的大小。Rank通常从8开始，但可以高达256。较高的Rank可以存储更多信息，但也会增加LoRA的计算和内存成本。此处我们将其设置为16。
- **Alpha**（α）：用于更新的缩放因子。Alpha直接影响适配器的贡献，通常设置为Rank值的1倍或2倍。
- **目标模块**：LoRA可以应用于模型的各个组件，包括注意力机制（Q、K、V矩阵）、输出投影、前馈块和线性输出层。虽然最初LoRA只关注注意力机制，但扩展到其他组件也显示出益处。然而，适配更多模块会增加可训练参数和内存需求。

在此，我们将r=16，α=16，并将LoRA应用于每个线性模块以最大化质量。我们不使用dropout和biases，以加快训练速度。

此外，我们将使用[Rank-Stabilized LoRA](https://arxiv.org/abs/2312.03732)（rsLoRA），它将LoRA适配器的缩放因子修改为与1/√r成正比，而不是1/r。这稳定了学习（特别是对于较高的适配器Rank），并允许随着Rank的增加改进微调性能。梯度检查点由Unsloth处理，将输入和输出嵌入下放到磁盘以节省显存。

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
    use_rslora=True,
    use_gradient_checkpointing="unsloth"
)
```

通过此LoRA配置，我们将仅训练8B参数中的4200万参数（0.5196%）。这显示了LoRA相比全量微调的高效性。

现在让我们加载并准备数据集。指令数据集存储在**特定格式**中：它可以是Alpaca、ShareGPT、OpenAI等格式。首先，我们需要解析此格式以检索我们的指令和答案。我们的[mlabonne/FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k)数据集使用ShareGPT格式，包含一个唯一的“conversations”列，其中包含JSONL格式的消息。与Alpaca等更简单的格式不同，ShareGPT非常适合存储多轮对话，这更接近用户与LLM的交互方式。

解析完指令-答案对后，我们希望将它们重新格式化以遵循一个**聊天模板**。聊天模板是一种结构化用户与模型之间对话的方式。它们通常包括特殊标记，用于标识消息的开始和结束、谁在说话等。基础模型没有聊天模板，因此我们可以自由选择：ChatML、Llama3、Mistral等。在开源社区中，ChatML模板（最初来自OpenAI）是一种流行的选择。它仅添加两个特殊标记（`` 和 ``）来指示谁在说话。

如果我们将此模板应用于之前的指令样本，结果如下：

```
&lt;|im_start|&gt;system
You are a helpful assistant, who always provide explanation. Think like you are answering to a five year old.&lt;|im_end|&gt;
&lt;|im_start|&gt;user
Remove the spaces from the following sentence: It prevents users to suspect that there are some hidden products installed on theirs device.
&lt;|im_end|&gt;
&lt;|im_start|&gt;assistant
Itpreventsuserstosuspectthattherearesomehiddenproductsinstalledontheirsdevice.&lt;|im_end|&gt;
```

在下面的代码块中，我们使用`mapping`参数解析我们的ShareGPT数据集，并包含ChatML模板。然后我们加载并处理整个数据集，将聊天模板应用于每个对话。

```python
tokenizer = get_chat_template(
    tokenizer,
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    chat_template="chatml",
)
def apply_template(examples):
    messages = examples["conversations"]
    text = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False) for message in messages]
    return {"text": text}

dataset = load_dataset("mlabonne/FineTome-100k", split="train")
dataset = dataset.map(apply_template, batched=True)
```

我们现在可以指定本次训练的超参数。我将简要介绍最重要的超参数：

- **学习率**：控制模型更新参数的力度。过低的学习率会导致训练缓慢，可能陷入局部最小值。过高的学习率会导致训练不稳定或发散，从

而降低性能。
- **学习率调度器**：在训练过程中调整学习率，通常会在开始时使用较高的学习率以快速取得初步进展，然后在后期逐渐降低。线性和余弦调度器是两种最常见的选项。
- **批大小**：在更新权重之前处理的样本数量。较大的批大小通常会产生更稳定的梯度估计，并可以提高训练速度，但也需要更多的内存。梯度累积允许通过在多个前向/后向传递中累积梯度来有效增加批大小，然后再更新模型。
- **训练轮数**：完整通过训练数据集的次数。更多的轮数允许模型多次看到数据，可能会导致更好的性能。然而，过多的轮数可能会导致过拟合。
- **优化器**：用于调整模型参数以最小化损失函数的算法。实践中，强烈推荐使用8位AdamW：它的表现与32位版本一样好，同时使用更少的GPU内存。AdamW的分页版本仅在分布式设置中有意义。
- **权重衰减**：一种正则化技术，为较大的权重添加惩罚，以损失函数中。它有助于防止过拟合，通过鼓励模型学习更简单、更具泛化性的特征。然而，过多的权重衰减可能会阻碍学习。
- **预热步数**：训练开始时，学习率从较小的值逐渐增加到初始学习率。预热可以帮助稳定早期训练，特别是当使用较大学习率或批大小时，通过让模型在进行大更新之前适应数据分布。
- **打包**：批次具有预定义的序列长度。我们可以将多个小样本组合成一个批次，从而提高效率。

我在整个数据集（10万样本）上使用Google Colab中的A100 GPU（40GB显存）对模型进行了训练。训练总共花费了4小时45分钟。当然，你也可以使用显存较小的GPU和较小的批次大小，但它们的速度并不快。例如，在L4上大约需要19小时40分钟，而在免费的T4上则需要47小时。

在这种情况下，我建议仅加载数据集的一部分以加快训练速度。你可以通过修改前面的代码块来实现，例如`dataset = load_dataset("mlabonne/FineTome-100k", split="train[:10000]")`只加载10,000个样本。或者，你可以使用像Paperspace、RunPod或Lambda Labs等更便宜的云GPU提供商。

```python
trainer=SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=True,
    args=TrainingArguments(
        learning_rate=3e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_steps=10,
        output_dir="output",
        seed=0,
    ),
)
trainer.train()
```

现在模型已经训练完成，让我们使用一个简单的提示来测试它。这不是严格的评估，而只是一个快速检查以检测潜在问题。我们使用`FastLanguageModel.for_inference()`来获得2倍速度的推理。

```python
model = FastLanguageModel.for_inference(model)

messages = [
    {"from": "human", "value": "Is 9.11 larger than 9.9?"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

text_streamer = TextStreamer(tokenizer)
_ = model.generate(input_ids=inputs, streamer=text_streamer, max_new_tokens=128, use_cache=True)
```

模型的响应是"9.9"，这是正确的！

现在让我们保存训练好的模型。你可能还记得LoRA和QLoRA部分，实际上我们训练的不是模型本身，而是一组适配器。Unsloth提供了三种保存方法：`lora`仅保存适配器，而`merged_16bit`/`merged_4bit`将适配器与模型合并为16位/4位精度。

在下面的代码中，我们以16位精度合并并保存以最大化质量。首先将其保存在本地的“model”目录中，然后上传到Hugging Face Hub。你可以在[mlabonne/FineLlama-3.1-8B](https://huggingface.co/mlabonne/FineLlama-3.1-8B)上找到训练好的模型。

```python
model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")
model.push_to_hub_merged("mlabonne/FineLlama-3.1-8B", tokenizer, save_method="merged_16bit")
```

Unsloth还允许你直接将模型转换为GGUF格式。这是一种为llama.cpp创建的量化格式，与大多数推理引擎兼容，如[LM Studio](https://lmstudio.ai/)、[Ollama](https://ollama.com/)和oobabooga的[text-generation-webui](https://github.com/oobabooga/text-generation-webui)。由于你可以指定不同的精度（参见[我关于GGUF和llama.cpp的文章](https://mlabonne.github.io/blog/posts/Quantize_Llama_2_models_using_ggml.html)），我们将循环一个列表，以`q2_k`、`q3_k_m`、`q4_k_m`、`q5_k_m`、`q6_k`、`q8_0`量化并将这些量化模型上传到Hugging Face。你可以在[mlabonne/FineLlama-3.1-8B-GGUF](https://huggingface.co/mlabonne/FineLlama-3.1-8B-GGUF)上找到所有的GGUF。

```python
quant_methods = ["q2_k", "q3_k_m", "q4_k_m", "q5_k_m", "q6_k", "q8_0"]

for quant in quant_methods:
    model.push_to_hub_gguf("mlabonne/FineLlama-3.1-8B-GGUF", tokenizer, quant)
```

恭喜你，我们从头开始微调了一个模型并上传了量化模型，现在你可以在你喜欢的推理引擎中使用它们。你可以尝试在[mlabonne/FineLlama-3.1-8B-GGUF](https://huggingface.co/mlabonne/FineLlama-3.1-8B-GGUF)上使用最终模型。接下来可以做什么？这里有一些使用你模型的想法：

- **评估**它在[Open LLM排行榜](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)上（你可以免费提交）或使用其他评估工具，如[LLM AutoEval](https://github.com/mlabonne/llm-autoeval)。
- 使用像[mlabonne/orpo-dpo-mix-40k](https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k)这样的偏好数据集，通过直接偏好优化进行**对齐**以提升性能。
- 将其量化为其他格式，如EXL2、AWQ、GPTQ或HQQ，以实现更快的推理或更低精度，使用[AutoQuant](https://colab.research.google.com/drive/1b6nqC7UZVt8bx4MksX7s656GXPM-eWw4?usp=sharing)。
- 使用[ZeroChat](https://colab.research.google.com/drive/1LcVUW5wsJTO2NGmozjji5CkC--646LgC)将其部署到Hugging Face Space中，以便那些已经足够训练以遵循聊天模板的模型（约20,000个样本）。

## [](https://huggingface.co/blog/mlabonne/sft-llama3#conclusion)总结

本文提供了监督微调的全面概述，以及如何将其实际应用于Llama 3.1 8B模型。通过利用QLoRA的高效内存使用，我们在有限的GPU资源下成功微调了一个超高质量数据集上的8B LLM。我们还提供了更大规模运行的更高效替代方案，并建议了进一步的步骤，包括评估、偏好对齐、量化和部署。

希望本指南对你有所帮助。如果你对LLM感兴趣，建议查看[LLM课程](https://github.com/mlabonne/llm-course)。如果你喜欢这篇文章，可以在X上关注我[@maximelabonne](https://x.com/maxim

elabonne)和在Hugging Face上关注我[@mlabonne](https://huggingface.co/mlabonne)。祝你微调模型顺利！