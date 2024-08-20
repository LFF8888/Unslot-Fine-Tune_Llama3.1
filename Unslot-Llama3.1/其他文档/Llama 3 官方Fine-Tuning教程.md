[Fine-tuning | How-to guides (meta.com)](https://llama.meta.com/docs/how-to-guides/fine-tuning/)

# 微调

如果你想通过编写代码来学习，强烈推荐查看 [了解 Llama 3](https://github.com/meta-llama/llama-recipes/blob/main/recipes/quickstart/Getting_to_know_Llama.ipynb) 笔记本。这是一个很好的起点，涵盖了 Meta Llama 上最常见的操作。

## 微调

全参数微调是一种微调预训练模型所有层的所有参数的方法。通常，它可以实现最佳性能，但也是资源最密集和耗时最长的：它需要最多的 GPU 资源并且花费最长时间。

PEFT，即参数高效微调，允许以最小的资源和成本微调模型。有两种重要的 PEFT 方法：LoRA（低秩适应）和 QLoRA（量化 LoRA），其中预训练模型分别作为量化的 8 位和 4 位权重加载到 GPU。使用 LoRA 或 QLoRA 微调 Llama 2-13B 模型时，你可能只需一块拥有 24GB 内存的消费者级 GPU，使用 QLoRA 甚至需要更少的 GPU 内存和微调时间。

通常，应首先尝试 LoRA，或者如果资源极其有限，先尝试 QLoRA，微调完成后评估性能。仅当性能不理想时才考虑全参数微调。

## 实验跟踪

在评估 LoRA 和 QLoRA 等各种微调方法时，实验跟踪至关重要。它确保了可重复性，维护了结构化的版本历史，便于协作，并有助于识别最佳训练配置。特别是在有许多迭代、超参数和模型版本的情况下，像 [Weights & Biases](https://wandb.ai/) (W&B) 这样的工具变得不可或缺。通过与多个框架的无缝集成，W&B 提供了一个全面的仪表板，用于可视化指标、比较运行和管理模型检查点。通常，只需在训练脚本中添加一个参数即可实现这些优势——我们将在 Hugging Face PEFT LoRA 部分中展示一个示例。

## 配方 PEFT LoRA

llama-recipes 仓库详细介绍了不同的 [微调](https://github.com/facebookresearch/llama-recipes/blob/main/docs/LLM_finetuning.md) (FT) 选项，支持提供的示例脚本。特别是，它强调了 PEFT 作为首选微调方法，因为它减少了硬件需求并防止灾难性遗忘。对于特定情况，全参数微调仍然有效，并且可以使用不同的策略来防止对模型的过多修改。此外，微调可以在 [单 GPU](https://github.com/facebookresearch/llama-recipes/blob/main/docs/single_gpu.md) 或 [多 GPU](https://github.com/facebookresearch/llama-recipes/blob/main/docs/multi_gpu.md) 上使用 FSDP 完成。

要运行配方，请按照以下步骤操作：

1. 创建一个包含 pytorch 和其他依赖项的 conda 环境
2. 按照 [此处](https://github.com/facebookresearch/llama-recipes#install-with-pip) 的描述安装配方
3. 使用 git-lfs 或 llama 下载脚本从 hf 下载所需的模型
4. 配置完毕后，运行以下命令：

```
python -m llama_recipes.finetuning \
    --use_peft --peft_method lora --quantization \
    --model_name ../llama/models_hf/7B \
    --output_dir ../llama/models_ft/7B-peft \
    --batch_size_training 2 --gradient_accumulation_steps 2
```

## torchtune ([链接](https://github.com/pytorch/torchtune))

torchtune 是一个 PyTorch 原生库，可用于微调 Meta Llama 系列模型，包括 Meta Llama 3。它支持[端到端的微调生命周期](https://pytorch.org/torchtune/stable/tutorials/e2e_flow.html)，包括：

- 下载模型检查点和数据集
- 使用全参数微调、LoRA 和 QLoRA [微调 Llama 3](https://pytorch.org/torchtune/stable/tutorials/llama3.html) 的训练配方
- 支持在具有 24GB VRAM 的消费者级 GPU 上运行的单 GPU 微调
- 使用 PyTorch FSDP 将微调扩展到多个 GPU
- 使用 Weights & Biases 在训练期间记录指标和模型检查点
- 使用 EleutherAI 的 LM Evaluation Harness 评估微调后的模型
- 通过 TorchAO 进行微调后模型的量化
- 与推理引擎（包括 ExecuTorch）互操作

要[安装 torchtune](https://pytorch.org/torchtune/stable/install.html)，只需运行以下 pip 安装命令：

```
pip install torchtune
```

按照 Hugging Face [meta-llama](https://huggingface.co/meta-llama/Meta-Llama-3-8B) 仓库中的说明确保你有权访问 Llama 3 模型权重。一旦确认访问权限，可以运行以下命令将权重下载到本地。这也将下载分词器模型和负责任使用指南：

```
tune download meta-llama/Meta-Llama-3-8B \
    --output-dir <checkpoint_dir> \
    --hf-token <ACCESS TOKEN>
```

设置环境变量 HF_TOKEN 或传递 --hf-token 参数以验证你的访问权限。你可以在 https://huggingface.co/settings/tokens 找到你的令牌。

用于 Llama 3 的单设备 LoRA 微调的基本命令是：

```
tune run lora_finetune_single_device --config llama3/8B_lora_single_device
```

torchtune 包含的内置配方包括：

- 在[单设备](https://github.com/pytorch/torchtune/blob/main/recipes/full_finetune_single_device.py)和 [使用 FSDP 的多设备](https://github.com/pytorch/torchtune/blob/main/recipes/full_finetune_distributed.py) 上的全参数微调
- 在[单设备](https://github.com/pytorch/torchtune/blob/main/recipes/lora_finetune_single_device.py)和 [使用 FSDP 的多设备](https://github.com/pytorch/torchtune/blob/main/recipes/lora_finetune_distributed.py) 上的 LoRA 微调
- 在[单设备](https://github.com/pytorch/torchtune/blob/main/recipes/lora_finetune_single_device.py)上的 QLoRA 微调，具有 QLoRA 特定的[配置](https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama3/8B_qlora_single_device.yaml)

你可以通过阅读 torchtune [入门指南](https://github.com/pytorch/torchtune?tab=readme-ov-file#get-started)了解更多关于微调 Meta Llama 模型的信息。

## Hugging Face PEFT LoRA ([链接](https://github.com/huggingface/peft))

使用低秩适应（LoRA），Meta Llama 作为量化的 8 位权重加载到 GPU 内存中。

使用 Hugging Face PEFT LoRA 的微调 ([链接](https://huggingface.co/blog/llama2#fine-tuning-with-peft)) 非常简单 - 以 OpenAssistant 数据集上的 Meta Llama 2 7b 的一个示例微调运行可以通过三个简单步骤完成：

```
pip install trl
git clone https://github.com/huggingface/trl

python trl/examples/scripts/sft.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --dataset_name timdettmers/openassistant-guanaco \
    --load_in_4bit \
    --use_peft \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --log_with wandb
```

在单个 GPU 上运行大约需要 16 小时，并使用不到 10GB 的 GPU 内存；将批量大小更改为 8/16/32 将使用超过 11/16/25 GB 的 GPU 内存。微调完成后，你将在名为“output”的新目录中看到至少 adapter_config.json 和 adapter_model.bin - 运行下面的脚本，将基模型与微调后的新模型合并以进行推理：

```
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

model_name = "meta-llama/Llama-2-7b-chat-hf"
new_model = "output"
device_map = {"": 0}

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos

_token
tokenizer.padding_side = "right"

prompt = "Who wrote the book Innovator's Dilemma?"

pipe = pipeline(task="text-generation", model=base_model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])

pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])
```

## QLoRA 微调

QLoRA（量化的 LoRA）比 LoRA 更节省内存。在 QLoRA 中，预训练模型作为量化的 4 位权重加载到 GPU 中。使用 QLoRA 进行微调也非常容易运行 - 一个使用 OpenAssistant 对 Llama 2-7b 进行微调的示例可以通过四个简单步骤完成：

```
git clone https://github.com/artidoro/qlora
cd qlora
pip install -U -r requirements.txt
./scripts/finetune_llama2_guanaco_7b.sh
```

在单个 GPU 上运行大约需要 6.5 小时，并使用 GPU 的 11GB 内存。微调完成后，./scripts/finetune_llama2_guanaco_7b.sh 中指定的 output_dir 将包含 checkoutpoint-xxx 子文件夹，里面存放了微调的适配器模型文件。要运行推理，请使用以下脚本：

```
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import LoraConfig, PeftModel
import torch

model_id = "meta-llama/Llama-2-7b-hf"
new_model = "output/llama-2-guanaco-7b/checkpoint-1875/adapter_model" # 根据需要更改
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    load_in_4bit=True,
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    device_map='auto'
)
model = PeftModel.from_pretrained(model, new_model)
tokenizer = AutoTokenizer.from_pretrained(model_id)

prompt = "Who wrote the book innovator's dilemma?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])
```

[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) 是另一个开源库，可用于简化 Llama 2 的微调过程。一个使用 Axolotl 微调 Meta Llama 的好例子是四个笔记本，涵盖了整个微调过程（生成数据集、使用 LoRA 微调模型、评估和基准测试），见[这里](https://github.com/OpenPipe/OpenPipe/tree/main/examples/classify-recipes)。

## QLoRA 微调

注意：这仅在 Meta Llama 2 模型上进行了测试。

QLoRA（量化的 LoRA）比 LoRA 更节省内存。在 QLoRA 中，预训练模型作为量化的 4 位权重加载到 GPU 中。使用 QLoRA 进行微调也非常容易运行 - 一个使用 OpenAssistant 对 Llama 2-7b 进行微调的示例可以通过四个简单步骤完成：

```
git clone https://github.com/artidoro/qlora
cd qlora
pip install -U -r requirements.txt
./scripts/finetune_llama2_guanaco_7b.sh
```

在单个 GPU 上运行大约需要 6.5 小时，并使用 GPU 的 11GB 内存。微调完成后，./scripts/finetune_llama2_guanaco_7b.sh 中指定的 output_dir 将包含 checkoutpoint-xxx 子文件夹，里面存放了微调的适配器模型文件。要运行推理，请使用以下脚本：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import LoraConfig, PeftModel
import torch

model_id = "meta-llama/Llama-2-7b-hf"
new_model = "output/llama-2-guanaco-7b/checkpoint-1875/adapter_model" # 根据需要更改
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    load_in_4bit=True,
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    device_map='auto'
)
model = PeftModel.from_pretrained(model, new_model)
tokenizer = AutoTokenizer.from_pretrained(model_id)

prompt = "Who wrote the book innovator's dilemma?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])
```

注意：这仅在 Meta Llama 2 模型上进行了测试。

[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) 是另一个开源库，可用于简化 Llama 2 的微调过程。一个使用 Axolotl 微调 Meta Llama 的好例子是四个笔记本，涵盖了整个微调过程（生成数据集、使用 LoRA 微调模型、评估和基准测试），见[这里](https://github.com/OpenPipe/OpenPipe/tree/main/examples/classify-recipes)。