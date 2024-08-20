[GitHub - okuvshynov/slowllama: Finetune llama2-70b and codellama on MacBook Air without quantization](https://github.com/okuvshynov/slowllama)

## slowllama

在 Apple M1/M2 设备（例如 Macbook Air 或 Mac Mini）或消费级 nVidia GPU 上微调 Llama2 和 CodeLLama 模型，包括 70B/35B。

slowllama 不使用任何量化。相反，它在前向/后向传递过程中将模型的一部分卸载到 SSD 或主内存中。与从头开始训练大模型（无法实现）或推理（我们可能关心交互性）相比，如果让它运行一段时间，我们仍然可以微调一些东西。

当前版本使用 LoRA 来限制更新到一小部分参数。第一个版本也支持全微调，但我决定暂时删除它，更多信息见下文。

微调是唯一的重点，没有为推理做任何特殊处理，请考虑 [llama.cpp](https://github.com/ggerganov/llama.cpp)。

对于 CUDA 特定的实验，请参见 [a10 报告](https://github.com/okuvshynov/slowllama/blob/main/docs/a10.md)。

这一切都是非常实验性的，但对于 CUDA 来说更是如此。

### 示例

测试在具有 16Gb 内存的 Apple M1 和具有 24Gb 内存的 Apple M2 上进行。

为了微调 llama2 模型，我们需要：

1.  安装依赖项：`pip install torch sentencepiece numpy`。可选：安装 `pip install fewlines` 用于 [权重/梯度分布日志记录](https://github.com/okuvshynov/slowllama/blob/main/docs/lora_weights.md)。
2.  克隆 [llama2](https://github.com/facebookresearch/llama) 并按照说明下载模型。脚本还将下载分词器。`tokenizer.model` 应放置在与 llama 模型相同的目录中。使用 [codellama](https://github.com/facebookresearch/codellama) 获取 CodeLLama 模型。示例文件夹结构可能如下所示：

```
/parent/
    /slowllama/...   # &lt;- 此仓库
    /llama-2-7b/...  # &lt;- 将 tokenizer.model 放在这里
    /llama-2-13b/... # &lt;- 也放在这里
    /llama-2-70b/... # &lt;- 也放在这里
    /CodeLlama-34b-Python/... # 也放在这里
```

让我们从一个 [小例子](https://github.com/okuvshynov/slowllama/blob/main/test_data/cubestat.txt) 开始。这是另一个开源项目 [cubestat](https://github.com/okuvshynov/cubestat) 的介绍。文本足够短，可以作为提示的一部分，但作为插图是可以的，您可以自己在几秒钟内阅读。由于我最近才发布了该项目，原始 llama 不可能知道任何关于它的事情。

请求基本的 llama2-7b 完成提示 _"Cubestat reports the following metrics: "_ 会得到 _"1) the number of cubes in the system, 2) the number of cubes that are in the process of being created"_ 的结果。

第一步是将模型转换为更适合逐块加载到存储中的顺序格式。

输入和输出模型的路径在 conf 文件中配置。有一个基本文件 [conf.py](https://github.com/okuvshynov/slowllama/blob/main/conf.py) 和两个带有一些覆盖的文件 [conf\_fp16.py](https://github.com/okuvshynov/slowllama/blob/main/conf_fp16.py) 和 [conf\_fp32.py](https://github.com/okuvshynov/slowllama/blob/main/conf_fp32.py)。默认情况下， [prepare\_model.py](https://github.com/okuvshynov/slowllama/blob/main/prepare_model.py) 使用 fp16 配置。根据您的模型路径修改这些文件。下面的脚本也使用相同的配置文件。

现在我们可以尝试未微调的 llama2：

现在让我们微调 7b 模型。[finetune.py](https://github.com/okuvshynov/slowllama/blob/main/finetune.py) 是一个非常简单的脚本，它基于纯文本数据训练 LoRA 权重。您可以在此处更改一些设置，例如序列长度、批量大小、学习率、丢失率、迭代次数。当前设置基本上是一个猜测，如果需要可以更改。目前它使用 AdamW 优化器。

这是训练数据集的损失：

```
2023-09-10 22:05:35,569 backprop done, loss after forward pass = 2.9539270401000977
2023-09-10 22:06:08,022 backprop done, loss after forward pass = 2.9073102474212646
2023-09-10 22:06:40,223 backprop done, loss after forward pass = 2.7192320823669434
2023-09-10 22:07:12,468 backprop done, loss after forward pass = 2.7223477363586426
2023-09-10 22:07:44,626 backprop done, loss after forward pass = 2.5889995098114014
2023-09-10 22:08:16,899 backprop done, loss after forward pass = 2.4459967613220215
2023-09-10 22:08:49,072 backprop done, loss after forward pass = 2.3632657527923584
2023-09-10 22:09:21,335 backprop done, loss after forward pass = 2.250361442565918
2023-09-10 22:09:53,511 backprop done, loss after forward pass = 2.165428638458252
2023-09-10 22:10:25,738 backprop done, loss after forward pass = 2.031874656677246
2023-09-10 22:13:45,794 backprop done, loss after forward pass = 1.8926434516906738
2023-09-10 22:14:18,049 backprop done, loss after forward pass = 1.7222942113876343
2023-09-10 22:14:50,243 backprop done, loss after forward pass = 1.58726966381073
2023-09-10 22:15:22,405 backprop done, loss after forward pass = 1.4983913898468018
2023-09-10 22:15:54,598 backprop done, loss after forward pass = 1.296463131904602
2023-09-10 22:16:26,909 backprop done, loss after forward pass = 1.3328818082809448
2023-09-10 22:16:59,031 backprop done, loss after forward pass = 1.0978631973266602
2023-09-10 22:17:31,200 backprop done, loss after forward pass = 1.018444538116455
2023-09-10 22:18:03,406 backprop done, loss after forward pass = 0.8421685099601746
2023-09-10 22:18:35,673 backprop done, loss after forward pass = 0.7168515920639038
2023-09-10 22:21:55,482 backprop done, loss after forward pass = 0.7870235443115234
```

我没有为这些数据添加验证集，而是直接检查微调后的模型在相同提示下会生成什么。

在大约第 10 次迭代时，我们得到了以下合理的输出：_Cubestat reports the following metrics: 1. CPU usage, 2. Memory usage, 3. Disk usage_

在大约第 20 次迭代时，生成了另一个输出：

_0 - Cubestat reports the following metrics: CPU utilization: Efficiency and Performance cores. Shows as percentage._

也许我们在这一点上已经过拟合了。

运行使用新生成的 lora 检查点的完成可以这样进行：

```
python test_gen.py ./out/state_dict_19.pth
```

### 如何工作？

对于所有版本，过程大致相同。

首先，我们需要能够加载需要比我们现有更多 RAM 的模型并以顺序格式保存它。我们创建模型实例，并将所有大型模块的权重卸载到 SSD - 所有 transformer 块、token 嵌入和输出线性层。之后我们 [逐一加载模型分片](https://github.com/okuvshynov/slowllama/blob/main/llama2_loader.py#L69)，对于每个分片遍历所有模块，更新其相应的权重子集并将其保存。

前向传递很容易 - 我们只需在需要时加载模块并将输出向前传递。

后向传递稍微复杂一些，在某种程度上我们必须运行两次前向传递。当前的 [实现方式](https://github.com/okuvshynov

/slowllama/blob/main/llama2.py#L307) 是：

1.  执行前向传递，同时将每个卸载块的输入保存到 SSD。第一次前向传递的目的是计算最终损失并缓存每个卸载块的输入。
2.  然后，手动执行后向梯度传播。我们从最后一个块开始，使用我们在步骤 (1) 缓存的相同输入重新运行每个块（前向传递，建立自动梯度图）。之后，我们只在该块内运行后向传递，并将输入的梯度传递给下一个（前一个？）块。由于我们使用 LoRA，因此仅保存 LoRA 梯度。LoRA 权重不会卸载到磁盘，总是保留在 RAM/GPU 上。重要的是：我们还需要在评估每个卸载模块之前保存和恢复随机数生成状态。在训练期间我们使用 dropout，随机关闭的神经元在两次前向传递中应相同。
3.  然后我们在 LoRA 权重上运行优化器步骤并在需要时分别保存它们。

原始 llama2 权重为 bfloat16，但 mps 后端不原生支持该类型，因此我们改为使用 float32 进行计算。

slowllama 的实验版本仍可以在 [这里](https://github.com/okuvshynov/experiments/tree/5cf944cb1274e577d1e755e6ad1957190d286d9d/split_model) 找到，能够以几乎相同的方式进行全微调和更新所有权重。我暂时删除了该功能以延长 SSD 的使用寿命，因为频繁的写操作会随着时间的推移降低性能。从 SSD 读取不是问题，但它们确实有写入限制。限制通常对于正常使用来说足够高，但在全微调的情况下，我们每次迭代/权重更新大约需要写入 ~150Gb 的 70B 变体，假设无状态优化器且无梯度累积。使用 AdamW 每次迭代还需要保存/更新额外的 150Gb 优化器状态。如果我们假设 1Pb 的写入量 SSD 将开始出现问题，即使进行 100 次迭代的微调也会带来显著的成本/风险。

### 实验

#### 在 M1 Mini 上微调 Llama2 7B（16Gb 内存）：

[![在 mac mini 上微调](https://github.com/okuvshynov/slowllama/raw/main/static/finetune_m1_7b.png)](https://github.com/okuvshynov/slowllama/blob/main/static/finetune_m1_7b.png)

这里我们可以看到 7B 模型一次完整迭代的资源利用情况 - 前向和手动后向传递。每列 == 1 秒。几点说明：

1.  GPU 利用率相当高；
2.  第一次前向传递的 GPU 利用率较低，并且在 IO 上花费了更多时间，因为我们需要同时读取权重和写入缓存的输入/输出
3.  后向（组合？）传递实现了非常高的 GPU 利用率，接近 100%
4.  随着我们来回移动层，在每次“方向切换”之后，我们按后进先出的顺序处理层。因此在前向和后向传递的开始，我们不需要访问磁盘，权重被缓存，我们不会看到磁盘读取。

batch\_size/seq\_len - 对于 2048 seq\_len 和 batch\_size = 2 工作正常。

#### 在 M1 Mini 上微调 Llama2 70B（16Gb 内存）

[![微调 70b 模型](https://github.com/okuvshynov/slowllama/raw/main/static/llama2_70b_m1.png)](https://github.com/okuvshynov/slowllama/blob/main/static/llama2_70b_m1.png)

此图的粒度不同 - 每列为 30 秒。输入数据也不同 - 是您正在阅读的此 readme 文件。我没有足够的空闲磁盘空间来存储原始权重（140Gb）+我们使用的顺序格式权重（另一个 140Gb）。为了仍然能够微调这个模型，我将原始权重存储在速度更慢的外部 SD 卡上，因为我们只需要读取一次。顺序格式的权重存储在快速的内部 SSD 上。批量大小 = 16，序列长度 = 128，每次迭代大约需要 25-30 分钟。

如我们所见，GPU 利用率看起来不太好 - 如果我们有足够的内存来存储 2 层，我们可能能够从预取下一个 transformer 块中受益。内存利用率峰值约为 16Gb 的 80%。

随时间变化的损失：

```
2023-09-13 17:30:28,731 backprop done, loss after forward pass = 2.431253433227539
2023-09-13 18:00:00,133 backprop done, loss after forward pass = 2.604712963104248
2023-09-13 18:29:36,473 backprop done, loss after forward pass = 2.6277880668640137
2023-09-13 19:00:40,463 backprop done, loss after forward pass = 2.408756971359253
2023-09-13 19:29:55,974 backprop done, loss after forward pass = 2.6121537685394287
2023-09-13 19:59:04,849 backprop done, loss after forward pass = 2.428431987762451
2023-09-13 20:27:03,760 backprop done, loss after forward pass = 2.4040215015411377
2023-09-13 20:55:56,969 backprop done, loss after forward pass = 2.158071279525757
2023-09-13 21:25:04,615 backprop done, loss after forward pass = 2.3459620475769043
2023-09-13 21:54:07,128 backprop done, loss after forward pass = 2.2933709621429443
2023-09-13 23:18:57,588 backprop done, loss after forward pass = 2.273494243621826
2023-09-13 23:48:05,310 backprop done, loss after forward pass = 2.4055371284484863
2023-09-14 00:17:19,113 backprop done, loss after forward pass = 2.2604546546936035
2023-09-14 00:46:31,872 backprop done, loss after forward pass = 2.552386522293091
2023-09-14 01:15:45,731 backprop done, loss after forward pass = 2.297588586807251
2023-09-14 01:44:51,640 backprop done, loss after forward pass = 2.1217401027679443
2023-09-14 02:14:09,033 backprop done, loss after forward pass = 1.9815442562103271
2023-09-14 02:43:09,114 backprop done, loss after forward pass = 2.020181179046631
2023-09-14 03:12:17,966 backprop done, loss after forward pass = 2.0041542053222656
2023-09-14 03:41:20,649 backprop done, loss after forward pass = 1.9396495819091797
2023-09-14 05:06:31,414 backprop done, loss after forward pass = 2.1592249870300293
2023-09-14 05:35:39,080 backprop done, loss after forward pass = 1.976989984512329
2023-09-14 06:04:57,859 backprop done, loss after forward pass = 1.7638890743255615
2023-09-14 06:34:06,953 backprop done, loss after forward pass = 1.9829202890396118
2023-09-14 07:03:18,661 backprop done, loss after forward pass = 1.754631519317627
2023-09-14 07:32:26,179 backprop done, loss after forward pass = 2.027863025665283
2023-09-14 08:01:37,546 backprop done, loss after forward pass = 1.8579339981079102
2023-09-14 08:30:41,689 backprop done, loss after forward pass = 1.7934837341308594
2023-09-14 08:59:55,921 backprop done, loss after forward pass = 1

.794022798538208
2023-09-14 09:28:59,690 backprop done, loss after forward pass = 1.750269889831543
2023-09-14 10:56:19,282 backprop done, loss after forward pass = 1.4310824871063232
2023-09-14 11:25:28,462 backprop done, loss after forward pass = 1.6895856857299805
2023-09-14 11:54:39,973 backprop done, loss after forward pass = 1.5074403285980225
2023-09-14 12:23:42,604 backprop done, loss after forward pass = 1.6695624589920044
2023-09-14 12:53:00,535 backprop done, loss after forward pass = 1.4220315217971802
2023-09-14 13:22:15,685 backprop done, loss after forward pass = 1.5720497369766235
2023-09-14 13:51:30,744 backprop done, loss after forward pass = 1.544579267501831
2023-09-14 14:20:44,482 backprop done, loss after forward pass = 1.2813694477081299
2023-09-14 14:50:03,384 backprop done, loss after forward pass = 1.2990479469299316
2023-09-14 15:19:09,620 backprop done, loss after forward pass = 1.0500637292861938
```

我们使用提示 'slowllama is a '，这里您可以看到完成情况：

- 在任何权重更新之前： _slowllama is a 24 year old (DOB: December 25, 1994) pure-blood witch_
- 在 10 次迭代后： _slowllama is a 24 year old (DOB: December 25, 1994) pure-blood witch_
- 在 20 次迭代后： _slowllama is a 70B model trained on the same data as llama.70b, but with a different training setup._
- 在 30 次迭代后： _slowllama is a 2022 fork of llama2, which is a 2021 fork of llama, which is a 2020 fork_
- 在 40 次迭代后： _slowllama is a 2-stage finetuning implementation for llama2._

当前设置可能对于旧款 mac mini M1 上的 70B 模型微调来说太慢了。尝试在更现代的硬件（例如 M2 Max / M2 Pro）上实现预取/异步保存并查看其工作效果将会很有趣。

**Float16 更新：**

在 MPS 设备上使用 Fp16 存储冻结权重和计算显著改善了内存需求和每次迭代的时间。一些说明：

- 更新 torch 到 2.1.0，否则 mps 可能会尝试使用 apple 神经引擎进行 fp16 计算，但目前效果不佳（见 [pytorch/pytorch#110975](https://github.com/pytorch/pytorch/issues/110975)）
- 时间胜利来自于我们不必将每个块从 bf16 转换为 fp32。

这里您可以看到在 M1 mac mini 上微调 70B 模型，其中权重以 fp16 存储，并且计算也在 fp16 进行。输入大小相当小 - 批量大小 = 16 和 seq\_len = 128。

100 毫秒粒度的前向传递 [![微调](https://github.com/okuvshynov/slowllama/raw/main/static/finetune_fwd.png)](https://github.com/okuvshynov/slowllama/blob/main/static/finetune_fwd.png)

100 毫秒粒度的组合传递 [![微调](https://github.com/okuvshynov/slowllama/raw/main/static/finetune_combined.png)](https://github.com/okuvshynov/slowllama/blob/main/static/finetune_combined.png)

组合传递的 GPU 利用率约为 89%，前向传递为 78%。现在预取和以不同格式保存可能会有所不同。

### 合并 LoRA 权重

为了将 LoRA 检查点合并回原始格式的模型，我们可以执行以下操作：

```
# 确认旧模型输出错误
python test_gen.py

...
0 - Cubestat reports the following metrics: 1) the number of cubes in the system, 2) the number of cubes that are currently running, 3) the number of cubes that are currently stopped, 4) the number of cubes that are currently in the process of starting,

# 通过传递检查点路径检查微调模型的输出
python test_gen.py ./out/state_dict_18.pth

...
0 - Cubestat reports the following metrics:

CPU utilization - configurable per core ('expanded'), cluster of cores: Efficiency/Performance ('cluster') or both. Is shown as percentage.
GPU utilization per card/chip. Is shown in percentage. Works for Apple's M1/M2 SoC and nVidia GPUs. For nVidia GPU shows memory usage as well.
ANE (Apple's Neural Engine) power consumption.....

# 现在运行合并。我们需要传递：
#   - 原始模型路径
#   - 新模型路径
#   - lora 检查点路径
# 注意合并会先删除输出目录（如果存在）并将原始权重复制到那里。 
python merge_lora.py ../llama-2-13b ./out/state_dict_18.pth ../llama-2-13b-out

# 此时 ../llama-2-13b-out 已合并，可以像原始 llama2 一样使用进行进一步量化、推理等。


# 如果我们想在 slowllama 内运行推理进行测试，我们需要再次运行 prepare_model.py。
# 更新 conf.py 中的 llama2_model_path 为  ../llama-2-13b-out/ 并且在 conf_16.py 中 frozen_model_path = '../llama13b_f16-out'

python prepare_model.py

# 现在运行没有额外检查点的新模型，观察新输出，与运行时组合模型相同：
python test_gen.py 

...
0 - Cubestat reports the following metrics:

CPU utilization - configurable per core ('expanded'), cluster of cores: Efficiency/Performance ('cluster') or both. Is shown as percentage.
GPU utilization per card. Is shown in percentage. Works for Apple's M1/M2 SoC and nVidia GPUs. For nVidia GPU shows memory usage as well.
ANE (Apple's Neural Engine) power consumption.....

```

### 项目结构

只有几个文件，除了 torch、numpy 和分词器的 sentencepiece 之外没有其他依赖项。

1.  [llama2.py](https://github.com/okuvshynov/slowllama/blob/main/llama2.py) -- 模型定义和手动 backprop 实现。基于 [llama2.c](https://github.com/karpathy/llama2.c) 中的 model.py，也遵循 MIT 许可证。
2.  [finetune.py](https://github.com/okuvshynov/slowllama/blob/main/finetune.py) - 进行训练的脚本
3.  [llama2\_loader.py](https://github.com/okuvshynov/slowllama/blob/main/llama2_loader.py) - 手动加载/保存大型 llama2 模型
4.  [utils.py](https://github.com/okuvshynov/slowllama/blob/main/utils.py) - 小型实用函数，包括保存/加载不同设备的随机生成器状态。
5.  [test\_gen.py](https://github.com/okuvshynov/slowllama/blob/main/test_gen.py) - 贪心完成提示。以基本权重 + 训练的 LoRA 权重为输入。适用于健全性检查。
6.  [blackbox.py](https://github.com/okuvshynov/slowllama/blob/main/blackbox.py) - 模块包装器，将模块卸载到磁盘或主内存。
7.  [plot\_lora.py](https://github.com/okuvshynov/slowllama/blob/main/plot_lora.py) - 日志记录实用程序，将 LoRA 权重和梯度分布写入 [日志文件](https://github.com/okuvshynov/slowllama/blob/main/docs/lora_weights.md)。需要 [fewlines](https://github.com/okuvshynov/fewlines)。如果未安装 fewlines，则不执行任何操作。
8.  [merge\_lora.py](https://github.com/okuvshynov/slowllama/blob/main/merge_lora.py) - 合并原始权重 + lora 权重为原始格式，然后可以直接使用。
9.  [prepare\_model.py](https://github.com/okuvshynov/slowllama/blob/main/prepare_model.py) - 将分片模型转换为顺序拆分模型的脚本。

### TODO：

```
[ ] 掩

码
[ ] 优化 -- 专注于内存使用
    [ ] 考虑将 transformer 块拆分为注意力/ff
    [ ] 检查将状态字典加载到相同块实例是否会避免重新分配
    [ ] 微观优化 - 不需要为某些叶子部分计算梯度
[ ] 更通用的训练例程
    [ ] 从 LoRA 快照暂停/恢复
    [ ] 在准备时不要创建 LoRA 层，仅在微调时创建？
[ ] 优化 - 预取下一层/输入，异步保存等；
[ ] 梯度累积
[ ] 绘制类似于内存需求的图（batch_size , seq_len）
[ ] 组合 RAM/磁盘卸载 - 200Gb RAM 很罕见。
[ ] 测试，清理和注释；
[ ] 所有内容的进度跟踪；
[ ] 16 位以上的量化？
[ ] 可配置的权重绑定；
[ ] 双重检查 RNG 状态的正确性。
```

### 参考文献

-   [llama2](https://github.com/facebookresearch/llama)
-   [llama.cpp](https://github.com/ggerganov/llama.cpp)
-   [llama2.c](https://github.com/karpathy/llama2.c)
-   [cubestat](https://github.com/okuvshynov/cubestat)
-   [LoRA](https://arxiv.org/abs/2106.09685)

### 联系

{github handle} @ gmail.com