# Transformer from Scratch

**机制验证 + 真实文本语言模型完整训练路径。** 用 PyTorch 基础张量运算原创实现 scaled dot-product attention、多头注意力、sin/cos 位置编码、LayerNorm、残差连接和一个 encoder block。

这是独立的机制实现项目。**尚未实现完整 encoder-decoder 训练，也没有复现论文的 WMT/BLEU 指标。** 历史小型合成实验保留；新增真实技术文本的99,968参数因果语言模型、固定文档划分、验证集选模、bigram基线及独立测试评分。不代表语言理解或个人技能已掌握。

## 同语义CPU缓存基准（2026-11-18，UTC+08:00）

三组固定请求，按30组交替顺序配对计时；每组各5次完整请求，parity检查在计时外。
暖完整请求中位数（完整前缀→缓存）为 **27.103→19.842、27.997→19.771、80.628→40.961ms**。
构造快照单列 **3.445ms**；prefill并非全赢，不宣称普遍提速。

**本次原checkpoint重建未通过身份校验。** 计时前已冻结透明修订：只用同一份已重建的新工件
比较两条路径，没有重复训练挑模型。18个进程的RSS读数无区分力，不能据此比较内存；
31/62/126KiB仅是K/V张量payload，不是总内存。

[完整分布、身份失败、内存限制与运行方法](docs/cpu-kv-benchmark.md) ·
[冻结方案](docs/cpu-kv-protocol.md) · [原始记录](results/cpu-kv-v1/report.json)。**191项CPU测试与离线记录审计通过**。

## 可审计 KV-cache 推理（2026-11-17，UTC+08:00）

新增有界CPU推理会话：权重快照、prefill、单token/分块append、reset，以及可安装的
`transformer-cached-generate` 命令。原模型与训练路径不改，不是重复SQLite评分。

- 恢复一份原checkpoint，文件与tensor摘要均完全匹配；与完整前缀forward比较，
  最大logit绝对差 **1.072884e-5**，通过预设 **atol=rtol=1e-5组合容差**，不是逐bit相同。
- 两行各16步greedy字节完全一致，报告新进程重放逐字节一致；**163项CPU测试通过**。
- 严格总上下文预算，不偷偷滑窗或重置位置；仅同长无padding批次、CPU float32/64。
  未测试吞吐/延迟，不宣称提速或生产服务能力；生成仍重复、不连贯。

[接口、运行命令、精确结果与边界](docs/cached-inference.md) ·
[验证协议](docs/inference-protocol.md) · [逐chunk/逐步报告](results/inference-v1/report.json)。

## 同目标、等历史长度格式对照（2026-11-16，UTC+08:00）

对同一批SQLite prose目标，分别输入紧邻的64字节raw或保守纯文本投影历史，只评分末位目标，
不沿用旧block reset。先冻结768个changed-history样本及192个完全相同history负控，复用原六模型。

- 投影−raw NLL：有位置 **−0.046488**，无显式位置 **−0.039231**（3种子均值）。
- 负控逐目标差为 **0**，评分重放逐字节一致；107项CPU测试通过。
- **此抽样上bigram优于全部六模型**；18个种子/位置设置/文档组合中有3个反而变差，不宣称普遍清理收益。

[冻结协议](docs/format-control-protocol.md) · [抽样、每文档结果、干预边界与重跑](docs/sqlite-format-control.md)。
投影不是完整官方渲染；等字节预算不等于相同语义跨度，不能从本实验推断跨域泛化。

## 同 checkpoint 外部评估（2026-11-15，UTC+08:00）

保留原选模锁，用同六个模型评估固定版本的 SQLite 官方文档，共52,351目标。
旧本地权重缺失后按原协议重建，六份文件SHA256与tensor摘要均和原记录完全一致；
这是权重恢复，不是新训练成果。未微调、未重新选模、未对外部数据调参。

- 有位置 byte NLL：**3.064089 ± 0.015219**；无显式位置：**3.074392 ± 0.015358**。
- 固定Flask训练集bigram：**3.089247**。一个无位置种子略差于bigram，平均优势很小。
- 数据同时改变项目/主题和源码格式（RST→HTML/模板`.in`），**不是纯域偏移或稳健泛化证明**。

[冻结协议](docs/transfer-protocol.md) · [恢复身份、完整结果与重跑命令](docs/sqlite-transfer.md) ·
[原始报告](results/sqlite-transfer-v1/report.json)。89项CPU测试通过；同输入评分重放JSON逐字节一致。

## 固定预算配对实验（2026-11-14，UTC+08:00）

新增 **3 种子 × 有/无显式位置编码** 对照，不把昨日600步单次重复当作新结果。
先冻结协议，每臂同初始化、同训练采样、同99,968参数、同600步预算；全部验证选模
完成后，才另进程对3份新文档的25,532个目标评分。旧test不参与本次选择或评分。

- 有位置编码 byte NLL：**2.414795 ± 0.006506**。
- 无显式位置编码：**2.440424 ± 0.019620**。
- 配对 NLL 差均值（无位置减有位置）：**+0.025629**，三个种子方向一致。
- ± 为样本标准差，n=3只作描述，**不宣称显著性、跨域泛化或普遍优势**。

[冻结方案](docs/stability-protocol.md) · [重跑命令、完整结果和边界](docs/position-study.md) ·
[机器可读报告](results/position-v1/report.json)。74项CPU测试通过；权重与下载正文不入库。

## 真实文本里程碑（2026-11-13，UTC+08:00）

从手写 attention/norm 出发，新增 byte tokenizer → causal LM →600步训练→验证集选checkpoint→test/bigram同目标评分→保存→新进程生成。不是继续拆分合成小实验。

```bash
# 安装同实验CPU版本及开发依赖
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
# 显式联网下载固定commit的官方文档并验证SHA256/保留BSD许可证
python examples/download_text.py --output-dir corpus/flask
python examples/train_text.py --data-dir corpus/flask --output-dir results/text-local
python examples/generate_text.py --checkpoint results/text-local/checkpoint.pt
```

无需GPU、付费API或密钥。下载约96KB，训练全程CPU；正文不提交到仓库。
Flask官方文档按文件分train/validation/test，固定256字节词表，无跨文档window。
600step固定预算、seed20261113；验证集选择step600，test10,447目标一次评分。
byte PPL **11.2362 vs bigram14.1115**，bits/byte **3.49009 vs3.81880**。
这仅是单seed/单文档证据；生成样本仍明显不连贯，不宣称可用语言生成。

[冻结协议](docs/text-protocol.md) · [完整实验与失败分析](docs/text-experiment.md) ·
[原始曲线](results/text-v1/text-training.json) · [数据摘要](data/text-manifest.json)。
59项唯一CPU测试通过；权重仅本地生成，仓库发布源代码与可复核结果。

## 为什么做这个项目

把「能调用 Transformer」推进到「能解释张量维度、遮罩语义、归一化和梯度流，并用参考实现验证」。相比已有 [pytorch-from-zero](https://github.com/zjDing1024/pytorch-from-zero) 的训练工程，这里专注模型机制，代码与运行入口独立。

## 快速运行

Python 3.10+；实测 Python 3.12.14、PyTorch 2.14.1+cpu。无需 GPU、API key 或数据下载。

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
python -m pytest -q
python examples/verify_mechanisms.py --output results/local.json
python -m ruff check .
python -m ruff format --check .
```

不安装项目也可在已装依赖的环境运行：`PYTHONPATH=src python examples/verify_mechanisms.py`。执行测试不需要网络；安装依赖需要访问官方包源。

## 结构与实现

```text
src/transformer_core/
  attention.py    # QK^T 缩放、遮罩、softmax、V 加权、多头投影
  layers.py       # 位置编码、手写归一化、残差、FFN、encoder
examples/verify_mechanisms.py
tests/           # 公式检查、官方参考对齐、梯度、异常输入
docs/            # 设计说明、论文/源码阅读、验证边界
results/day1.json
```

encoder 路径：`x -> attention -> Add & Norm -> FFN -> Add & Norm`。采用 post-norm；位置编码独立于 block。实现显式保存 `[B,H,L,S]` 注意力矩阵，便于检查，不使用 fused SDPA 或 `nn.MultiheadAttention` 代替核心逻辑。

```python
import torch
from transformer_core import EncoderBlock, SinusoidalPositionalEncoding, causal_mask

x = torch.randn(2, 6, 16)  # 已有连续特征/嵌入，不是 token ID
x = SinusoidalPositionalEncoding(16)(x)
block = EncoderBlock(d_model=16, num_heads=4, d_ff=32)
y = block(x, allowed_mask=causal_mask(6))
assert y.shape == x.shape
```

普通 encoder 不需要 causal mask；例子加上它只是展示遮罩。未来 decoder 会使用因果遮罩和 cross-attention。

## 关键约定

- 所有布尔遮罩都是 **True=允许参与**，与 `F.scaled_dot_product_attention` 一致，与 `nn.MultiheadAttention` 的遮罩正好相反。对齐测试显式取反。
- attention 输入为 `[B,H,L,Dk] / [B,H,S,Dk] / [B,H,S,Dv]`；多头接口为 batch-first `[B,L,D]`。
- `allowed_mask` 接受二维或四维广播形式；三维形式拒绝，以免 batch/head 含义不明确。`key_keep_mask` 为 `[B,S]`。
- 全遮罩行产生零 attention 概率与零加权值，反向有限。多头输出经过带 bias 的投影后可以是 bias，不承诺整层为零。
- key padding mask 只移除 key，**不会自动清零 padded query**。未来 loss 和 pooling 必须另外排除无效 query。
- 输入必须有限。half/bfloat16 的 attention 与归一化中间计算升至 float32；本项目还没有验证 CUDA AMP。
- 位置编码支持奇数维和 offset；调用者自己负责 token embedding 的 `sqrt(d_model)` 缩放。

## 已运行的证据

记录日期：2026-11-12。固定 seed=20261112，CPU 单线程。

- float64 attention 前向对齐最大绝对误差：`6.66e-16`。
- attention 输入梯度对齐最大绝对误差：`4.44e-16`。
- LayerNorm 前向对齐最大绝对误差：`4.44e-16`。
- 因果遮罩的未来位置非零权重数：`0`。
- 合成固定 batch，2,343 参数，SGD 10 步，交叉熵 `2.055674 -> 1.465322`。仅是优化器 smoke test，没有 held-out set。
- 完整测试与检查命令、依赖限制见 [验证记录](docs/validation.md)；机器可读结果见 [day1.json](results/day1.json)。

## 性能与取舍

注意力的计算随 `L*S` 增长，保存权重需要 `O(B*H*L*S)` 内存。代码优先可读性和数值检查，仅提供固定CPU请求的描述性基准，不提供普遍吞吐优势或生产性能结论。新增基础KV cache仅验证有界推理正确性；没有 FlashAttention、变长批处理内核或分布式训练。

## 下一步

1. 已完成causal decoder-only真实文本路径、可安装的有界缓存生成入口与单checkpoint精确恢复；训练入口封装仍待完成。
2. 已完成冻结的三seed配对位置编码消融及新文档留出；后续另立协议/新评估集，不能将本次test回流调参。
3. 再评估pre-norm/长上下文与fused SDPA；cross-attention翻译模型仍未实现。

## 来源和许可证

理论来源：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)，Vaswani 等。阅读范围与实现差异见 [source-study.md](docs/source-study.md)。对齐参考为 PyTorch 官方 API 和源码，**没有复制其实现代码或论文图表**。

原创代码采用 [MIT](LICENSE)。依赖及参考材料保留各自许可，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
