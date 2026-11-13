# Transformer from Scratch

**机制验证 + 真实文本语言模型完整训练路径。** 用 PyTorch 基础张量运算原创实现 scaled dot-product attention、多头注意力、sin/cos 位置编码、LayerNorm、残差连接和一个 encoder block。

这是独立的机制实现项目。**尚未实现完整 encoder-decoder 训练，也没有复现论文的 WMT/BLEU 指标。** 历史小型合成实验保留；新增真实技术文本的99,968参数因果语言模型、固定文档划分、验证集选模、bigram基线及独立测试评分。不代表语言理解或个人技能已掌握。

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

注意力的计算随 `L*S` 增长，保存权重需要 `O(B*H*L*S)` 内存。代码优先可读性和数值检查，不提供吞吐优势或生产性能结论。没有 FlashAttention、KV cache、变长批处理内核或分布式训练。

## 下一步

1. 已完成causal decoder-only真实文本路径；下一步将训练/生成封装为installed入口并完善checkpoint恢复。
2. 已完成冻结的三seed配对位置编码消融及新文档留出；后续另立协议/新评估集，不能将本次test回流调参。
3. 再评估pre-norm/长上下文与fused SDPA；cross-attention翻译模型仍未实现。

## 来源和许可证

理论来源：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)，Vaswani 等。阅读范围与实现差异见 [source-study.md](docs/source-study.md)。对齐参考为 PyTorch 官方 API 和源码，**没有复制其实现代码或论文图表**。

原创代码采用 [MIT](LICENSE)。依赖及参考材料保留各自许可，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
