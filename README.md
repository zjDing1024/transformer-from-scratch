# Transformer from Scratch

**Day 1: Transformer 机制与正确性验证。** 用 PyTorch 基础张量运算原创实现 scaled dot-product attention、多头注意力、sin/cos 位置编码、LayerNorm、残差连接和一个 encoder block。

这是独立的机制实现项目。**尚未实现完整 encoder-decoder 训练，也没有复现论文的 WMT/BLEU 指标。** 小型合成实验只验证前向、反向与优化器连接，不代表语言理解或泛化能力。

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

1. 实现 decoder block、cross-attention 组合、token embedding 与 shifted target，验证自回归不泄漏。
2. 用明确许可的小型任务做训练/验证拆分，记录学习曲线与失败案例。
3. 对照 fused SDPA 做有控制的速度/显存实验，再评估 pre-norm、缓存和混合精度。

## 来源和许可证

理论来源：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)，Vaswani 等。阅读范围与实现差异见 [source-study.md](docs/source-study.md)。对齐参考为 PyTorch 官方 API 和源码，**没有复制其实现代码或论文图表**。

原创代码采用 [MIT](LICENSE)。依赖及参考材料保留各自许可，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
