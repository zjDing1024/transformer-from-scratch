# 论文与官方实现阅读

日期：2026-11-12。

## 研究对象

- 论文：Vaswani et al., *Attention Is All You Need*, 2017，[arXiv](https://arxiv.org/abs/1706.03762)。阅读架构、attention、FFN、位置编码和训练说明。本项目只实现其中的基础机制，未运行论文训练配置。
- 官方工程参考：[pytorch/pytorch](https://github.com/pytorch/pytorch)，作者/维护者为 PyTorch contributors。PyTorch 提供张量、autograd、算子和模块系统，适合把自己实现的计算与可信接口交叉验证。
- 实际检查了已安装 PyTorch 2.14.1 的 `TransformerEncoderLayer.forward`、`_sa_block`、`_ff_block`，比较 post-norm 顺序和 dropout 位置。在线源码入口：[transformer.py](https://github.com/pytorch/pytorch/blob/v2.14.1/torch/nn/modules/transformer.py)。
- API 语义：[SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)、[LayerNorm](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html)。

## 从原理到工程

注意力先比较 query/key，再通过概率对 value 聚合。除以每头 key 维度的平方根，使 logits 尺度不随维度直接放大。多个头分别投影并汇合。位置编码提供顺序信号，FFN 对每个位置使用相同权重。残差路径与归一化共同构成 block。

本项目把形状契约、遮罩语义和全遮罩行行为写成 API 约定，再用公式测试、官方接口对齐和有限差分检查验证它们。浮点误差存在，不要求不同内核逐位相等。

## 官方源码值得学习的部分

官方 `torch/nn/modules/transformer.py` 组织 encoder/decoder，`activation.py` 包含多头模块，`functional.py` 提供函数接口。工程实现必须同时考虑 padding、训练/评估模式、不同 dtype 和高效路径，不能只写出数学公式就认为完成。

本项目不搬运这些实现。采用独立的 Q/K/V Linear 模块和明确返回的每头权重，便于观察；官方多头模块使用合并投影并提供多条优化路径。测试通过映射参数、反转 boolean mask 对齐两种 API，检查输出与梯度。

## 范围差异

- 只有一个可组合 encoder block；没有完整 decoder、tokenizer、输出移位、权重绑定、beam search。
- 使用 post-norm，默认 dropout=0，便于确定性测试。FFN 隐层 dropout 对齐 PyTorch encoder 结构；论文复现应另行固定训练配置。
- 支持奇数维位置编码和 offset，是工程扩展。
- 全遮罩行定义为零 attention，并保证 backward 有限。原始 softmax 全为负无穷会出现 NaN，因此先替换空行 logits，再清零概率。
- 没有 WMT 数据、BLEU、论文规模参数、学习率计划或多 GPU 训练。不能用当前 smoke loss 评价论文复现质量。

## 对求职展示的价值

可展示张量布局、数值边界、autograd、基准实现对齐和可复现验证。尚不足以证明 NLP 任务建模、真实语料训练或生产推理能力，这些需要下一阶段独立证据。
