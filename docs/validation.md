# Day 1 验证记录

日期：2026-11-12。范围：原创 Transformer 机制，不是完整论文指标复现。

## 实际环境

- Linux，CPU，Python 3.12.14，PyTorch 2.14.1+cpu。
- pytest 9.1.1，Ruff 0.16.10，测试固定随机种子；数值实验单线程。
- 复用已有免费 CPU 执行环境，没有外部模型 API、付费算力或私有训练数据。
- 环境没有 NumPy，PyTorch 导入会产生一条可见警告；本项目只使用 Torch 与 Python 标准库，全部测试仍通过。没有隐藏该警告。
- `pip check` 无破损依赖。本次没有在全新环境重新下载所有依赖，因此不能把包源可用性写为已验证。

## 完成的检查

1. `python -m pytest -q`：**52 passed**，无 skip / xfail；一条上述 NumPy 警告。
2. `python -m ruff check .` 与 `python -m ruff format --check .`：通过。
3. `python examples/verify_mechanisms.py --output results/day1.json`：通过，全部内置阈值检查通过。
4. `python -m pip wheel . --no-deps --no-build-isolation --wheel-dir dist`：构建成功。
5. 将 wheel 安装到独立临时目录，从项目目录之外运行同一例子，验证加载的是 wheel 包，而不是 `src/`；JSON 与源码路线逐字相同，见 `results/wheel-verification.json`。
6. 独立复核新增 7 项检查；复核者在修复后独立重跑，7 项全过。

## 测试覆盖

- 手算两 key 的缩放/softmax/加权值，与官方 SDPA 的前向、Q/K/V 梯度对齐。
- 因果未来不可见、padding key 不泄漏、二维/四维和每头遮罩、交叉注意力不同序列长度。
- 全遮罩行输出、权重与梯度为零，既检查 value 输出反向，也检查返回权重反向。
- 多头输出、每头权重、投影权重梯度与 `nn.MultiheadAttention` 对齐。
- 标量 sin/cos 公式、奇数维度、offset 分块一致、输入梯度和 dtype。
- LayerNorm 前向、输入/affine 参数梯度、常量/单特征输入、half/bfloat16、有限差分 gradcheck。
- post-norm 残差、FFN 位置独立性，完整 encoder 与官方 `TransformerEncoderLayer` 的输出和所有参数梯度对齐。
- eval dropout 确定性、state_dict 回放、全 padding 的训练反向、无效输入错误路径。
- 源码路线和 wheel 路线的实验结果一致。

## 复核中修复的问题

外层 autocast 可以把已转换为 float32 的矩阵乘法重新降为 float16，导致 Q=K=300 这样的有限输入出现溢出。已在显式 attention 数学运算内禁用 autocast，再按 API 转回输入 dtype。CPU autocast 回归现在通过；**没有声称 GPU/AMP 训练全部通过**。

## 数值结果

`results/day1.json` 保存准确数值、seed、环境和每步损失：

- attention 最大绝对误差 `6.661338147750939e-16`。
- attention 梯度最大绝对误差 `4.440892098500626e-16`。
- LayerNorm 最大绝对误差 `4.440892098500626e-16`。
- 未来位置非零 attention 权重 `0`。
- 10 次 SGD 更新，合成固定 batch loss `2.0556743144989014 -> 1.465321660041809`。

这些是特定小张量和已列测试上的正确性证据，不是所有输入的形式证明。没有真实数据集评测、held-out accuracy、BLEU 或泛化结论。

## 尚未验证

CUDA、MPS、分布式、长上下文压力测试、混合精度完整训练、torch.compile、模型序列化文件往返，以及不同版本的 PyTorch。CPU workflow 已写入仓库；在远端实际运行前不能称为 GitHub CI 通过。

## 发布前恢复复跑

从已交付压缩包恢复，压缩包 SHA-256 与内部 26 项文件摘要全部匹配。恢复后再次运行 52 项测试、Ruff 检查和验证例子，全部通过。`results/publication-recheck.json` 保留这次输出。不同执行实例的浮点末位有差异：attention/gradient 最大误差仍在 `1e-15` 内，个别中间 float32 loss 相差约 `2.4e-7`；首尾损失不变。不承诺不同 CPU 实例逐字复现。源码与 wheel 逐字匹配的结论只指前面记录的同一实例检查。
