# 生成先验、几何约束盲去噪与原增强链联合方案 v5

## 结论

v5 已按“生成先验 → 盲去噪内几何约束 → 原有增强/后处理 → 独立长度审计”完成实现和单图验证。推荐输出为：

```text
results_sota/02_geometry_quality/QUALITY_boundary_preserved_16bit.tif
```

在本张 2200 × 1600、16位图像上，最终平坦区域高频粗糙度较原图下降 **28.66%**，数字边缘锐度提升 **5.60%**，局部对比度提升 **2.04%**。最大左右中位 FWHM 变化为 **1.22%**。独立审计检测100条层纹，98条自动通过，2条因原始端点本身不确定而标记复核。

## 为什么采用该盲模型

本工程以 [Blind2Sound（ICCV 2025）](https://openaccess.thecvf.com/content/ICCV2025/html/Liu_Blind2Sound_Self-Supervised_Image_Denoising_without_Residual_Noise_ICCV_2025_paper.html) 的自适应 re-visible 与噪声建模为主要依据。该论文针对无干净真值的自监督盲去噪，并特别报告了单通道图像和 Poisson-Gaussian 噪声上的优势。对本任务而言，它比依赖自然图像大模型重建的通用恢复方法更适合作为测量链基础。

“目前最强”在此被严格限定为：截至实现时，在可获取论文与可复现实现中，对“单幅、单通道、未知 Poisson-Gaussian 噪声”这一条件最匹配的近期同行评审方案之一。这里是面向工业 CT 的 clean-room 工程适配，不是官方源码复刻，也不宣称在所有数据集上绝对第一。

## 完整处理顺序

```text
原始16位图 + 生成候选图
  → 生成图仅蒸馏为限幅低频先验
  → 4×4子晶格自盲训练 + 自适应re-visible似然
  → x/y边缘、横向宽度、纵向端点、多尺度形状损失
  → 16相位严格盲推理 + 方差估计
  → Poisson-Gaussian数据投影 + 原图层纹中心坐标锚定
  → 逐档选择最大安全去噪强度
  → 原有横向边缘增强、DoG结构增强、MAD限幅和羽化
  → 必要时才做亚像素端点几何校正
  → 原有结构感知细颗粒/雾状残留净化
  → 独立100层长度审计
```

## 1. 安全生成先验

生成候选先缩放到原图尺寸并做分位数亮度匹配。原图与生成图分别用 `σ=6` 的高斯核提取低频分量。只在原图低梯度位置保留先验差值：

\[
p=x+g_{low}\,\operatorname{clip}(G_6(y)-G_6(x),-c_p,c_p)
\]

其中 `x` 是原始图，`y` 是生成图，`c_p=max(1.25σ_n,4/65535)`。先验只进入低频损失，不提供峰中心、层数、端点或像素回填。

## 2. 自适应 re-visible 盲训练

训练每次从目标区域取 96 × 96 patch，并隐藏4 × 4子晶格中的一个相位。隐藏像素只由上下左右邻点插值输入网络。U-Net 输出信号均值与方差。设盲输入预测为 `(m_b,v_b)`，可见输入预测为 `(m_v,v_v)`，后者停止梯度；可见权重 `β` 在训练40%以后由3线性增加到11：

\[
m=\frac{m_b+\beta\,\operatorname{sg}(m_v)}{1+\beta},\quad
v=\frac{v_b+\beta^2\operatorname{sg}(v_v)}{(1+\beta)^2}
\]

可学习噪声方差为：

\[
v_n=\sigma_g^2+s_p\max(m,10^{-4})
\]

主要似然损失为：

\[
\mathcal L_{nll}=\frac{(x-m)^2}{v+v_n}+\log(v+v_n)
\]

并同时使用盲位置 Charbonnier 损失。

## 3. 训练内几何一致性

几何损失不是推理后的补丁，而是在训练时作用于每轮盲预测：

\[
\mathcal L_g=0.34L_{edge-x}+0.26L_{edge-y}+0.20L_{width}+0.12L_{endpoint}+0.08L_{shape}
\]

- `edge-x`：保护横向半高宽边缘；
- `edge-y`：保护上下端点；
- `width`：约束纵向平均后的横向轮廓梯度；
- `endpoint`：约束横向平均后的纵向轮廓梯度；
- `shape`：约束5 × 5尺度下的整体形状。

总损失为：

\[
\mathcal L=\mathcal L_{nll}+0.35L_{blind}+7.5L_g+0.04L_{prior}
\]

## 4. 16相位盲推理与几何投影

推理遍历全部16个子晶格相位。每个内部像素只采用其被隐藏时的预测，避免可见中心直接复制导致“看似保真但实际未去噪”。

模型均值和方差先与观测构成 Poisson-Gaussian 后验；再用原图梯度门保护强边缘。额外检测左右测量轮廓中的原始峰中心，在投影时锚定这些横坐标。锚点只是一列坐标约束，不写入生成图，也不进行后置原图块覆盖。

程序搜索投影残差强度 `0、0.65、0.80、1.00、1.20、1.50`，对每档重新检测100条层纹。只在宽度、峰中心、中央高亮、SSIM、端点和长度全部通过时，按平坦区域粗糙度下降最大原则选择。本图选择 `1.50`。

## 5. 原增强与最终去雾仍完整执行

盲模型之后继续运行原有：

1. 横向边缘门控增强；
2. DoG结构增强；
3. MAD改变量限幅；
4. 18 px ROI羽化；
5. 亚像素端点几何校正（仅在已有结果不合格时执行）；
6. `σ=0.90/2.20` 的结构感知细颗粒与雾状残留净化。

由于新盲基线更平滑，边缘强度搜索扩展至2.0，但像素改变量仍受原MAD上限控制。本图选择 `edge=2.0、structure=0.06`，雾净化选择 `fine=0.12、haze=0.015`。增强后的几何已经通过，因此未执行不必要的第二次端点重采样。

## 6. 数值结果

| 指标 | v4 | v5 |
|---|---:|---:|
| 高频粗糙度下降 | 26.28% | **28.66%** |
| 数字边缘锐度增益 | 4.89% | **5.60%** |
| 局部对比度增益 | **2.72%** | 2.04% |
| 最大中位FWHM变化 | **0.188%** | 1.218% |
| 端点偏移P95 | 0.260 px | **0.152 px** |
| 可靠层最大端点偏移 | 0.695 px | **0.486 px** |
| 长度差P95（集成） | 0.433 px | **0.259 px** |
| 长度差P95（独立） | 0.454 px | **0.287 px** |

v5 的FWHM变化高于v4，但仍远低于4%守卫；去噪、边缘锐度和端点/长度尾部误差更好。局部对比度略低于v4，是更强去噪的可见权衡。

## 7. 复现

```bash
docker compose run --rm ct-sota
```

或：

```bash
python app/run_sota_pipeline.py \
  --source input/source_16bit.tif \
  --generative-prior input/generative_candidate_visual_only.png \
  --outdir results_sota --iterations 600 --device cpu
```

单元测试：

```bash
python -m unittest discover -s tests -v
```

## 8. 限制

当前结论来自一张图的内部验证，没有配对无噪声真值、重复扫描和跨设备数据。像素坐标一致不等于毫米/微米计量认证；绝对长度仍需要像素尺寸、标准件、PSF/MTF和重复性标定。2条原始端点低置信层必须人工复核。

参考实现与比较来源：[Blind2Sound official repository](https://github.com/Jiazheng-Liu/Blind2Sound)、[TBSN official repository](https://github.com/nagejacob/TBSN)、[BIR-D NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/25869dbf7682272357bc2cbbf860e1c8-Abstract-Conference.html)。
