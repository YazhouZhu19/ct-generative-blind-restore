# 工业 CT 层纹生成式增强与保真盲去噪：完整方法及代码说明

> **稳定版本说明：** 生成式增强链以 v11 为保留版，固定入口为 `app/run_v11_pipeline.py`、固定参数为 `config/v11_profile.json`。v12/v13 只保留为研究对比，不覆盖 v11 结果。见 [`V11_STABLE_RELEASE_ZH.md`](V11_STABLE_RELEASE_ZH.md)。

> **v8 更新：** 当前推荐入口为 `app/run_sota_pipeline.py`。它在 v7 全链基础上，把公开初版中最强的轴向连续性去噪移到盲后验数据投影之后，并新增原图横向梯度、纵向梯度、固定亚像素端点及逐候选长度守卫。v8 方法见 [`GEOMETRY_LOCKED_DIRECTIONAL_V8_REPORT.md`](GEOMETRY_LOCKED_DIRECTIONAL_V8_REPORT.md)，v7 残差方法见 [`RESIDUAL_DENOISE_V7_REPORT.md`](RESIDUAL_DENOISE_V7_REPORT.md)。

> **v11 生成约束更新：** `app/generative_shape_constraint.py --guide ...` 先在几何保持的盲去噪图上测量层纹与夹层，再用原图同坐标证据计算置信度；同侧中心线采用联合束正则化。`app/generative_shape_project.py --guide ...` 去除生成器错误周期，并用三轮逐层FWHM闭环完成解析硬投影。详见 [`GENERATIVE_DENOISED_GUIDE_CONSTRAINT_V11_REPORT.md`](GENERATIVE_DENOISED_GUIDE_CONSTRAINT_V11_REPORT.md)。

> **v11 尺寸锁定：** `app/run_v11_pipeline.py` 在生成器输出后、任何增强后处理之前执行原图尺寸锁定。生成器原生 `1470×1070` 候选先确定性重采样为 `2200×1600`，随后所有后处理、硬投影和最终 PNG/TIFF 均在原图坐标系运行；机器可读记录位于 `generation_preprocess_size_manifest.json`、`size_lock_manifest.json` 和 `hard_shape_projection_metrics.json`。

> **v12 双证据更新：** 解析片层的横向/端点过渡进一步收紧；最终图同时与去噪引导图和原始输入逐条比较层纹、夹层的长度与宽度，并输出两个CSV。原图只作为带测量容差的非退化证据，不覆盖稳定的去噪引导坐标。详见 [`GENERATIVE_RAW_DUAL_EVIDENCE_V12_REPORT.md`](GENERATIVE_RAW_DUAL_EVIDENCE_V12_REPORT.md)。

> **v13 盲去噪细节引导更新：** 返回 v11 的几何投影主线，把同一张配准盲去噪图再次作为生成和硬投影的结构细节参考。它提供前景低/中频结构、中央实体块纹理和每条层纹沿长度方向的对比度曲线，但不能改变中心线、端点、宽度或夹层。详见 [`GENERATIVE_BLIND_GUIDE_DETAIL_V13_REPORT.md`](GENERATIVE_BLIND_GUIDE_DETAIL_V13_REPORT.md)。

> **v15 像素级结构一致性更新：** 长宽参数一致仍不足以保证局部缺口、起伏和夹层纹理一致。`app/run_v15_pipeline.py` 因此输出两张语义隔离的图：v13生成式结果只用于视觉观察；测量图全幅仅使用同坐标盲去噪结构载体，生成像素权重为0，并禁止重采样、配准、强度映射和解析层纹替换。详见 [`STRUCTURE_CARRIER_DUAL_OUTPUT_V15_REPORT.md`](STRUCTURE_CARRIER_DUAL_OUTPUT_V15_REPORT.md)。

> **v16 测量质量更新：** `app/measurement_quality_optimize.py` 以v15结构载体为不可覆盖基线，搜索受端点保护、零相位、幅度受限的残差去噪候选。本图中所有非零层纹强度均被逐层双证据审计拒绝，最终仅对中央实体块执行非局部均值去噪，中央高频噪声下降9.77%，层纹与夹层几何指标不变。详见 [`MEASUREMENT_QUALITY_V16_REPORT.md`](MEASUREMENT_QUALITY_V16_REPORT.md)。

> **v17 条件生成实验：** `app/structure_conditioned_diffusion.py` 把v16结构载体、弯曲中心线、有限宽边界、端点、夹层、置信度和不确定度作为生成网络的多通道条件，并在扩散训练中加入宽度、长度、边缘、边界场、夹层和中尺度细节损失。输出是载体中心的有界生成残差，生成后仍须通过独立逐层审计。本图选择0.85生成残差强度，层纹/中央/平坦区噪声相对v16分别下降1.05%/2.67%/1.51%。因为含生成像素，文件标记为 `MEASUREMENT_CANDIDATE`。详见 [`STRUCTURE_CONDITIONED_DIFFUSION_V17_REPORT.md`](STRUCTURE_CONDITIONED_DIFFUSION_V17_REPORT.md)。

> **v18 规整边界与细节兼容实验：** `app/constrained_detail_fusion.py` 保留v17生成式去噪，只把v11有限宽几何场用于零相位边界/端点增强与夹层残差收缩；不写入解析片层像素。每个候选完成逐条层纹和夹层审计，退化结构局部恢复为v17。本图有74条层纹保留增强、26条自动回退，夹层高频噪声再下降0.472%，层纹宽度P95误差保持0.3407%。详见 [`CONSTRAINED_DETAIL_FUSION_V18_REPORT.md`](CONSTRAINED_DETAIL_FUSION_V18_REPORT.md)。

> **v19 测量不变量分区恢复实验：** `app/structure_anchored_multiregion_denoise.py` 完整保留v17生成残差和v18规整边界，在其后把残余噪声拆为“层纹/夹层束、中央实体、端点外雾区、低结构背景”四区独立处理。层纹束只沿纵向滤波，端点包络像素逐位恢复为v18，并增加逐行FWHM、拓扑、回退比例及写盘后uint16复核。详见 [`MEASUREMENT_INVARIANT_ZONED_RESTORATION_V19_REPORT.md`](MEASUREMENT_INVARIANT_ZONED_RESTORATION_V19_REPORT.md)。

本图的完整容器搜索选中`(stack, central, fog, flat)=(0.02, 0.50, 0.60, 0.50)`：中央实体、端点外雾区和低结构背景的固定高频残差分别下降9.51%、14.26%和11.99%，9/100条临界层纹被局部恢复；层纹/夹层宽度误差P95为0.3150%/0.2096%，逐行宽度漂移P95为0.003053 px，16位写回后的全部发布守卫通过。这里的残差下降是无真值条件下的代理量，不是绝对噪声误差。

## v19 完整流程入口

v19 是 v17→v18 生成式增强链之后的受审计去噪阶段，不替换前面的生成模型、增强或后处理。它需要原始16位图、非生成的v16结构载体和已经通过约束融合的v18图：

```text
原始16位图 ───────────────┐
v16非生成结构载体 ────────┼→ 固定中心线/端点/宽度/夹层及审计基线
v18生成增强候选 ──────────┘
              ↓
四分区零相位残差去噪
  ├─ 完整层纹/夹层束：仅纵向，自适应Wiener，不跨厚度方向混合
  ├─ 中央实体：独立强度
  ├─ 端点外雾区：独立强度
  └─ 低结构背景：独立强度
              ↓
端点包络逐位锚定 → 每三行FWHM复测 → 不安全层及相邻夹层局部回退
              ↓
uint16量化候选审计 → 写入TIFF → 重新读取 → 发布审计
```

默认 CPU 容器入口：

```bash
docker compose run --rm ct-v19-measurement-invariant-zoned
```

等价的直接运行命令：

```bash
python app/structure_anchored_multiregion_denoise.py \
  --source input/source_16bit.tif \
  --carrier results_generative_shape_v16_measurement_quality/FINAL_MEASUREMENT_v16_quality_enhanced_2200x1600_16bit.tif \
  --input results_generative_shape_v18_clean_edges_detail_preserved/MEASUREMENT_CANDIDATE_v18_clean_edges_detail_preserved_16bit.tif \
  --outdir results_generative_shape_v19_structure_anchored_multiregion
```

候选选择不是以观感单指标决定。每一档参数先量化为最终会写入TIFF的uint16像素，再同时检查：层纹和夹层数量、端点和长度漂移、聚合宽度、逐行局部宽度、边缘和多尺度细节相关性、分区噪声非退化、SSIM、硬锚点逐位一致以及局部回退比例。任何不安全层连同相邻夹层都恢复为同坐标v18像素；全流程不做尺寸变化、配准、形变或解析片层重绘。

主要输出为：

- `MEASUREMENT_CANDIDATE_v19_structure_anchored_multiregion_16bit.tif`：原尺寸、uint16候选图；
- `MEASUREMENT_CANDIDATE_v19_structure_anchored_multiregion.png` 与 `MEASUREMENT_CANDIDATE_v19_comparison.png`：显示和前后对比；
- `AUDIT_v19_multiregion_masks.png`：四区掩膜及保护区审计；
- `lamella_v19_comparison.csv`、`interlayer_v19_comparison.csv`、`local_row_width_v19_comparison.csv`、`structure_detail_v19.csv`：逐结构数值证据；
- `structure_anchored_multiregion_v19_metrics.json`：候选搜索、回退、全部守卫和写盘后发布审计的机器可读事实源。

v19仍保留v17生成像素，因此“守卫通过”只表示当前自动审计没有发现超阈值结构退化，不等于计量认证。正式长度和厚度结论仍应以v16载体、导出的数值约束以及经像素尺寸和PSF/MTF标定的验证为准。

## 1. 任务目标

输入是一张 1600 × 2200、单通道 uint16 工业 CT/射线 TIFF。希望达到：

1. 降低片层主体中的颗粒噪声和竖向断续；
2. 保持片层数量、中心位置、横向厚度和中央高亮边界；
3. 输出完整尺寸的 16 位图像；
4. 同时尝试生成式模型所能达到的视觉清晰度；
5. 避免把生成式模型虚构的层纹用于测厚。

这类任务和普通照片增强的关键差别是：图像不仅要“看起来清晰”，还要维持可测量结构。因此工程被明确拆成两条相互隔离的链路。

```text
原始 16 位 TIFF
   ├─ 测量链路：自监督盲去噪 → 数据一致性 → 方向性连续化 → 几何审计
   │                                      └→ MEASUREMENT_*.tif
   └─ 视觉链路：生成式候选 → 对比度/边缘后处理
                                          └→ GENERATIVE_visual_only_*
```

v5/v6 只把生成图转换成受限的低频损失目标；生成像素永远不会写入测量结果，所有几何坐标仍来自原始16位图。未配准参考图在 v6 中也只定义外观意图，不参与像素或坐标计算。

## 2. 方法边界

本工程的测量链路借鉴了 APR-RD 的相邻像素替换思想，并结合 Noise2Self 式掩膜监督、不确定性集成和显式数据一致性。它是面向本张单幅 16 位工业图像的轻量工程适配，不是 APR-RD、Blind2Sound 或 FoundIR-v2 官方训练代码的逐行复现，也不宣称本轻量网络本身就是新的 SOTA。

采用这种适配的原因是：

- APR-RD/AP-BSN 主要在真实 sRGB 相机噪声数据上验证；现成权重的噪声域和本工业 CT 不同。
- FoundIR-v2 依赖 SDXL、LLaVA 13B 等大模型和 CUDA GPU，其自然图像生成先验可能重构或规则化片层。
- 当前只有一张待处理图，没有配对的同位置无噪声真值，也没有同设备的低剂量/高剂量训练集。
- 当前机器没有 CUDA；v8 已在 Docker Desktop CPU 容器中完成10步、五阶段端到端冒烟验证，600步正式盲结果由同一固定依赖集在本机 CPU 环境完成。

因此本次实际运行选择“单图内部自监督 + 保真约束”，并把生成式结果降级为视觉目标。

## 3. 输入和归一化

代码使用 `tifffile` 读取 TIFF，并要求输入为二维整数图像。uint16 输入按完整整数范围归一化：

\[
x = I / 65535
\]

输出时执行反变换：

\[
I_{out}=\operatorname{round}(65535\cdot\operatorname{clip}(x_{out},0,1))
\]

不使用参考 JPG 对原始强度做直方图标定；参考图与输入图不是像素配准真值。

本张图使用的区域为：

- 训练/增强区域：`y=620:1210, x=320:1880`
- 左侧测量审计区域：`y=700:1110, x=370:970`
- 右侧测量审计区域：`y=700:1110, x=1220:1830`

网络只改变训练/增强区域，区域边缘使用 18 像素羽化融合；完整 1600 × 2200 画布得以保留。

## 4. 测量链路

### 4.1 APR 启发的相邻像素替换

从单幅图像随机裁取 96 × 96 patch。每个 patch 随机选择约 12% 的像素作为监督位置。被选中的中心像素不直接输入网络，而是由八种邻域偏移之一替换：

\[
\tilde{x}_i=x_{i+\delta_i},\qquad
\delta_i\in\{(-2,0),(2,0),(0,-2),(0,2),(\pm1,\pm1)\}
\]

相邻替换的目的不是合成噪声，而是阻止网络在监督位置直接复制中心像素，并减弱空间相关噪声经中心像素泄漏到输出的风险。

### 4.2 单图掩膜自监督

原始未替换像素只作为上下文；损失仅在被隐藏的中心位置计算。网络预测的是绝对干净强度，不是带有中心捷径的残差：

\[
\mathcal{L}=\frac{1}{|M|}\sum_{i\in M}
\sqrt{\left(f_\theta(\tilde{x})_i-x_i\right)^2+\epsilon^2}
\]

其中 `M` 是本轮掩膜，`ε=10⁻³`。Charbonnier 损失比纯平方误差更不容易被极亮尖端和少量异常值支配。

### 4.3 网络结构

`AprN2SLite` 是约两万参数以内的轻量卷积网络：

```text
输入 1 通道
  → 3×3 Conv，1→12，LeakyReLU
  → 3 组 3×3 Conv，12→12，LeakyReLU
  → 3×3 Conv，12→1
  → 预测中心强度
```

主要训练参数：

| 参数 | 数值 |
|---|---:|
| 训练迭代 | 600 |
| batch size | 4 |
| patch | 96 × 96 |
| 优化器 | AdamW |
| 初始学习率 | 8×10⁻⁴ |
| 最低学习率 | 3×10⁻⁵ |
| 学习率策略 | CosineAnnealing |
| 梯度范数上限 | 1.0 |
| 随机种子 | 23 |

### 4.4 掩膜集成和不确定性

推理时仍然不能把待预测中心直接暴露给网络。代码执行 8 次随机掩膜推理，每次隐藏约 55% 像素。对每个像素，仅统计它被隐藏时得到的预测：

\[
\mu_i=\frac{1}{N_i}\sum_{k=1}^{N_i}\hat{x}^{(k)}_i
\]

\[
u_i=\sqrt{\frac{1}{N_i}\sum_{k=1}^{N_i}(\hat{x}^{(k)}_i)^2-\mu_i^2}
\]

`u_i` 被保存为 `MEASUREMENT_uncertainty_float32.tif`。它不是经过统计标定的置信区间，但可以指出不同掩膜上下文给出不一致预测的位置。

### 4.5 噪声尺度估计

先用高通残差估计归一化噪声尺度：

\[
h=x-G_{1.2}*x
\]

\[
\hat{\sigma}_n=\frac{\operatorname{median}(|h-\operatorname{median}(h)|)}{0.6745}
\]

本图估计得到 `σₙ=0.009324`。

### 4.6 边缘门控与数据一致性

网络输出不会直接覆盖观测。首先计算平滑观测的二维梯度，并构造边缘保护门：

\[
g=\exp\left[-\left(\frac{|\nabla(G_{0.7}*x)|}{T}\right)^2\right]
\]

`T` 是增强区域梯度的第 78 百分位。平坦区域 `g` 较大，可以较充分去噪；片层横向边缘处 `g` 接近零，尽量保持厚度边界。

预测改变量受到噪声尺度约束：

\[
c=2.75\hat{\sigma}_n
\]

\[
z=x+\alpha g\,\operatorname{clip}(\mu-x,-c,c)
\]

本次 `α=0.20`，改变量上限 `c=0.02564`。这个步骤防止神经网络凭先验进行大幅改写。

### 4.7 沿片层方向的连续性处理

片层主要沿竖直方向延伸。为了降低层内断续，同时避免横向混合相邻片层，使用各向异性核 `σ=(3,0)`：

\[
v=G_{(3,0)}*z
\]

再使用横向 Sobel 梯度构造保护门 `gₓ`：

\[
z'=z+\beta g_x(v-z),\qquad \beta=0.70
\]

最终总改变量再次裁剪到 `[-c,c]`。因为横向高梯度处权重很低，该步骤主要平滑同一片层内部的竖向颗粒，不直接跨越左右厚度边界。v4 保留这一步及其原参数不变。

### 4.8 完整图回填

增强只在目标区域执行，随后用从边缘到内部逐渐增大的羽化权重回填完整图：

\[
y=(1-w)x+wz'
\]

这样可以避免 ROI 边缘出现矩形接缝，并保留区域外原始数据。

### 4.9 原增强链后的亚像素几何校正

v4 先完整执行上述盲去噪、方向连续化、质量参数搜索、边缘增强、结构增强、限幅和羽化。原始图只用于测量每条层纹的目标端点坐标，不向结果回写任何原始像素。

对可靠端点计算增强坐标与原始坐标之差 `Δy`，在已增强图上建立半径为1 px的局部位移场。常规单次校正最多0.35 px；超过3 px的竞争峰关联异常允许一次不超过1 px的局部重关联：

\[
I_{final}(x,y)=I_{enhanced}(x,y+d_y(x,y))
\]

图像通过三次插值重采样，移动的是已经去噪和增强的像素。原始噪声不会重新进入结果。候选随后接受逐层端点、长度和横向 FWHM 守卫；如果不通过则拒绝输出。

### 4.10 几何校正后的结构感知弱去雾

为抑制层纹端点外侧残留的细颗粒和雾状区域，v4 在几何校正之后增加弱净化，但不替换或减弱前面的任何算法。先从几何结果计算平滑梯度，并以第45百分位构造低结构门：

\[
g=\exp[-(|\nabla G_{0.70}(I)|/T_{45})^2](1-P_{endpoint})
\]

`P_endpoint` 只由原图测得的端点坐标生成，保护核心半径为纵向5 px、横向3 px；它不包含原图灰度。净化只从增强图计算两个尺度：

\[
I'=I+g[\alpha(G_{0.90}(I)-I)+\beta(G_{2.20}(I)-G_{0.90}(I))]
\]

程序搜索 `(α,β)=(0,0)、(0.08,0.010)、(0.10,0.012)、(0.12,0.015)、(0.14,0.018)`。改变量限制在估计噪声MAD的0.65倍内并在目标ROI边缘羽化。最终只接受同时满足边缘锐度保留≥99%、局部对比度保留≥99.5%、相对几何结果SSIM≥0.9999以及全部宽度/端点/长度守卫的候选。本图自动选择 `(0.12,0.015)`。

### 4.11 v6 几何锁定边界清洁

v6 不替换4.1–4.10的任何步骤，而是在完整 v5 输出之后增加 `app/boundary_cleanup.py`。程序用原图测得的100条层纹上下端点拟合左右两组端点包络，构造内部、边界和外部软掩膜。内部使用 `σ=(2.00,0.18)` 的轴向细平滑，避免跨横向厚度边缘；外部使用 `σ=1.15` 的细平滑并削减 `G₂.₂-G₇` 的正向雾状残差。

端点核心邻域强制保护，所有改变量限制为噪声MAD的0.8倍。八档候选只有同时满足横向FWHM、端点、长度、边缘锐度、局部对比度、边界梯度和SSIM阈值才允许输出。本图选中的强度为 `fine=0.65, halo=1.8`，未再叠加额外锐化；相对 v5 输出，外部光晕降低10.99%，内部轴向噪声降低10.12%，同时横向边缘锐度保持100.01%。

### 4.12 v7 分区残差去噪与固定路径审计

`app/residual_denoise.py` 把 v6 后仍存在的噪声分为两类。左右层纹内部使用 `σ=(2.50,0.18)` 的轴向残差收缩，并用横向梯度门避免跨厚度边缘；中央高亮内部单独使用 `patch=5、distance=5、h=0.85σ` 的非局部均值，18像素软边界避免影响中央块轮廓。两类操作共同受原始端点核心保护和 `0.65×MAD` 单像素限幅。

正式图像自动选择 `lamella=0.60、central=0.70`。相对 v6，层纹轴向残噪再降低6.97%，中央残噪再降低8.41%；相对原图，最终平坦区高频噪声降低38.38%，边缘锐度仍提高5.39%。`length_optimize.py --geometry-guide` 固定使用处理前的中心和弯曲路径，避免后处理结果重新检测路径而制造虚假的前后长度差异。

### 4.13 v8 强方向去噪的正确插入位置

公开初版 `pipeline.py::directional_continuity` 的 `σ=(3,0)` 轴向平滑对断续颗粒最有效，但原实现只有横向梯度门，因此能保护厚度边缘，却没有显式保护上下端点。v8 不在末尾修补这一问题，而是在盲模型的 Poisson-Gaussian 数据投影之后、原有锐化和对比度增强之前调用 `geometry_locked_directional_continuity`。此时强去噪尚未破坏测量证据，程序可以在候选进入后续链路之前拒绝它。

新门控同时使用原图横向梯度、原图纵向梯度和固定端点保护区。程序搜索方向强度 `0、0.25、0.40、0.55、0.70`，并用同一组原图中心、弯曲路径和亚像素端点检查横向 FWHM、端点、长度、锐度、对比度和 SSIM。本图选择0.55；0.70因SSIM降至0.99418而被拒绝。选择后盲阶段粗糙度相对原图降低31.94%，端点偏移P95为0.034 px、长度变化P95为0.057 px。

因此形状一致性不是单个末端模块，而是“处理前固定参考—训练损失—数据投影—强去噪入口—逐候选拒绝—最终固定路径审计”的闭环。

## 5. 生成式视觉链路

生成式模型同时读取：

- 图 1：原始 16 位 TIFF，作为编辑目标；
- 图 2：处理参考 JPG，只作为清晰度、材质和对比度参考。

提示词要求保持完整画面、层数、位置、厚度、中心高亮和几何关系，并禁止增加、删除、复制或拉直片层。尽管有这些约束，实际生成结果仍把层纹规则化，并重构了局部形态。这说明文本约束不能替代物理数据一致性。

生成结果只进行确定性后处理：

1. 0.4%～99.6% 强度窗归一化；
2. CLAHE 局部对比度增强，权重 0.28；
3. `σ=0.65` 的轻微平滑；
4. 梯度门控的弱反锐化，强度 0.16；
5. 保存为视觉专用 PNG 和带警告描述的 16 位 TIFF。

文件名和 TIFF description 都包含 `VISUAL_ONLY`。该链路适合制作展示图或定义目标观感，不适合测厚、缺陷验收和训练真值生成。

### 5.1 v13 同坐标盲去噪细节引导

v11 只把盲去噪图转成中心线、端点、FWHM 和夹层走廊，再把这些几何量送给生成器。硬投影时，每条片层使用单一生成亮度，容易把真实的纵向衰减和中央块内部结构规则化。v13 保持 v11 的几何算法不变，同时增加第二条细节通道：

1. 将盲去噪引导图的 `1%–99.5%` 稳健强度窗匹配到生成候选，但不执行配准、位移或形变；
2. 在覆盖左右层纹束及中央实体块的羽化前景区域，按 `0.55` 权重融合盲去噪图的低/中频结构；
3. 对每条层纹沿固定中心线采样“中心—左右夹层”对比度，平滑后形成有界轴向调制曲线；
4. 端点前后 `4–18 px` 内把细节调制渐退到1，防止亮度变化移动半高端点；
5. 使用 v11 的双 logistic 有限宽片层和三轮 FWHM 闭环，重新锁定所有宽度、中心、端点与夹层。

因此，细节通道只能改变已测片层内部的亮度分布和前景低频结构，不能修改几何。`structure_detail_consistency.csv` 对100条层纹分别记录盲去噪引导与软生成/硬投影的轴向相关性；JSON 还记录低频、中频和梯度幅值相关性。几何验收继续使用逐层、逐夹层的长宽CSV。

## 6. 几何与保真审计

### 6.1 片层数与位置

在左右测量区域分别对竖向平均轮廓做高斯平滑和峰值检测。处理前后的峰进行最近邻匹配，只允许不超过 5 px 的候选匹配，并将 1 px 设为最终守卫上限。

### 6.2 FWHM

对每个片层峰以半高宽计算横向宽度。守卫要求中位 FWHM 相对盲去噪基线变化不超过 5%，且相对原始图变化不超过 4%。这项指标比单纯的视觉锐度更直接对应“片状层纹是否被变成尖线”。

### 6.3 逐层端点与长度尾部守卫

新版沿每条弯曲中心路径对原始图和最终几何校正/弱净化图重新检测亚像素上下端点。自动候选必须满足：可靠层端点偏移绝对 95% 分位不超过 0.35 px、最大值不超过 0.75 px；长度差绝对 95% 分位不超过 0.50 px、最大值不超过 1.00 px。原始端点内部不确定度大于 3 px的层单独标为 `review`，保留在审计文件中但不伪装成可靠真值。

### 6.4 连续性和断裂率

对每个片层附近逐行取局部最大值，以较大尺度高斯趋势作为该层的低频亮度趋势：

\[
CV=\frac{\operatorname{std}(l-G_{10}*l)}{\operatorname{median}(l)}
\]

逐行亮度低于趋势中位数 70% 的比例记为断裂率。

### 6.4 中央高亮完整性

在左右片层与中央区域交界处分别建立窄带，计算高亮边界的连续性 CV 和断裂率。守卫不允许处理后的断裂率比原图增加超过 0.005。

### 6.5 SSIM 和残差

测量联合区域计算处理前后 SSIM。同时输出：

- `residual = output - input` 的 float32 TIFF；
- 对称显示窗的残差预览；
- 多掩膜预测不确定性 TIFF 和预览。

SSIM 只作为一个整体变化上限，不能单独证明测量真实性。最终守卫必须同时满足：

| 守卫 | 阈值 |
|---|---:|
| 左右片层数量变化 | ≤ 1 |
| 最大片层中心位移 | ≤ 1 px |
| FWHM 相对变化 | ≤ 5% |
| 测量 ROI SSIM | ≥ 0.970 |
| 中央高亮断裂率增加 | ≤ 0.005 |

## 7. 本次运行结果

| 指标 | 左侧 | 右侧 |
|---|---:|---:|
| 片层数，前 → 后 | 25 → 25 | 30 → 30 |
| 最大中心位移 | 0 px | 0 px |
| 中位 FWHM，前 → 后 | 4.6203 → 4.6226 px | 4.9495 → 4.9673 px |
| FWHM 相对变化 | 0.049% | 0.358% |
| 连续性 CV，前 → 后 | 0.11295 → 0.07839 | 0.10950 → 0.07915 |
| 断裂率，前 → 后 | 0.03659 → 0.01707 | 0.03049 → 0.00732 |

其他结果：

- 测量 ROI SSIM：0.973383
- 残差均值：-0.0000648
- 残差 RMS：0.007882
- 99% 绝对残差：0.02418
- 中央左高亮边界断裂率：0.00488 → 0
- 中央右高亮边界断裂率：0 → 0
- 配置的自动守卫结果：通过

“通过”表示这套自动检查没有发现明显的片层几何破坏，不代表已经完成绝对厚度计量认证。绝对厚度仍需像素尺寸、标准件和系统 PSF/MTF 标定。

## 8. 代码结构

```text
ct_generative_blind_restore/
├── app/
│   ├── pipeline.py                  # 盲去噪、生成式后处理和基础 QA
│   └── length_optimize.py           # 边缘优化、中心线跟踪和长度测量
├── config/
│   └── method_registry.yaml         # SOTA 候选及适用边界
├── input/
│   ├── source_16bit.tif
│   ├── reference_unpaired.jpg
│   └── generative_candidate_visual_only.png
├── results/
│   ├── MEASUREMENT_blind_denoised_16bit.tif
│   ├── MEASUREMENT_residual_float32.tif
│   ├── MEASUREMENT_uncertainty_float32.tif
│   ├── measurement_model.pt
│   ├── run_manifest.json
│   └── GENERATIVE_visual_only_postprocessed.*
├── results_length/
│   ├── MEASUREMENT_length_optimized_16bit.tif
│   ├── layer_lengths.csv
│   ├── layer_length_overlay_roi.png
│   └── length_qa.json
├── Dockerfile.cpu
├── Dockerfile.cuda
├── compose.yaml
├── requirements-common.txt
├── requirements-cpu.txt
├── README.md
└── EXPERIMENT_REPORT.md
```

所有算法实现集中在 `app/pipeline.py` 和 `app/length_optimize.py`，没有隐藏的训练脚本或私有模块。

## 9. 不使用 Docker 直接运行

```bash
python app/pipeline.py \
  --input input/source_16bit.tif \
  --generated input/generative_candidate_visual_only.png \
  --outdir results \
  --iterations 600 \
  --passes 8 \
  --batch 4 \
  --device cpu
```

可调参数：

```text
--train-roi y0,y1,x0,x1
--left-roi y0,y1,x0,x1
--right-roi y0,y1,x0,x1
--iterations 600
--patch 96
--batch 4
--passes 8
--strength 0.20
--directional-strength 0.70
--directional-sigma 3.0
--seed 23
--device auto|cpu|cuda
```

## 10. Docker CPU

```bash
docker compose run --rm ct-restore-cpu
```

等价的完整命令：

```bash
docker build -f Dockerfile.cpu -t ct-generative-blind-restore:cpu .

docker run --rm \
  -v "$PWD/input:/data/input:ro" \
  -v "$PWD/results:/data/results" \
  ct-generative-blind-restore:cpu \
  --input /data/input/source_16bit.tif \
  --generated /data/input/generative_candidate_visual_only.png \
  --outdir /data/results \
  --device cpu
```

## 11. Docker CUDA

需要 Linux、NVIDIA 驱动、Docker 和 NVIDIA Container Toolkit：

```bash
docker compose --profile cuda run --rm ct-restore-cuda
```

CUDA 镜像基于：

```text
pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
```

Apple Silicon Mac 可以运行 CPU 镜像，不能运行 NVIDIA CUDA 镜像。

## 12. 如何迁移到更多图像

1. 每张图首先保持原始 uint16 TIFF，不要预先转成 8 位 JPEG。
2. 根据工件位置调整三个 ROI。
3. 先运行较弱参数，例如 `--strength 0.15`。
4. 检查 `run_manifest.json` 是否通过守卫。
5. 逐张查看残差图，确认残差主要是颗粒噪声而非连续片层。
6. 查看不确定性高值是否集中在尖端、边界、缺陷或高亮饱和区域。
7. 只有在多张同设备图像上重复验证后，才固定参数做批处理。
8. 如果能取得配对低噪声真值，应改为监督或 Noise2Noise 域内训练，并重新建立测量偏差评估。

## 13. 进一步使用官方模型时的建议

- **Blind2Sound/APR-RD 类模型**：优先使用同设备的多张 16 位 CT 图像重新训练，保留原始灰度，不直接使用自然相机预训练权重做最终测量。
- **FoundIR-v2/扩散模型**：只放入视觉链路；若要研究其测量能力，必须单独进行合成缺陷、层数、厚度偏差和重复性盲测。
- **域内扩散去噪**：需要匹配的噪声物理，例如不同曝光剂量、重复采集或已校准的 Poisson-Gaussian 噪声模型。

## 14. 参考方法

- APR-RD, AAAI 2025：<https://ojs.aaai.org/index.php/AAAI/article/view/32447>
- Blind2Sound, ICCV 2025 官方代码：<https://github.com/Jiazheng-Liu/Blind2Sound>
- AP-BSN, CVPR 2022 官方代码：<https://github.com/wooseoklee4/AP-BSN>
- FoundIR-v2, CVPR 2026 官方代码：<https://github.com/cschenxiang/FoundIR-v2>
- Diffusion X-ray image denoising, MIDL 2024：<https://proceedings.mlr.press/v250/sanderson24a.html>

## 15. 面向层纹长度的第二阶段优化

长度测量与厚度测量的边缘方向不同：厚度主要依赖左右边界，长度主要依赖上下端点。`app/length_optimize.py` 负责独立测量和复核：

1. 原始图只提供几何坐标和梯度门，不向增强图回写原始像素；
2. 可选的对称高斯反锐化只作用于已增强像素，并相对增强图限幅；
3. 在主体区域检测每条可见亮层纹；
4. 从主体向上、向下逐行跟踪中心路径，限制每行最多移动 2 px，并禁止跳到相邻层；
5. 沿弯曲中心路径提取强度剖面，以梯度极值和三点二次曲线拟合上下亚像素端点；
6. 用三条相邻路径估计端点重复性和长度不确定度；
7. 推荐长度始终来自原始 16 位数据的路径拟合；增强图只帮助观察和初始化，不覆盖推荐测量值；
8. 输出增强图重新检测的端点偏移、同坐标梯度增益和逐层人工复核标志。

v4 的推荐质量输出已经在 `app/quality_optimize.py` 中完成几何校正和结构感知弱净化；长度脚本可用 `--sharpen-amount 0` 对该结果做不改变像素的独立审计。任何绝对物理长度仍需像素尺寸和系统 PSF/MTF 标定。
