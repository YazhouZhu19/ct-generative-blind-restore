# 工业 CT 层纹生成式增强与保真盲去噪：完整方法及代码说明

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

生成式输出永远不会被读回测量链路。

## 2. 方法边界

本工程的测量链路借鉴了 APR-RD 的相邻像素替换思想，并结合 Noise2Self 式掩膜监督、不确定性集成和显式数据一致性。它是面向本张单幅 16 位工业图像的轻量工程适配，不是 APR-RD、Blind2Sound 或 FoundIR-v2 官方训练代码的逐行复现，也不宣称本轻量网络本身就是新的 SOTA。

采用这种适配的原因是：

- APR-RD/AP-BSN 主要在真实 sRGB 相机噪声数据上验证；现成权重的噪声域和本工业 CT 不同。
- FoundIR-v2 依赖 SDXL、LLaVA 13B 等大模型和 CUDA GPU，其自然图像生成先验可能重构或规则化片层。
- 当前只有一张待处理图，没有配对的同位置无噪声真值，也没有同设备的低剂量/高剂量训练集。
- 当前机器没有 CUDA，并且没有 Docker/Podman 容器运行时。

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

最终总改变量再次裁剪到 `[-c,c]`。因为横向高梯度处权重很低，该步骤主要平滑同一片层内部的竖向颗粒，不直接跨越左右厚度边界。

### 4.8 完整图回填

增强只在目标区域执行，随后用从边缘到内部逐渐增大的羽化权重回填完整图：

\[
y=(1-w)x+wz'
\]

这样可以避免 ROI 边缘出现矩形接缝，并保留区域外原始数据。

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

## 6. 几何与保真审计

### 6.1 片层数与位置

在左右测量区域分别对竖向平均轮廓做高斯平滑和峰值检测。处理前后的峰进行最近邻匹配，只允许不超过 5 px 的候选匹配，并将 1 px 设为最终守卫上限。

### 6.2 FWHM

对每个片层峰以半高宽计算横向宽度。守卫要求中位 FWHM 相对变化不超过 5%。这项指标比单纯的视觉锐度更直接对应“片状层纹是否被变成尖线”。

### 6.3 连续性和断裂率

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

长度测量与厚度测量的边缘方向不同：厚度主要依赖左右边界，长度主要依赖上下端点。原连续性处理中沿竖直方向的平滑可能改善观感，却会降低端点定位的可信度，因此新增 `app/length_optimize.py`：

1. 在上下端点搜索带恢复原始 16 位数据权重，生成式结果完全不参与；
2. 使用对称高斯反锐化做零相位、限幅的端点增强；
3. 在主体区域检测每条可见亮层纹；
4. 从主体向上、向下逐行跟踪中心路径，限制每行最多移动 2 px，并禁止跳到相邻层；
5. 沿弯曲中心路径提取强度剖面，以梯度极值和三点二次曲线拟合上下亚像素端点；
6. 用三条相邻路径估计端点重复性和长度不确定度；
7. 推荐长度始终来自原始 16 位数据的路径拟合；增强图只帮助观察和初始化，不覆盖推荐测量值；
8. 输出增强图重新检测的端点偏移、同坐标梯度增益和逐层人工复核标志。

本次识别 100 条可见层纹，99 条通过，L20 的内部长度不确定度为 3.411 px，需要人工复核。中位长度不确定度为 0.369 px，95% 分位为 0.991 px；增强图的同坐标端点梯度中位提升 2.19%，增强 ROI 相对原图 SSIM 为 0.99639。
