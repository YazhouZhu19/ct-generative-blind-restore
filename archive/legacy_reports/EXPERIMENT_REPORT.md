# 生成式 + 盲去噪增强实验报告

## 结论

可以建立“先生成、再后处理”的链路，但生成结果不能作为厚度测量输入。本实验因此交付两条物理隔离的输出：

- **视觉链路**：内置生成式图像工具依据原始 TIFF 与非配对参考图生成清晰候选，再做局部对比度和边缘门控后处理。输出统一命名为 `GENERATIVE_visual_only_*`。
- **测量候选链路**：直接从原始 16 位 TIFF 训练单图自监督盲去噪网络，使用 APR-RD 启发的相邻像素替换、Noise2Self 掩膜预测、8 次掩膜不确定性估计、数据一致性限幅和沿片层方向的连续性处理。生成式像素不进入本链路。

当前机器没有 Docker、Podman 或 OrbStack，且 Apple Silicon 8 GB 环境没有 CUDA。因此 Docker CPU/CUDA 工程已经创建，但无法在本机执行镜像构建；核心 Python 代码已在现有 PyTorch CPU 环境完成等价运行。

## 最终测量候选结果

输入：1600 × 2200，单通道 uint16 TIFF。

| 指标 | 左侧 | 右侧 |
|---|---:|---:|
| 检出片层数（前 → 后） | 25 → 25 | 30 → 30 |
| 最大中心位移 | 0 px | 0 px |
| FWHM（前 → 后） | 4.6203 → 4.6226 px | 4.9495 → 4.9673 px |
| FWHM 相对变化 | 0.049% | 0.358% |
| 层纹连续性 CV（前 → 后） | 0.11295 → 0.07839 | 0.10950 → 0.07915 |
| 断裂率（前 → 后） | 0.03659 → 0.01707 | 0.03049 → 0.00732 |

测量 ROI SSIM 为 0.97338。残差均值为 -0.0000648（归一化强度），残差 RMS 为 0.007882，99% 绝对残差为 0.02418；估计噪声标准差为 0.009324，单像素改变量被限制在 0.02564 以内。

中心高亮边界检查：

- 左边界连续性 CV：0.09002 → 0.06582，断裂率：0.00488 → 0。
- 右边界连续性 CV：0.07524 → 0.05748，断裂率保持 0。

本次配置通过以下守卫：左右层数变化不超过 1、峰中心移动不超过 1 px、FWHM 相对变化不超过 5%、测量 ROI SSIM 不低于 0.97、中心高亮断裂率不允许显著增加。通过守卫仅说明本次自动检查未发现明显几何破坏，不替代标准件、像素尺寸及系统 PSF 标定。

## 当前方法选择

- APR-RD（AAAI 2025）针对真实空间相关噪声提出 Adjacent Pixel Replacer 与 Recharged Distillation。本地可运行实现只借用了“相邻替换去相关”的思想，并非完整论文复现。
- Blind2Sound（ICCV 2025）是值得在有 CUDA 和 CT 训练集后加入的第二个自监督候选；官方仓库提供灰度训练脚本，但仍需做工业 CT 域验证。
- FoundIR-v2（CVPR 2026）是当前通用生成式图像恢复候选之一，但官方推理依赖 SDXL、LLaVA 13B 等模型并要求一到两张 CUDA GPU。它可进入视觉链路，不能零样本用于测厚。
- MIDL 2024 的扩散 X 射线去噪说明扩散模型在匹配 X 射线物理与噪声分布训练时有潜力；它不能证明自然图像生成模型对本工业 CT 单图是保真的。

主要来源：

- <https://ojs.aaai.org/index.php/AAAI/article/view/32447>
- <https://github.com/Jiazheng-Liu/Blind2Sound>
- <https://github.com/cschenxiang/FoundIR-v2>
- <https://proceedings.mlr.press/v250/sanderson24a.html>
- <https://github.com/wooseoklee4/AP-BSN>

## 生成式视觉候选

生成方式：Codex 内置图像生成工具；并非容器内的 FoundIR-v2。输入图 1 是原始 TIFF，输入图 2 是非配对参考 JPG。

最终提示词：

> Use case: precise-object-edit. Asset type: visual-only industrial CT enhancement candidate, explicitly not for metrology. Enhance Image 1, the full grayscale industrial CT/radiographic image, so the laminated plate body is visually clear in the style and clarity of Image 2; Image 2 is only a quality/material/contrast reference. Preserve the exact full-image framing, specimen position, layer count, spacing, plate thickness, boundaries, curvature, central gap/bridge width, and every large structural feature of Image 1. Do not crop or zoom. Use a realistic monochrome industrial X-ray/CT appearance. Make internal plate-like layers continuous and visibly thick rather than needle-like; keep the central white/highlighted material complete and continuous; improve local contrast and suppress grain. Do not invent, remove, duplicate, shift or straighten lamellae; no text, false color, labels, scale bars or watermark. Avoid product-photo appearance, fake edges, sharpened spikes, repeated texture cloning, hallucinated plates, overexposure and clipped highlights.

尽管提示词要求锁定几何，生成结果仍出现了明显的规则化和重构。这正是它只能作为视觉目标、不能用于测量的直接证据。

## 交付文件

- `results/MEASUREMENT_blind_denoised_16bit.tif`：后续测量验证候选。
- `results/MEASUREMENT_residual_float32.tif`：结构误删检查。
- `results/MEASUREMENT_uncertainty_float32.tif`：高风险区域检查。
- `results/run_manifest.json`：完整参数与指标。
- `results/GENERATIVE_visual_only_postprocessed.png`：生成式视觉结果。
- `Dockerfile.cpu`、`Dockerfile.cuda`、`compose.yaml`：容器配置。
- `app/pipeline.py`：与本次实际运行完全相同的管线。
