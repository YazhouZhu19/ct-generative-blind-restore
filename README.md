# 16 位 CT：自监督盲去噪 + 保真边缘增强容器

完整原理、公式、参数、QA 和代码结构见 `METHOD_AND_CODE_GUIDE.md`。

> 数据安全：本仓库只保存代码、容器配置和技术文档。原始 TIFF、参考图、生成图、模型权重及运行结果由 `.gitignore` 排除，请通过只读挂载或本地目录提供。

工程仍保留两类输出能力，但本轮默认只运行保真盲去噪链路：

1. `GENERATIVE_*`：生成式视觉候选及其后处理结果，只用于观察清晰度方向，禁止测厚、缺陷判定或当作真值。
2. `MEASUREMENT_*`：单图自监督盲去噪结果。采用 APR-RD 思路中的相邻像素替换、Noise2Self 掩膜预测、多次掩膜不确定性，以及边缘/数据一致性门控；输出残差和不确定性图。

当前实现是为本张 16 位灰度图做的工程化适配，不声称复现完整 APR-RD、Blind2Sound 或 FoundIR-v2。参考 JPG 不参与像素级训练。

## 当前质量优化流程

本轮不执行层纹长度测量。先运行原有单图自监督盲去噪模型，再执行受限的方向性边缘与局部对比增强：

```bash
docker compose run --rm ct-restore-cpu
docker compose run --rm ct-quality
```

`ct-quality` 会自动搜索边缘增益和结构增益参数，并以层数稳定、层中心位移、FWHM、平坦区高频起伏及 SSIM 为守卫条件。首选结果是 `results_quality/QUALITY_balanced_16bit.tif`，它保持 2200 x 1600、16 位灰度且不含生成式像素。

## 输出说明

- `MEASUREMENT_blind_denoised_16bit.tif`：唯一可进入后续标定/测量验证的候选。
- `MEASUREMENT_residual_float32.tif`：处理结果减原始观测，检查是否误删结构。
- `MEASUREMENT_uncertainty_float32.tif`：多掩膜预测标准差，高值区域应人工复核。
- `run_manifest.json`：方法、参数、层纹数量/FWHM/位移/SSIM 守卫指标。
- `GENERATIVE_visual_only_postprocessed*.{png,tif}`：生成式视觉结果，严禁测量。
- `comparison.png`：原图、保真盲去噪、生成式视觉候选对比。
- `results_quality/QUALITY_balanced_16bit.tif`：本轮面向清晰边缘、整体质量和去噪的首选完整图。
- `results_quality/QUALITY_balanced_preview.png`：首选完整图的 8 位预览。
- `results_quality/QUALITY_display_only.png`：仅用于观看的局部对比映射，不保留定量灰度。
- `results_quality/QUALITY_comparison.png`：原图、上一版盲去噪、本轮优化及显示版的层纹区域放大对比。
- `results_quality/quality_metrics.json`：参数搜索、质量增益和结构守卫指标。

## Docker CPU（Mac/无 NVIDIA GPU）

```bash
docker compose run --rm ct-restore-cpu
```

或：

```bash
docker build -f Dockerfile.cpu -t ct-generative-blind-restore:cpu .
docker run --rm \
  -v "$PWD/input:/data/input:ro" \
  -v "$PWD/results:/data/results" \
  ct-generative-blind-restore:cpu \
  --input /data/input/source_16bit.tif \
  --outdir /data/results --device cpu
```

## Docker CUDA（NVIDIA 主机）

需要 NVIDIA Container Toolkit：

```bash
docker compose --profile cuda run --rm ct-restore-cuda
```

默认 600 次单图训练；快速验证可在命令末尾加 `--iterations 120 --passes 4`。生产复核建议至少 600 次并查看 `guardrail_pass`、残差图和不确定性图。

## 已保留但本轮不启用：长度优化与逐层测量

只有后续重新需要长度研究时，才在盲去噪后运行：

```bash
docker compose run --rm ct-length
```

长度阶段会保护上下端点、执行受限的零相位对称锐化、逐行跟踪每条弯曲层纹的中心路径，并沿路径拟合亚像素端点。输出位于 `results_length/`：

- `MEASUREMENT_length_optimized_16bit.tif`：长度测量视觉候选；
- `layer_lengths.csv`：每条层纹的推荐像素长度、端点、内部不确定度及质量标志；
- `layer_length_overlay_roi.png`：绿色为通过，红色为人工复核；
- `length_qa.json`：整体准确性与数据质量检查。

如果已经获得经标准件标定的像素尺寸，例如每像素 `0.012 mm`，可直接运行：

```bash
docker compose run --rm --entrypoint python ct-restore-cpu \
  app/length_optimize.py \
  --source /data/input/source_16bit.tif \
  --denoised /data/results/MEASUREMENT_blind_denoised_16bit.tif \
  --outdir /data/results_length \
  --pixel-size 0.012 --unit mm
```

不得从 TIFF 的显示尺寸推断像素物理尺寸；当前文件没有 XResolution、YResolution 或 ResolutionUnit 标记。

## 为什么没有直接把 FoundIR-v2 当作测量图

FoundIR-v2 使用 SDXL、LLaVA 等大模型依赖，官方推理脚本面向一到两张 CUDA GPU。它是通用图像恢复模型，并非针对当前工业 CT、16 位强度或本设备 PSF 标定训练。它可以产生非常清晰的层纹，但清晰不等于真实；新增、删除或移动一层都会使厚度结论失效。

因此本工程允许把任意生成式结果放入 `--generated` 做视觉后处理，却永远不会把它混入 `MEASUREMENT_*` 链路。

## 本机验证边界

交付机器为 Apple Silicon 8 GB，且未安装 Docker/Podman/OrbStack。因此镜像文件已经生成，但本机不能构建镜像；核心脚本使用同一依赖的本机 PyTorch CPU 环境执行验证。换到有 Docker 的机器即可复跑。
