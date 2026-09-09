# CT 生成式盲去噪与结构引导增强

本仓库已经收敛为单一、可复现的工业 CT 图像增强流程，固定了本次获得认可结果的真实处理顺序：

```text
外部生成候选图 → 立即校正到原图尺寸
原始 16 位 CT → 盲去噪测量引导图 → 测量层纹/端点/宽度/夹层
两路数据 → 原尺寸画布上的有界连续二维结构引导 → PNG + 16 位 TIFF + 审计报告
```

主要保证：

- 输出尺寸与原始图严格一致，不裁剪、不补边。
- 尺寸校正在生成后立即发生，早于结构引导及其他处理。
- 中间块与左右层纹掩膜外像素完全锁定。
- 最终显示灰度只来自生成候选图，不回填原图或引导图像素。
- 只允许小幅连续二维形变，并通过 SSIM、写入区域、层纹中心误差等门控。
- 本次认可样例自动选择强度 `0.16`，实际水平和垂直最大位移约为 `0.8 px`。

> 增强结果属于测量辅助图，不是标定真值。每次处理都应同时保存原始图、盲去噪引导图、坐标配置和运行清单。

## Docker 快速运行

把文件放入 `input/`：

```text
source_16bit.tif
generative_candidate.png
measurement_guide_16bit.tif   # 可选；已有时运行更快
```

使用已有引导图：

```bash
docker compose build enhance
docker compose run --rm enhance
```

自动针对当前图像训练盲去噪引导模型：

```bash
docker compose --profile train-guide run --rm enhance-auto-guide
```

也可以直接运行：

```bash
python -m pip install -r requirements-cpu.txt
python app/run_pipeline.py \
  --source input/source_16bit.tif \
  --generated input/generative_candidate.png \
  --guide input/measurement_guide_16bit.tif \
  --outdir output
```

不传 `--guide` 时，会以当前原始图像进行自监督盲去噪训练。生成模型是明确的外部接口，仓库不会保存远程账号、密钥或专有权重；接口规范见
[docs/GENERATOR_INTERFACE.md](docs/GENERATOR_INTERFACE.md)。

## 处理同类图像

默认配置针对本次相同构图，并可从 `2200 × 1600` 参考尺寸自动缩放坐标。若拍摄位置发生变化，请复制并修改
[`config/reference_geometry.json`](config/reference_geometry.json)，重点核对：

- 左右层纹主体 ROI；
- 上端点与下端点搜索区间；
- 中间完整高亮块锁定区域；
- 最大形变量与候选强度。

完整原理、约束和质量门控见 [docs/METHOD.md](docs/METHOD.md)，容器验证方法见
[docs/VALIDATION.md](docs/VALIDATION.md)。
