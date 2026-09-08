# v11 稳定保留版

本文件将“不使用配准”的 v11 固化为项目当前选定的稳定生成式增强方法。v12、v13 以及已放弃的 v14 配准实验仅保留用于研究对比，不替换 v11 的代码入口、参数和既有结果。

## 固化的处理链

1. 对原始16位图执行不使用配准、不改变坐标的盲去噪；
2. 在原图坐标系下的盲去噪图上测量左侧49条、右侧51条层纹的中心路径、亚像素端点、有限宽FWHM，以及98个逐行夹层；
3. 使用原图同坐标证据验证测量可信度，但不把噪声较大的原图测量平均回引导坐标；
4. 以原图为唯一内容目标、v11约束图为几何条件、未配准参考图为外观条件，生成视觉候选；
5. 生成器原生输出后立即用确定性 Lanczos 重采样锁定到原图宽高；
6. 在原图尺寸坐标网格上执行全部确定性对比度、平滑和梯度门控后处理；本次尺寸均为 `2200×1600`；
7. 去除生成器错误周期，用 v11 双 logistic 有限宽片层逐条重建；
8. 执行三轮逐层FWHM闭环校准，并输出尺寸与几何审计。

固定参数为：横向过渡 `0.38 px`、端点过渡 `0.55 px`、盲去噪图直接纹理融合权重 `0`。因此 v11 中盲去噪图负责测量约束，不直接向最终图注入纹理。

配准已明确停用：不估计也不执行平移、仿射、分段或非刚性变换。发布清单会写入 `registration.enabled=false` 和 `transform_applied=false`，便于机器审计。

生成器供应端候选可以保留原生 `1470×1070` 仅用于溯源；一键入口首先生成 `GENERATIVE_denoised_guide_shape_conditioned_2200x1600.png`，然后才运行增强后处理。处理顺序记录在 `generation_preprocess_size_manifest.json` 中，几何投影会断言输入与输出数组均与原图严格同尺寸。

## 生成后的一键复现

外部生成候选本身无法由本地容器确定性重建。已有软生成候选后，可运行：

```bash
python app/run_v11_pipeline.py \
  --source input/source_16bit.tif \
  --guide results_sota/01_sota_blind/MEASUREMENT_sota_geometry_blind_16bit.tif \
  --generated input/GENERATIVE_denoised_guide_shape_conditioned_raw.png \
  --outdir results_generative_shape_v11
```

容器运行：

```bash
docker run --rm --entrypoint python \
  -v "$PWD:/workspace" -w /workspace \
  ct-generative-blind-restore:cpu \
  app/run_v11_pipeline.py \
  --source input/source_16bit.tif \
  --guide results_sota/01_sota_blind/MEASUREMENT_sota_geometry_blind_16bit.tif \
  --generated input/GENERATIVE_denoised_guide_shape_conditioned_raw.png \
  --outdir results_generative_shape_v11
```

固定参数与预期指标见 [`config/v11_profile.json`](config/v11_profile.json)，完整技术说明见 [`GENERATIVE_DENOISED_GUIDE_CONSTRAINT_V11_REPORT.md`](GENERATIVE_DENOISED_GUIDE_CONSTRAINT_V11_REPORT.md)。

容器测试和归档/复现像素完全一致的机器可读证据见 [`config/v11_reproduction_validation.json`](config/v11_reproduction_validation.json)。

本次尺寸锁定的逐阶段证据同时写入 `02_postprocessed/size_lock_manifest.json`、`03_hard_shape_projection/hard_shape_projection_metrics.json` 和根目录的 `v11_release_manifest.json`。

## 本图固化结果

| 指标 | v11结果 |
|---|---:|
| 层纹 | 左49 + 右51 |
| 夹层 | 98 |
| 约束匹配层纹 | 100 / 100 |
| 端点偏移P95 | 0.0856 px |
| 长度变化P95 | 0.1006 px |
| FWHM相对误差中位数 | 0.0182% |
| FWHM相对误差P95 | 0.2010% |
| 目标/输出中位FWHM | 4.251057 / 4.251983 px |

## 文件边界

GitHub源码包包含代码、Docker文件、测试、配置和数值报告，不包含原始图、参考图、模型权重或生成结果。本地完整实验包另行保存这些数据。v11最终图含生成像素，只能作为视觉增强结果，不能作为计量认证图或无噪声真值。
