# v11 去噪引导双证据生成约束报告

## 目标

v10 直接在原始16位图像上检测层纹中心线、端点和宽度。该方案能够锁定生成结果的几何，但原图噪声可能进入中心线跟踪和半高宽测量。v11 改为先获得不改变层纹几何的盲去噪图，再从该图测量约束；原图仍是生成模型的唯一内容目标，并在相同坐标上提供独立验证。

## 数据流与角色隔离

```text
原始16位图 ──盲去噪──> 去噪测量引导图 ──测量──> 几何约束图/CSV
     │                         │                    │
     │                         └──原图同坐标校验──> 置信度
     └──────────生成目标 + 几何约束 + 外观参考────> 软生成图
                                                     │
                               后处理 + 解析硬投影 <──┘
```

- 原图：唯一生成内容、视野和场景目标；不再直接定义约束坐标。
- 盲去噪引导图：检测层数、中心轨迹、亚像素端点和横向FWHM。
- 原图同坐标证据：只计算测量吻合度和置信度，不与引导坐标做平均，因此不会把噪声拉回约束。
- 参考图：只提供洁净边界、片状层纹和完整中央高亮的外观目标。

## 约束改进

### 1. 去噪后测量与双证据置信度

每条层纹均在去噪引导图上完成25个横截面的FWHM统计和三条平行路径的端点重复测量。随后在原图的同一条路径上重新测量。置信度由引导图端点重复性、端点SNR、有效宽度样本数，以及原图/引导图的端点、长度和宽度吻合度组成。最终坐标始终取自去噪引导图。

### 2. 层纹束联合中心线正则化

旧版逐条中心线独立平滑仍可能保留不相关的蛇形抖动。v11 对左右两组层纹分别估计共同低频位移，并只保留强平滑后的30%单层偏差：

\[
p_i(y)=a_i+c(y)+0.30\,r_i(y)
\]

其中 `a_i` 是第 `i` 条层纹横向锚点，`c(y)` 是同侧层纹束的稳健共同变形，`r_i(y)` 是单层残差。中心线步进标准差中位数由 `0.2769 px` 降至 `0.00767 px`，同时保留层纹束的整体缓慢弯曲。

### 3. 逐行夹层约束

夹层宽度不再由单一中心距近似，而是在相邻层纹共同存在的每一行计算：

\[
g_i(y)=p_{i+1}(y)-p_i(y)-\frac{w_i+w_{i+1}}{2}
\]

CSV 同时记录中位数、P10、P90、样本数、共同长度和相邻层纹最低置信度。生成条件图把完整夹层走廊编码为蓝色区域，而不只是画一条中心线。

### 4. 生成周期抑制、有限宽片层与FWHM闭环

软生成器可能产生错误的层纹周期。硬投影先以不小于一个实测层距的横向Gaussian核去掉生成器周期，只保留低频背景和亮度。随后用双边logistic剖面按每条层纹实测中心线、端点和FWHM重建具有明确左右边界的有限宽片层，而不是尖状Gaussian线。由于背景和亚像素光栅化会轻微改变最终FWHM，系统用与审计相同的测量算子执行三轮逐层闭环校准，但不移动中心线或端点。

## 本图结果

| 指标 | v11结果 |
|---|---:|
| 左/右层纹 | 49 / 51 |
| 夹层 | 98 |
| 通过约束质量门限 | 100 / 100 |
| 约束置信度中位数 | 0.9849 |
| 原图—引导端点差异P95 | 0.184 px |
| 原图—引导宽度相对差异P95 | 15.13% |
| 引导测得层纹宽度中位数 | 4.251 px |
| 约束匹配可分辨层纹 | 100 / 100 |
| 最终端点偏移P95 | 0.0856 px |
| 最终长度变化P95 | 0.1006 px |
| 最终宽度相对误差中位数 | 0.0182% |
| 最终宽度相对误差P95 | 0.2010% |
| 最终宽度中位数 | 4.251 px |

相较v10，宽度误差中位数由4.54%降至0.0182%，P95由12.33%降至0.2010%。生成器原生图先被重采样到原图尺寸，再执行全部后处理和投影。普通无匹配峰值检测器会受宽片内部轮廓和极近相邻层影响；因此验收使用约束坐标逐层匹配，100条层纹均具有超过阈值的中心—夹层对比度。

## 代码与运行

约束生成：

```bash
python app/generative_shape_constraint.py \
  --source input/source_16bit.tif \
  --guide results_sota/01_sota_blind/MEASUREMENT_sota_geometry_blind_16bit.tif \
  --outdir results_generative_shape_v11/00_guide_constraints
```

生成模型以原图、`SHAPE_generation_condition.png` 和未配准外观参考为三个角色明确的输入。生成后先锁定原图尺寸，再运行原有 `postprocess_visual`：

```bash
python app/generative_postprocess.py \
  --generated results_generative_shape_v11/01_generated/GENERATIVE_denoised_guide_shape_conditioned_raw.png \
  --outdir results_generative_shape_v11/02_postprocessed \
  --match-source input/source_16bit.tif \
  --resize-before-postprocess
```

然后运行几何投影：

```bash
python app/generative_shape_project.py \
  --profile v11 \
  --source input/source_16bit.tif \
  --guide results_sota/01_sota_blind/MEASUREMENT_sota_geometry_blind_16bit.tif \
  --generated results_generative_shape_v11/02_postprocessed/GENERATIVE_visual_only_postprocessed.png \
  --outdir results_generative_shape_v11/03_hard_shape_projection
```

## 使用边界

最终图的几何由测量约束解析重建，但灰度纹理、低频背景和视觉外观仍含生成成分，不能作为计量认证真值。厚度、长度和缺陷验收应使用非生成式16位测量链、像素标定和系统PSF/MTF验证。
