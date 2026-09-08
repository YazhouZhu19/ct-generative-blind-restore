# v17 结构载体条件扩散生成技术报告

## 1. 目标与定位

v15/v16为了保证测量安全，最终测量图不包含生成像素。该策略避免了结构幻觉，但也限制了层纹主体和雾状背景的进一步增强。v17验证另一条路线：把已经计算并审计过的v16结构载体前移为生成模型的显式条件，在生成网络内部加入可微结构/细节损失，并在生成后继续执行独立逐层审计。

v17输出包含生成残差，因此命名为`MEASUREMENT_CANDIDATE`。它没有替代v16；v16仍是不可覆盖的回退基线和数值参考。

## 2. 条件生成架构

`app/structure_conditioned_diffusion.py`实现轻量有界残差DDIM。网络不是自由生成整张图，而是预测结构载体附近的有限残差：

\[
\hat{x}_0=C+A\odot\delta_{\max}\tanh R_\theta(x_t,G,t)
\]

其中`C`为v16结构载体，`A`为结构允许修改场，`G`包含十个条件通道：

1. 结构载体灰度；
2. 低频外观生成候选；
3. 载体横向梯度；
4. 载体纵向梯度；
5. 弯曲层纹中心线场；
6. 有限宽层纹边界场；
7. 上下端点保护场；
8. 夹层场；
9. 原图/载体双证据置信度；
10. 盲去噪预测不确定度。

边界和端点位置的允许残差接近零；层纹内部的残差自由度低于平坦夹层和中央实体区。网络输出头采用零初始化，因此训练失败或未收敛时从结构载体而非随机图像开始。

## 3. 单图扩散适配目标

当前只有一张输入图，不能把完整生成器从零训练为通用模型。v17因此使用单图Patch适配：从载体构造受边界保护的伪干净目标，再训练条件扩散残差。

伪目标组合：

- 非层纹低结构区：0.58强度非局部均值；
- 层纹内部：0.24强度纵向零相位平滑；
- 低梯度区域：0.10强度、严格限幅的生成外观低频差；
- 总变化：由载体MAD噪声估计给出硬上限；
- 生成候选像素不直接写回伪目标或最终结果。

正式训练参数为320轮、96×96 Patch、Batch 2、12个基础特征通道、48个扩散噪声级和6步确定性DDIM采样。全分辨率推理使用192像素分块与40像素重叠。

## 4. 复合结构与细节损失

总损失包含：

\[
\begin{aligned}
L={}&0.20L_{diff}+3.00L_{recon}+0.80L_{carrier}\\
&+3.00L_{edge-x}+2.20L_{edge-y}\\
&+2.00L_{width}+1.50L_{endpoint}\\
&+1.20L_{detail}+0.80L_{gap}+2.00L_{boundary}.
\end{aligned}
\]

- `diff`：扩散噪声预测一致性；
- `recon`：伪干净目标Charbonnier重建；
- `carrier`：按端点、边界和双证据置信度加权的载体一致性；
- `edge-x`：保护横向厚度边缘；
- `edge-y`：保护纵向端点边缘；
- `width`：横向投影梯度，约束层纹宽度分布；
- `endpoint`：纵向投影梯度，约束长度和端点；
- `detail`：3/9像素尺度差分，保护层纹中尺度轴向细节；
- `gap`：保护夹层低频灰度和分隔；
- `boundary`：保护有限宽边界场的梯度幅值。

这些损失直接作用于每次扩散预测的干净图，而不是只在生成结束后比较一次。

## 5. 采样与后处理约束

生成结束后执行零相位低结构残差清理，并将所有变化重新投影到以载体为中心的幅度范围内。不执行缩放、配准、坐标形变或解析层纹重绘。

随后搜索生成残差强度`0、0.12、0.20、0.30、0.42、0.55、0.70、0.85、1.00`。每个候选重新测量100条层纹和98个夹层，并检查：

- 端点和长度P95；
- 层纹FWHM宽度和双证据通过数；
- 夹层宽度、长度和双证据通过数；
- 边缘清晰度；
- 低频、中频和梯度相关性；
- 每条层纹轴向细节相关性；
- SSIM。

强度0对应未经修改的v16载体，是强制保留的失败回退。

## 6. 当前图像结果

正式运行选择生成残差强度0.85：

| 指标 | v17相对v16 |
|---|---:|
| 输出尺寸/类型 | 2200×1600 / uint16 |
| 层纹轴向噪声下降 | 1.05% |
| 中央高频噪声下降 | 2.67% |
| 平坦区高频噪声下降 | 1.51% |
| 边缘清晰度变化 | +0.24% |
| 层纹宽度误差P95 | 0.3409% |
| 端点偏差P95 | 0.00043 px |
| 长度偏差P95 | 0.05993 px |
| 夹层宽度误差P95 | 0.1911% |
| 夹层长度误差P95 | 0.07970 px |
| 低频/中频相关性 | 1.000000 / 0.999990 |
| 梯度相关性 | 0.999982 |
| 层纹细节相关性中位数/P10 | 0.999991 / 0.999985 |
| SSIM | 0.999968 |

全部连续几何、细节、边缘和SSIM门槛通过。画布目标ROI以外与v16逐像素一致。

0.30至0.70以及1.00强度在离散层纹双证据计数上下降，因而被拒绝；0.85恢复到基线计数并通过连续误差门槛。这种非单调阈值行为说明单图审计仍不足以证明跨样本安全性。

## 7. 运行

```bash
docker compose run --rm ct-v17-structure-diffusion
```

或：

```bash
python app/structure_conditioned_diffusion.py \
  --source input/source_16bit.tif \
  --carrier results_generative_shape_v16_measurement_quality/FINAL_MEASUREMENT_v16_quality_enhanced_2200x1600_16bit.tif \
  --proposal input/generative_candidate_visual_only.png \
  --uncertainty results_sota/01_sota_blind/MEASUREMENT_sota_uncertainty_float32.tif \
  --outdir results_generative_shape_v17_structure_conditioned_diffusion
```

主要输出为：

- `MEASUREMENT_CANDIDATE_v17_structure_conditioned_16bit.tif`；
- `MEASUREMENT_CANDIDATE_v17_comparison.png`；
- `MEASUREMENT_CANDIDATE_v17_boundary_overlay.png`；
- `AUDIT_v17_structure_conditions.png`；
- `lamella_v17_comparison.csv`；
- `interlayer_v17_comparison.csv`；
- `structure_detail_v17.csv`；
- `structure_conditioned_diffusion_v17_metrics.json`；
- `structure_conditioned_diffusion_v17.pt`。

## 8. 使用边界

v17证明了“结构载体条件输入 + 生成训练损失 + 采样幅度约束 + 独立审计”的完整工程闭环，但当前仅在一张图像上进行了单图适配。投入长度或厚度测量前，必须使用同设备多张重复扫描、标准样件、像素尺寸标定和PSF/MTF测试验证偏差。未完成这些验证时，v16仍是更保守的测量输入。
