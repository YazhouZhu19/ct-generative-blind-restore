SELECT
  noise_reduction,
  edge_gain,
  contrast_gain,
  max_fwhm_drift,
  quality_ssim,
  length_pass_fraction,
  image_count,
  image_width,
  image_height,
  bit_depth
FROM headline_metrics
LIMIT 1;
