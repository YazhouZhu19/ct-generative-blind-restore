SELECT
  metric,
  stage,
  value,
  side,
  improvement,
  fwhm_drift
FROM blind_before_after
ORDER BY
  CASE metric
    WHEN '左侧连续性 CV' THEN 1
    WHEN '右侧连续性 CV' THEN 2
    WHEN '左侧断裂率' THEN 3
    WHEN '右侧断裂率' THEN 4
    ELSE 5
  END,
  CASE stage WHEN '处理前' THEN 1 ELSE 2 END;
