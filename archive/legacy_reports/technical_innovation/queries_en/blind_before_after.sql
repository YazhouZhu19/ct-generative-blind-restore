SELECT metric, stage, value, side, improvement, fwhm_drift
FROM blind_before_after
ORDER BY
  CASE metric
    WHEN 'Left continuity CV' THEN 1
    WHEN 'Right continuity CV' THEN 2
    WHEN 'Left dropout rate' THEN 3
    WHEN 'Right dropout rate' THEN 4
    ELSE 5
  END,
  CASE stage WHEN 'Before' THEN 1 ELSE 2 END;
