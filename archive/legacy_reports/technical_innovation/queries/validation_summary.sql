SELECT
  "order",
  metric,
  observed,
  threshold,
  status,
  interpretation
FROM validation_summary
ORDER BY "order" ASC;
