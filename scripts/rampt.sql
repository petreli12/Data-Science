SELECT
  DATE_TRUNC('day', transaction_time) AS transaction_date,
  AVG(transaction_amount) OVER (ORDER BY DATE_TRUNC('day', transaction_time) RANGE BETWEEN INTERVAL '2' DAY PRECEDING AND CURRENT ROW) AS rolling_avg_3_day
FROM
  transactions
WHERE
  DATE_TRUNC('day', transaction_time) BETWEEN '2021-01-29' AND '2021-01-31'
ORDER BY
  transaction_date;
