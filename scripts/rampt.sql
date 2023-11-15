SELECT
  transaction_date,
  AVG(transaction_amount) AS rolling_avg_3_day
FROM (
  SELECT
    DATE(transaction_time) AS transaction_date,
    transaction_amount
  FROM
    transactions
) AS date_amount
WHERE
  transaction_date BETWEEN '2021-01-29' AND '2021-01-31'
GROUP BY
  transaction_date
HAVING
  transaction_date = '2021-01-31'
ORDER BY
  transaction_date;

