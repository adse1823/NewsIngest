-- Company and sector node table for the knowledge graph
CREATE OR REPLACE TABLE entity_table AS
SELECT
    ticker,
    CASE
        WHEN ticker IN ('AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META') THEN 'Technology'
        WHEN ticker IN ('AMZN')                                    THEN 'Consumer Discretionary'
        WHEN ticker IN ('TSLA')                                    THEN 'Automotive'
        WHEN ticker IN ('JPM', 'BAC', 'GS')                       THEN 'Financials'
        ELSE 'Other'
    END AS sector,
    ROW_NUMBER() OVER (ORDER BY ticker) - 1 AS node_id
FROM (SELECT DISTINCT ticker FROM raw_news) t;
