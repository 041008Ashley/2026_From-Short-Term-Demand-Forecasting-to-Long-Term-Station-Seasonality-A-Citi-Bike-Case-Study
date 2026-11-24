USE citibike_202511_202610;

-- ==================================================
-- 1. 查看固定的 20 个站点，不重新选择或排名
-- ==================================================

SELECT station_id
FROM selected_stations
ORDER BY station_id;


-- ==================================================
-- 2. 生成三个月的完整小时序列
-- 共 92 天 × 24 小时 = 2208 个小时
-- ==================================================

SET SESSION cte_max_recursion_depth = 3000;

DROP TABLE IF EXISTS analysis_hours;

CREATE TABLE analysis_hours (
    `hour` DATETIME NOT NULL PRIMARY KEY
) ENGINE = InnoDB;

INSERT INTO analysis_hours (`hour`)
WITH RECURSIVE hours AS (
    SELECT
        CAST('2026-08-01 00:00:00' AS DATETIME) AS h

    UNION ALL

    SELECT
        h + INTERVAL 1 HOUR
    FROM hours
    WHERE h + INTERVAL 1 HOUR < '2026-11-01 00:00:00'
)
SELECT h
FROM hours;


-- ==================================================
-- 3. 创建“站点—小时”结果表
--
-- departures：这一小时从该站出发的订单数
-- arrivals：这一小时到达该站的订单数
-- net_flow：到达量 - 出发量
-- ==================================================

DROP TABLE IF EXISTS station_hourly_demand;

CREATE TABLE station_hourly_demand (
    station_id VARCHAR(64)
        CHARACTER SET ascii
        COLLATE ascii_bin NOT NULL,

    `hour` DATETIME NOT NULL,

    departures BIGINT NOT NULL,
    arrivals BIGINT NOT NULL,
    net_flow BIGINT NOT NULL,

    PRIMARY KEY (station_id, `hour`),

    INDEX idx_hour (`hour`),

    CHECK (departures >= 0 AND arrivals >= 0),
    CHECK (net_flow = arrivals - departures)
) ENGINE = InnoDB;


-- ==================================================
-- 4. 分别聚合出发量和到达量，补齐零订单小时
--
-- 出发量按 started_at 统计。
-- 到达量按 ended_at 统计。
-- 两者分别匹配站点，不要求起终点都属于这 20 个站点。
-- ==================================================

INSERT INTO station_hourly_demand (
    station_id,
    `hour`,
    departures,
    arrivals,
    net_flow
)

WITH departures AS (
    SELECT
        t.start_station_id AS station_id,

        CAST(
            DATE_FORMAT(t.started_at, '%Y-%m-%d %H:00:00')
            AS DATETIME
        ) AS h,

        COUNT(*) AS n

    FROM trips AS t

    INNER JOIN selected_stations AS s
        ON s.station_id = t.start_station_id

    WHERE t.started_at >= '2026-08-01 00:00:00'
      AND t.started_at <  '2026-11-01 00:00:00'

    GROUP BY
        t.start_station_id,
        h
),

arrivals AS (
    SELECT
        t.end_station_id AS station_id,

        CAST(
            DATE_FORMAT(t.ended_at, '%Y-%m-%d %H:00:00')
            AS DATETIME
        ) AS h,

        COUNT(*) AS n

    FROM trips AS t

    INNER JOIN selected_stations AS s
        ON s.station_id = t.end_station_id

    WHERE t.ended_at >= '2026-08-01 00:00:00'
      AND t.ended_at <  '2026-11-01 00:00:00'

    GROUP BY
        t.end_station_id,
        h
)

SELECT
    s.station_id,
    h.`hour`,

    COALESCE(d.n, 0) AS departures,
    COALESCE(a.n, 0) AS arrivals,

    CAST(COALESCE(a.n, 0) AS SIGNED)
    - CAST(COALESCE(d.n, 0) AS SIGNED) AS net_flow

FROM selected_stations AS s

CROSS JOIN analysis_hours AS h

LEFT JOIN departures AS d
    ON d.station_id = s.station_id
   AND d.h = h.`hour`

LEFT JOIN arrivals AS a
    ON a.station_id = s.station_id
   AND a.h = h.`hour`;


-- ==================================================
-- 5. 更新结果表统计信息
-- ==================================================

ANALYZE TABLE station_hourly_demand;


-- ==================================================
-- 6. 检查结果规模
--
-- station_count：20
-- hour_count：2208
-- total_rows：44160
-- first_hour：2026-08-01 00:00:00
-- last_hour：2026-10-31 23:00:00
-- ==================================================

SELECT
    COUNT(DISTINCT station_id) AS station_count,
    COUNT(DISTINCT `hour`) AS hour_count,
    COUNT(*) AS total_rows,
    MIN(`hour`) AS first_hour,
    MAX(`hour`) AS last_hour
FROM station_hourly_demand;


-- ==================================================
-- 7. 检查每个站点是否都有 2208 个小时
-- 正常情况下返回 0 行
-- ==================================================

SELECT
    station_id,
    COUNT(*) AS hour_count
FROM station_hourly_demand
GROUP BY station_id
HAVING COUNT(*) <> 2208;


-- ==================================================
-- 8. 检查数量与净流量是否存在异常
-- 正常情况下返回 0 行
-- ==================================================

SELECT *
FROM station_hourly_demand
WHERE departures < 0
   OR arrivals < 0
   OR net_flow <> arrivals - departures;


-- ==================================================
-- 9. 查看前 100 条结果
-- ==================================================

SELECT
    station_id,
    `hour`,
    departures,
    arrivals,
    net_flow
FROM station_hourly_demand
ORDER BY
    station_id,
    `hour`
LIMIT 100;