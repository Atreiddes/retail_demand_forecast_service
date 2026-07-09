"""Генератор дашбордов Grafana: собирает валидный JSON, имена метрик совпадают с prom.py.
Запуск: python monitoring/gen_dashboards.py - перезаписывает monitoring/grafana/dashboards/*.json.
"""
import json
import pathlib

OUT = pathlib.Path(__file__).resolve().parent / "grafana" / "dashboards"
OUT.mkdir(parents=True, exist_ok=True)
DS = {"type": "prometheus", "uid": "prometheus"}


def targets(specs, instant=False):
    out = []
    for i, (expr, legend) in enumerate(specs):
        t = {"refId": chr(65 + i), "datasource": DS, "expr": expr,
             "instant": instant, "range": not instant}
        if legend:
            t["legendFormat"] = legend
        out.append(t)
    return out


def stat(pid, title, specs, gp, unit="short"):
    return {
        "id": pid, "type": "stat", "title": title, "datasource": DS,
        "gridPos": gp, "targets": targets(specs, instant=True),
        "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
        "options": {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                    "orientation": "auto", "colorMode": "value", "graphMode": "area",
                    "textMode": "value_and_name"},
    }


def ts(pid, title, specs, gp, unit="short"):
    return {
        "id": pid, "type": "timeseries", "title": title, "datasource": DS,
        "gridPos": gp, "targets": targets(specs),
        "fieldConfig": {"defaults": {"unit": unit, "custom": {
            "drawStyle": "line", "lineWidth": 1, "fillOpacity": 10, "showPoints": "never"}},
            "overrides": []},
        "options": {"legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
                    "tooltip": {"mode": "multi", "sort": "desc"}},
    }


def dash(title, uid, panels):
    return {"title": title, "uid": uid, "schemaVersion": 39, "version": 1, "editable": True,
            "time": {"from": "now-1h", "to": "now"}, "timezone": "", "refresh": "15s",
            "templating": {"list": []}, "annotations": {"list": []}, "panels": panels}


def gp(x, y, w, h):
    return {"x": x, "y": y, "w": w, "h": h}


# Сервис прогноза
service = dash("Сервис прогноза", "forecast-service", [
    stat(1, "Активные прогоны", [('forecast_active_runs{job="forecast-api"}', None)], gp(0, 0, 6, 4)),
    stat(2, "Пачек/с (5м)", [("sum(rate(forecast_chunks_processed_total[5m]))", None)], gp(6, 0, 6, 4)),
    stat(3, "Запросов/с (5м)", [("sum(rate(http_requests_total[5m]))", None)], gp(12, 0, 6, 4), "reqps"),
    stat(4, "WRMSSE (12 уровней)", [('forecast_quality{metric="wrmsse12_mean"}', None)], gp(18, 0, 6, 4)),
    ts(5, "Прогоны по статусам", [("forecast_runs_status", "{{status}}")], gp(0, 4, 12, 8)),
    ts(6, "Обработка пачек, шт/с", [("sum by(result)(rate(forecast_chunks_processed_total[5m]))", "{{result}}")], gp(12, 4, 12, 8)),
    ts(7, "HTTP: запросы/с по маршрутам", [("sum by(path)(rate(http_requests_total[5m]))", "{{path}}")], gp(0, 12, 12, 8), "reqps"),
    ts(8, "HTTP: задержка p95, с", [("histogram_quantile(0.95, sum by(le, path)(rate(http_request_duration_seconds_bucket[5m])))", "{{path}}")], gp(12, 12, 12, 8), "s"),
    ts(9, "Время обработки пачки, с", [
        ('histogram_quantile(0.5, sum by(le)(rate(forecast_chunk_duration_seconds_bucket{job="forecast-worker"}[5m])))', "p50"),
        ('histogram_quantile(0.9, sum by(le)(rate(forecast_chunk_duration_seconds_bucket{job="forecast-worker"}[5m])))', "p90"),
    ], gp(0, 20, 12, 8), "s"),
    ts(10, "Дрейф признаков: PSI и KS", [
        ("forecast_data_drift_psi", "PSI {{feature}}"),
        ("forecast_drift_ks", "KS {{feature}}"),
    ], gp(12, 20, 12, 8)),
    stat(11, "Метрики качества модели", [("forecast_quality", "{{metric}}")], gp(0, 28, 24, 8)),
    stat(12, "Деградация (1=да)", [('forecast_degraded{job="forecast-api"}', None)], gp(0, 36, 6, 4)),
    stat(13, "WMAPE прогноз-факт", [('forecast_accuracy_wmape{job="forecast-api"}', None)], gp(6, 36, 6, 4), "percentunit"),
    stat(14, "Смещение прогноза", [('forecast_accuracy_bias{job="forecast-api"}', None)], gp(12, 36, 6, 4), "percentunit"),
    stat(15, "Покрытие P10-P90", [('forecast_interval_coverage{job="forecast-api"}', None)], gp(18, 36, 6, 4), "percentunit"),
    ts(16, "Точность прогноз-факт во времени", [
        ('forecast_accuracy_wmape{job="forecast-api"}', "WMAPE"),
        ('forecast_interval_coverage{job="forecast-api"}', "покрытие P10-P90"),
    ], gp(0, 40, 24, 8), "percentunit"),
    ts(17, "WMAPE по горизонту", [('forecast_wmape_by_horizon{job="forecast-api"}', "h={{h}}")], gp(0, 48, 12, 8), "percentunit"),
    stat(18, "WMAPE по сегментам", [('forecast_wmape_segment{job="forecast-api"}', "{{segment}}")], gp(12, 48, 12, 8), "percentunit"),
    stat(19, "FVA против MA-4, %", [('forecast_fva_ma4_pct{job="forecast-api"}', None)], gp(0, 56, 6, 4), "percent"),
    stat(20, "Плановое смещение", [('forecast_planning_bias{job="forecast-api"}', None)], gp(6, 56, 6, 4), "percentunit"),
    stat(21, "Смещение по сегментам", [('forecast_bias_segment{job="forecast-api"}', "{{segment}}")], gp(12, 56, 12, 4), "percentunit"),
    ts(22, "Смещение по штату", [('forecast_bias_by_state{job="forecast-api"}', "{{state}}")], gp(0, 60, 24, 6), "percentunit"),
    stat(23, "Полнота последней недели", [('forecast_latest_week_completeness{job="forecast-api"}', None)], gp(0, 66, 5, 4), "percentunit"),
    stat(24, "Свежесть факта", [('forecast_last_actual_week{job="forecast-api"} * 1000', None)], gp(5, 66, 5, 4), "dateTimeAsIso"),
    stat(25, "Новые ряды", [('forecast_series_new{job="forecast-api"}', None)], gp(10, 66, 4, 4)),
    stat(26, "Выбывшие ряды", [('forecast_series_dead{job="forecast-api"}', None)], gp(14, 66, 4, 4)),
    stat(27, "Стабильность прогноза (CoV P50)", [('forecast_revision_volatility{job="forecast-api"}', None)], gp(18, 66, 6, 4), "percentunit"),
    stat(28, "Полнота прогона", [('forecast_run_coverage{job="forecast-api"}', None)], gp(0, 70, 6, 4), "percentunit"),
    stat(29, "Возраст артефакта, дней", [('forecast_artifact_age_days{job="forecast-api"}', None)], gp(6, 70, 6, 4)),
    ts(30, "Фоллбек на базу MA-4, рядов/с", [("sum(rate(forecast_fallback_series_total[5m]))", "фоллбек")], gp(12, 70, 12, 8)),
])

# Очередь RabbitMQ
rabbit = dash("Очередь RabbitMQ", "rabbitmq", [
    stat(1, "Готовы к доставке", [("rabbitmq_queue_messages_ready", None)], gp(0, 0, 6, 4)),
    stat(2, "В обработке (unacked)", [("rabbitmq_queue_messages_unacked", None)], gp(6, 0, 6, 4)),
    stat(3, "Соединения", [("rabbitmq_connections", None)], gp(12, 0, 6, 4)),
    stat(4, "Потребители", [("rabbitmq_consumers", None)], gp(18, 0, 6, 4)),
    ts(5, "Сообщения в очереди", [
        ("rabbitmq_queue_messages_ready", "готовы"),
        ("rabbitmq_queue_messages_unacked", "в обработке"),
    ], gp(0, 4, 12, 8)),
    ts(6, "Публикация и доставка, /с", [
        ("sum(rate(rabbitmq_channel_messages_published_total[5m]))", "публикация"),
        ("sum(rate(rabbitmq_channel_messages_delivered_total[5m]))", "доставка"),
    ], gp(12, 4, 12, 8)),
])

# База Postgres
postgres = dash("База Postgres", "postgres", [
    stat(1, "Доступность", [("pg_up", None)], gp(0, 0, 6, 4)),
    stat(2, "Соединения", [("sum(pg_stat_database_numbackends)", None)], gp(6, 0, 6, 4)),
    stat(3, "Размер БД", [('pg_database_size_bytes{datname="forecast"}', None)], gp(12, 0, 6, 4), "bytes"),
    stat(4, "Коммиты/с (5м)", [("sum(rate(pg_stat_database_xact_commit[5m]))", None)], gp(18, 0, 6, 4)),
    ts(5, "Транзакции/с", [
        ("sum(rate(pg_stat_database_xact_commit[5m]))", "commit"),
        ("sum(rate(pg_stat_database_xact_rollback[5m]))", "rollback"),
    ], gp(0, 4, 12, 8)),
    ts(6, "Соединения по базам", [("pg_stat_database_numbackends", "{{datname}}")], gp(12, 4, 12, 8)),
])

# Контейнеры (cAdvisor). На Docker Desktop имена не резолвятся, поэтому по cgroup-id
containers = dash("Контейнеры", "containers", [
    stat(1, "Память контейнеров, сумма", [('sum(container_memory_usage_bytes{id=~"/docker/.+"})', None)], gp(0, 0, 8, 4), "bytes"),
    stat(2, "Макс. занятость файловой системы", [("max(container_fs_usage_bytes / container_fs_limit_bytes)", None)], gp(8, 0, 8, 4), "percentunit"),
    stat(3, "Контейнеров", [('count(container_last_seen{id=~"/docker/.+"})', None)], gp(16, 0, 8, 4)),
    ts(4, "Память по контейнерам", [('container_memory_usage_bytes{id=~"/docker/.+"}', "{{id}}")], gp(0, 4, 12, 8), "bytes"),
    ts(5, "CPU по контейнерам, ядер", [('sum by(id)(rate(container_cpu_usage_seconds_total{id=~"/docker/.+"}[5m]))', "{{id}}")], gp(12, 4, 12, 8)),
])

for name, d in [("forecast-service", service), ("rabbitmq", rabbit), ("postgres", postgres),
                ("containers", containers)]:
    (OUT / f"{name}.json").write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print("написан", name)
