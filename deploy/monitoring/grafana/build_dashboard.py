"""Generate grafana/dashboards/triage.json (run from the repository root)."""

import json

DS = {"type": "prometheus", "uid": "prometheus"}
panels = []
next_id = [1]


def panel(kind, title, targets, x, y, w, h, **extra):
    p = {"id": next_id[0], "type": kind, "title": title, "datasource": DS,
         "gridPos": {"x": x, "y": y, "w": w, "h": h},
         "targets": [{"datasource": DS, "refId": chr(65 + i), "expr": e, "legendFormat": l}
                     for i, (e, l) in enumerate(targets)]}
    p.update(extra)
    next_id[0] += 1
    panels.append(p)


def row(title, y):
    panels.append({"id": next_id[0], "type": "row", "title": title, "collapsed": False,
                   "gridPos": {"x": 0, "y": y, "w": 24, "h": 1}, "panels": []})
    next_id[0] += 1


def thresholds(*steps):
    return {"mode": "absolute", "steps": [{"color": c, "value": v} for v, c in steps]}


row("Service", 0)
panel("stat", "Model loaded", [("triage_model_loaded", "")], 0, 1, 6, 4,
      fieldConfig={"defaults": {"mappings": [{"type": "value", "options": {
          "0": {"text": "Not loaded", "color": "red"}, "1": {"text": "Loaded", "color": "green"}}}],
          "noValue": "API down"}, "overrides": []},
      options={"colorMode": "background", "reduceOptions": {"calcs": ["lastNotNull"]}})
panel("stat", "Model load duration", [("triage_model_load_duration_seconds", "")], 6, 1, 6, 4,
      fieldConfig={"defaults": {"unit": "s", "thresholds": thresholds(
          (None, "green"), (60, "red"))}, "overrides": []},
      options={"colorMode": "value", "reduceOptions": {"calcs": ["lastNotNull"]}})
panel("stat", "Served weights (SHA-256) and training commit",
      [("triage_model_info", "{{weights_sha256}} · {{training_commit}}")], 12, 1, 12, 4,
      options={"textMode": "name", "colorMode": "none",
               "reduceOptions": {"calcs": ["lastNotNull"]}})
panel("timeseries", "/predict requests per second by status class",
      [('sum by (status_class) (rate(triage_requests_total{route="/predict"}[1m]))',
        "{{status_class}}")], 0, 5, 12, 8, fieldConfig={"defaults": {"unit": "reqps"}, "overrides": []})
panel("timeseries", "/predict latency",
      [('histogram_quantile(0.5, sum by (le) (rate(triage_request_duration_seconds_bucket{route="/predict"}[5m])))', "p50"),
       ('histogram_quantile(0.95, sum by (le) (rate(triage_request_duration_seconds_bucket{route="/predict"}[5m])))', "p95")],
      12, 5, 12, 8, fieldConfig={"defaults": {"unit": "s", "custom": {
          "thresholdsStyle": {"mode": "line"}}, "thresholds": thresholds((None, "green"), (0.5, "red"))},
          "overrides": []})
panel("timeseries", "5xx share of /predict (5 min)",
      [('sum(increase(triage_requests_total{route="/predict", status_class="5xx"}[5m])) / sum(increase(triage_requests_total{route="/predict"}[5m]))', "5xx share")],
      0, 13, 12, 7, fieldConfig={"defaults": {"unit": "percentunit", "custom": {
          "thresholdsStyle": {"mode": "line"}}, "thresholds": thresholds(
          (None, "green"), (0.01, "orange"), (0.05, "red"))}, "overrides": []})
panel("timeseries", "Requests in flight", [("triage_requests_in_flight", "in flight")],
      12, 13, 12, 7)

row("Model behaviour (drift, completed windows only)", 20)
panel("stat", "Current drift status", [("triage_drift == 1", "{{triage_drift}}")], 0, 21, 8, 4,
      description="'not_evaluated' means no window has completed since the API started; it does not mean there is no problem.",
      fieldConfig={"defaults": {"noValue": "API down"}, "overrides": []},
      options={"textMode": "name", "colorMode": "none",
               "reduceOptions": {"calcs": ["lastNotNull"]}})
panel("stat", "Minutes since last window result",
      [("(time() - triage_drift_last_result_timestamp_seconds) / 60 and triage_drift_last_result_timestamp_seconds > 0", "")],
      8, 21, 8, 4, fieldConfig={"defaults": {"unit": "m", "decimals": 0,
          "noValue": "No result since start"}, "overrides": []},
      options={"colorMode": "none", "reduceOptions": {"calcs": ["lastNotNull"]}})
panel("stat", "Minutes since evaluator heartbeat",
      [("(time() - triage_drift_heartbeat_timestamp_seconds) / 60", "")], 16, 21, 8, 4,
      fieldConfig={"defaults": {"unit": "m", "decimals": 1}, "overrides": []},
      options={"colorMode": "none", "reduceOptions": {"calcs": ["lastNotNull"]}})
panel("timeseries", "PSI by check (history; restarts marked)",
      [("triage_drift_psi", "{{check}}")], 0, 25, 12, 8,
      fieldConfig={"defaults": {"custom": {"thresholdsStyle": {"mode": "line"}},
          "thresholds": thresholds((None, "green"), (0.10, "orange"), (0.25, "red"))},
          "overrides": []})
panel("timeseries", "Uncertain rate (history; reference 7.5%)",
      [("triage_drift_uncertain_rate", "uncertain rate")], 12, 25, 12, 8,
      fieldConfig={"defaults": {"unit": "percentunit", "custom": {
          "thresholdsStyle": {"mode": "line"}}, "thresholds": thresholds(
          (None, "green"), (0.15, "orange"), (0.25, "red"))}, "overrides": []})

dashboard = {
    "uid": "triage-ops", "title": "Medical Specialty Triage — Operations",
    "description": "Operational metrics and drift results; no text or per-specialty values.",
    "schemaVersion": 39, "version": 1, "editable": False, "refresh": "15s",
    "time": {"from": "now-1h", "to": "now"}, "tags": ["triage"],
    "annotations": {"list": [{
        "name": "API restart", "datasource": DS, "enable": True, "iconColor": "purple",
        "expr": 'changes(process_start_time_seconds{job="triage-api"}[1m]) > 0',
        "titleFormat": "API restart"}]},
    "panels": panels,
}
with open("deploy/monitoring/grafana/dashboards/triage.json", "w", encoding="utf-8", newline="\n") as f:
    json.dump(dashboard, f, indent=2)
    f.write("\n")
print(len(panels), "panels")
