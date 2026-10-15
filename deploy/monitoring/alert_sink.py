"""Minimal webhook receiver: logs each Alertmanager notification as one JSON line.

Stands in for email or chat delivery in the local monitoring profile; its log is the
evidence that an alert travelled the whole path. Standard library only.
"""

import json
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 (http.server naming)
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        for alert in body.get("alerts", []):
            print(json.dumps({
                "received_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "status": alert.get("status"),
                "alert": alert.get("labels", {}).get("alertname"),
                "severity": alert.get("labels", {}).get("severity"),
                "starts_at": alert.get("startsAt"),
                "summary": alert.get("annotations", {}).get("summary"),
            }), flush=True)
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):  # silence per-request access logs
        pass


if __name__ == "__main__":
    print("alert-sink listening on :8080", file=sys.stderr, flush=True)
    HTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
