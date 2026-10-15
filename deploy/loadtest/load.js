// Steady load against /predict (docs/operations-plan.md §9). Each iteration posts one
// synthetic payload chosen at random: 24 short descriptions and 8 long joined ones.
import http from "k6/http";
import { check } from "k6";
import { SharedArray } from "k6/data";

const payloads = new SharedArray("payloads",
  () => JSON.parse(open("/loadtest/payloads.json")).payloads);

export const options = {
  vus: Number(__ENV.VUS || 1),
  duration: __ENV.DURATION || "3m",
  summaryTrendStats: ["avg", "med", "p(95)", "p(99)", "max"],
};

export default function () {
  const p = payloads[Math.floor(Math.random() * payloads.length)];
  const res = http.post(`${__ENV.BASE_URL || "http://api:8000"}/predict`,
    JSON.stringify({ description: p.description }),
    // Marked synthetic so load tests never enter the drift windows.
    { headers: { "Content-Type": "application/json", "X-Synthetic-Traffic": "1" },
      tags: { kind: p.kind } });
  check(res, { "status 200": (r) => r.status === 200 });
}
