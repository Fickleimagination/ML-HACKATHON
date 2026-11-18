const trip = document.querySelector("#trip");
const run = document.querySelector("#run");
const api = document.querySelector("#api");
const status = document.querySelector("#status");
const map = document.querySelector("#map");
const placeholder = document.querySelector("#map-placeholder");
const results = document.querySelector("#results");
const metrics = document.querySelector("#metrics");
const mapTrip = document.querySelector("#map-trip");
const metricNames = {
  top1_accuracy: "Top-1 accuracy",
  top1_high_confidence: "Top-1 (high-conf. labels)",
  macro_f1: "Candidate macro F1",
  candidate_miss_rate: "Candidate miss rate",
  continuity: "Observation continuity",
  route_continuity: "Route continuity",
  route_precision: "Route precision",
  route_recall: "Route recall",
};
const percent = new Intl.NumberFormat(undefined, { style: "percent", maximumFractionDigits: 1 });
const defaultApi = new URL(location.href);
const servedByApi = location.pathname.startsWith("/ui/");
if (!servedByApi) defaultApi.port = "8000";
defaultApi.pathname = "/";
let revision = 0;
let predictor;
let connectedApi;
api.value = new URLSearchParams(location.search).get("api")
  || (servedByApi ? defaultApi.origin : localStorage.getItem("pathsnap-api") || defaultApi.origin);

function message(text, error = false) {
  status.textContent = text;
  status.dataset.error = String(error);
}

function clearResult() {
  results.hidden = true;
  map.hidden = true;
  map.removeAttribute("srcdoc");
  mapTrip.textContent = "";
  placeholder.hidden = false;
}

async function request(base, path, options) {
  const server = new URL(base);
  if (!["http:", "https:"].includes(server.protocol)) {
    throw new Error("Enter an HTTP or HTTPS API URL.");
  }
  const response = await fetch(new URL(path, server), {
    signal: AbortSignal.timeout(options ? 120000 : 10000),
    ...options,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === "string" ? body.detail : `API returned HTTP ${response.status}.`);
  }
  return response.json();
}

async function connect() {
  const current = ++revision;
  clearResult();
  trip.replaceChildren(new Option("Loading trips…", ""));
  run.disabled = trip.disabled = true;
  message("Connecting to the API…");
  try {
    const base = api.value.trim();
    const data = await request(base, "/trips");
    if (current !== revision) return;
    predictor = data.predictor;
    connectedApi = base;
    trip.replaceChildren(...data.trips.map(id => new Option(id, id)));
    run.disabled = trip.disabled = data.trips.length === 0;
    message(data.trips.length ? predictor : "The API returned no trips.");
    localStorage.setItem("pathsnap-api", base);
  } catch (error) {
    if (current !== revision) return;
    trip.replaceChildren(new Option("Trips unavailable", ""));
    document.querySelector("details").open = true;
    message(`Could not load trips. ${error.message}`, true);
  }
}

document.querySelector("#connection").addEventListener("submit", event => {
  event.preventDefault();
  connect();
});

trip.addEventListener("change", () => {
  clearResult();
  message(predictor);
});

document.querySelector("#prediction").addEventListener("submit", async event => {
  event.preventDefault();
  const current = ++revision;
  const id = trip.value;
  clearResult();
  run.disabled = trip.disabled = true;
  message(`Running prediction for ${id}…`);
  try {
    const data = await request(connectedApi, "/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ trip_id: id }),
    });
    if (current !== revision) return;
    const entries = [
      ["GPS observations", data.observations.toLocaleString()],
      ["Inference time", `${data.inference_ms.toLocaleString()} ms`],
      ...Object.entries(metricNames).map(([key, label]) => [
        label, data.metrics[key] == null ? "Unavailable" : percent.format(data.metrics[key]),
      ]),
    ];
    metrics.replaceChildren(...entries.map(([label, value]) => {
      const row = document.createElement("div");
      const term = document.createElement("dt");
      const description = document.createElement("dd");
      term.textContent = label;
      description.textContent = value;
      row.append(term, description);
      return row;
    }));
    results.hidden = false;
    if (data.map_html) {
      map.hidden = false;
      placeholder.hidden = true;
      // Give Leaflet a visible viewport before the iframe's map script initializes.
      map.getBoundingClientRect();
      map.srcdoc = data.map_html;
    }
    mapTrip.textContent = data.trip_id;
    message(data.predictor);
  } catch (error) {
    if (current === revision) message(`Prediction failed. ${error.message}`, true);
  } finally {
    if (current === revision) run.disabled = trip.disabled = false;
  }
});

connect();
