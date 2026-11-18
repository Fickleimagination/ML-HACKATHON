"""Run every discovered trip through the live API and optionally save a metrics CSV."""
import argparse
import csv
import json
from pathlib import Path
from urllib.request import Request, urlopen


def fetch(url, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    request = Request(url.rstrip("/") + path, data, {"Content-Type": "application/json"})
    with urlopen(request, timeout=120) as response:
        return json.load(response)


def compare_expected(trip, result, expected):
    for field in ("arcs", "route", "candidate_arcs", "confidence", "diagnostics"):
        if result[field] == expected.get(field):
            continue
        actual, wanted = result[field], expected.get(field)
        if isinstance(actual, list) and isinstance(wanted, list):
            index = next((i for i, (a, b) in enumerate(zip(actual, wanted)) if a != b), None)
            detail = (
                f"first mismatch at index {index}: got {actual[index]!r}, expected {wanted[index]!r}"
                if index is not None else f"length {len(actual)}, expected {len(wanted)}"
            )
        else:
            detail = f"got {actual!r}, expected {wanted!r}"
        raise ValueError(f"{trip} {field}: {detail}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--count", type=int, default=10)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--csv", type=Path)
    output.add_argument("--readiness-only", action="store_true",
                        help="Verify responses without printing or saving scores")
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument("--expected-dir", type=Path, help="Compare outputs with per-trip JSON")
    reference.add_argument("--direct-data", type=Path,
                           help="Compare API outputs with a direct predictor run on local trip files")
    args = parser.parse_args()

    listing = fetch(args.api, "/trips")
    trips = listing["trips"]
    if len(trips) != args.count or len(set(trips)) != len(trips):
        raise ValueError(f"Expected {args.count} unique trips, received {trips}")
    if args.direct_data:
        from api.predictor import load_predictor
        from api.trips import discover_trips, load_trip

        direct_predictor, direct_name = load_predictor()
        if direct_name != listing["predictor"]:
            raise ValueError(f"Local predictor is {direct_name}; API uses {listing['predictor']}")
        direct_trips = discover_trips(args.direct_data)
        if set(trips) != direct_trips.keys():
            raise ValueError("The local data and API have different trip IDs")
    if args.expected_dir or args.direct_data:
        priority = {"00000007": 0, "00000065": 1, "00000056": 2}
        trips = sorted(trips, key=lambda trip: priority.get(trip, len(priority)))
    rows = []
    for trip in trips:
        result = fetch(args.api, "/predict", {"trip_id": trip})
        if result["trip_id"] != trip or result["predictor"] != listing["predictor"]:
            raise ValueError(f"Unexpected prediction metadata for {trip}")
        if len(result["arcs"]) != result["observations"] or not result["route"]:
            raise ValueError(f"Incomplete prediction for {trip}")
        if result.get("candidate_arcs") is not None and len(result["candidate_arcs"]) != result["observations"]:
            raise ValueError(f"Candidate lists do not cover every observation for {trip}")
        if {"reference_route", "reference_arcs", "reference_labels", "labels"} & result.keys():
            raise ValueError(f"Reference data exposed in response for {trip}")
        html = result["map_html"]
        if "Predicted route" not in html or "GPS observations" not in html or "Reference route" in html:
            raise ValueError(f"Unexpected map layers for {trip}")
        if args.expected_dir:
            compare_expected(trip, result, json.loads((args.expected_dir / f"{trip}.json").read_text()))
        if args.direct_data:
            expected = direct_predictor(*load_trip(direct_trips[trip]))
            serializable = json.loads(json.dumps(expected, default=lambda value: value.tolist()))
            compare_expected(trip, result, serializable)
        metrics = result["metrics"]
        for name in ("top1_accuracy", "route_precision", "route_recall"):
            if metrics[name] is None or not 0 <= metrics[name] <= 1:
                raise ValueError(f"Missing or invalid {name} for {trip}")
        if args.readiness_only:
            print(f"{trip}  HTTP 200  route, map and scoring verified", flush=True)
            continue
        rows.append({
            "trip_id": trip, "predictor": result["predictor"],
            "observations": result["observations"], "inference_ms": result["inference_ms"],
            **metrics,
        })
        print(f'{trip}  HTTP 200  {result["observations"]:>5} obs  '
              f'Top-1 {metrics["top1_accuracy"]:.1%}  '
              f'{result["inference_ms"]:.1f} ms', flush=True)

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    print(f"Verified {len(trips)}/{len(trips)} trips via {args.api}")


if __name__ == "__main__":
    main()
