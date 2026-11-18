"""Serve the evaluator and relay its requests to the API laptop."""
import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import ProxyHandler, Request, build_opener


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", required=True, help="API laptop URL, such as http://192.168.1.20:8000")
    parser.add_argument("--port", type=int, default=8081, help="Local UI port (default: 8081)")
    args = parser.parse_args()
    upstream = args.api.rstrip("/")
    url = urlsplit(upstream)
    if (url.scheme not in ("http", "https") or not url.hostname or url.username is not None
            or url.path or url.query or url.fragment):
        parser.error("--api must be the base URL without /trips or another path")
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(upstream + "/trips", timeout=5) as response:
            trips = json.load(response)["trips"]
            if not isinstance(trips, list) or not trips:
                raise ValueError("the API returned no trips")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Cannot load {upstream}/trips from this laptop: {error}\n")

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=Path(__file__).resolve().parent, **kwargs)

        def do_GET(self):
            if self.path == "/trips":
                self.relay()
            else:
                super().do_GET()

        def do_POST(self):
            if self.path != "/predict":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.send_error(400)
                return
            if not 0 < length <= 65536:
                self.send_error(400)
                return
            self.relay(self.rfile.read(length))

        def relay(self, data=None):
            request = Request(upstream + self.path, data=data,
                              headers={"Content-Type": "application/json"})
            try:
                response = opener.open(request, timeout=120)
            except HTTPError as error:
                response = error
            except (URLError, TimeoutError) as error:
                self.send_payload(502, json.dumps({"detail": f"API connection failed: {error}"}).encode())
                return
            with response:
                self.send_payload(response.status, response.read())

        def send_payload(self, status, payload):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        local = f"http://127.0.0.1:{server.server_port}"
        print(f"Connected to {upstream}: {len(trips)} trips.\n"
              f"Open {local}/?{urlencode({'api': local})}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
