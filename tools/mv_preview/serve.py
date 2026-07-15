import glob
import http.server
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)


class H(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Model manifest for view.html. index.html shadows the directory listing, so the
        # .glb files are enumerated here instead; built per-request so a model that finishes
        # rendering shows up on refresh.
        if self.path.split("?")[0] == "/models.json":
            names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(ROOT, "*.glb")))
            body = json.dumps(names).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        return super().do_GET()

    def do_PUT(self):
        n = int(self.headers["Content-Length"])
        name = os.path.basename(self.path)
        with open(name, "wb") as f:
            f.write(self.rfile.read(n))
        self.send_response(200)
        self.end_headers()

    def log_message(self, *a):
        pass


port = int(sys.argv[1]) if len(sys.argv) > 1 else 8177
print(f"serving {ROOT} on http://127.0.0.1:{port}  ->  /view.html", flush=True)
http.server.ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
