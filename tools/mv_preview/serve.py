import http.server
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


class H(http.server.SimpleHTTPRequestHandler):
    def do_PUT(self):
        n = int(self.headers["Content-Length"])
        name = os.path.basename(self.path)
        with open(name, "wb") as f:
            f.write(self.rfile.read(n))
        self.send_response(200)
        self.end_headers()

    def log_message(self, *a):
        pass


http.server.ThreadingHTTPServer(("127.0.0.1", 8177), H).serve_forever()
