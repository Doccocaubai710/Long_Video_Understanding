"""
server.py -- tiny standalone backend for the Video Compression Atlas static
site (no more Claude artifact runtime). Serves index.html, a JSON state
store at /api/state (GET returns it, POST overwrites it), static images
under /images/<name> (GET), and an image upload endpoint at /api/upload
(POST, base64 JSON body -> writes to images/, returns the URL to embed via
markdown ![alt](images/<name>) in an idea's body). No auth, by request --
anyone with the URL can read and write.
"""

import base64
import json
import mimetypes
import os
import re
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SITE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(SITE_DIR, "state.json")
INDEX_PATH = os.path.join(SITE_DIR, "index.html")
IMAGES_DIR = os.path.join(SITE_DIR, "images")
PORT = int(os.environ.get("PORT", 8090))
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20MB, du cho anh visualize PNG

os.makedirs(IMAGES_DIR, exist_ok=True)
state_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args))

    def _send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/state":
            with state_lock:
                with open(STATE_PATH, "r", encoding="utf-8") as f:
                    data = f.read()
            body = data.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/images/"):
            name = os.path.basename(self.path[len("/images/"):].split("?")[0])
            fpath = os.path.join(IMAGES_DIR, name)
            if not name or not os.path.isfile(fpath):
                self._send_json({"error": "not found"}, status=404)
                return
            ctype = mimetypes.guess_type(fpath)[0] or "application/octet-stream"
            with open(fpath, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "public, max-age=86400")
            self.end_headers()
            self.wfile.write(body)
            return
        # anything else -> serve index.html (single-page app)
        with open(INDEX_PATH, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path == "/api/upload":
            self._handle_upload()
            return
        if self.path != "/api/state":
            self._send_json({"error": "not found"}, status=404)
            return
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            obj = json.loads(raw)
        except Exception as e:
            self._send_json({"error": "invalid json: %s" % e}, status=400)
            return
        with state_lock:
            tmp_path = STATE_PATH + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, STATE_PATH)
        self._send_json({"ok": True})

    def _handle_upload(self):
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_UPLOAD_BYTES * 2:  # base64 ~1.37x raw size, chua bien do
            self._send_json({"error": "file too large"}, status=413)
            return
        raw = self.rfile.read(length)
        try:
            obj = json.loads(raw)
            filename = str(obj.get("filename") or "upload.png")
            data_url = str(obj.get("data") or "")
        except Exception as e:
            self._send_json({"error": "invalid json: %s" % e}, status=400)
            return
        # data co the la data-URL day du ("data:image/png;base64,...") hoac
        # chi phan base64 thuan -- xu ly ca 2 truong hop.
        if "," in data_url and data_url.strip().startswith("data:"):
            data_url = data_url.split(",", 1)[1]
        try:
            raw_bytes = base64.b64decode(data_url)
        except Exception as e:
            self._send_json({"error": "invalid base64: %s" % e}, status=400)
            return
        if len(raw_bytes) > MAX_UPLOAD_BYTES:
            self._send_json({"error": "file too large (max 20MB)"}, status=413)
            return
        ext = os.path.splitext(filename)[1].lower()
        if ext not in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"):
            ext = ".png"
        safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", os.path.splitext(filename)[0])[:60] or "image"
        out_name = f"{safe_name}_{uuid.uuid4().hex[:8]}{ext}"
        out_path = os.path.join(IMAGES_DIR, out_name)
        with open(out_path, "wb") as f:
            f.write(raw_bytes)
        self._send_json({"ok": True, "url": f"images/{out_name}"})


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print("Serving vlm-paper-site on http://0.0.0.0:%d" % PORT)
    server.serve_forever()
