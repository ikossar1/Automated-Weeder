"""
Pick the four corners in a browser, when there is no screen to click on.

Over SSH there is no display, so the OpenCV window cannot open - it
fails with "could not connect to display". This serves the photo as a
web page from the Pi instead. You open it from whatever machine you
are sitting at, click the four corners there, and the coordinates come
back here.

    python3 pick_corners_web.py photos/bed_1.jpg --out corners.json

It prints a URL. In VS Code's Remote-SSH the port is forwarded for you,
so http://localhost:8000 just works; otherwise use the Pi's own address.

Nothing is installed for this - it is the standard library and a page
of plain JavaScript.
"""
from __future__ import annotations

import argparse
import http.server
import json
import pathlib
import socket
import sys
import threading

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>Pick the corners</title>
<style>
  body { margin: 0; background: #111; color: #eee;
         font: 15px system-ui, sans-serif; }
  header { padding: 12px 16px; background: #1b1b1b; position: sticky; top: 0; }
  #wrap { position: relative; display: inline-block; margin: 16px; }
  img { max-width: calc(100vw - 32px); display: block; }
  .dot { position: absolute; width: 18px; height: 18px; margin: -9px 0 0 -9px;
         border: 2px solid #fff; border-radius: 50%; background: #d6782a;
         font: bold 11px system-ui; color: #fff; text-align: center;
         line-height: 18px; pointer-events: none; }
  b { color: #ffd479; }
  button { font: inherit; padding: 6px 14px; margin-left: 8px; }
</style>
<header>
  <span id="msg">Click <b>near-left</b></span>
  <button onclick="reset()">Start over</button>
</header>
<div id="wrap"><img id="pic" src="photo"></div>
<script>
const ORDER = ["near-left", "near-right", "far-right", "far-left"];
let pts = [];
const pic = document.getElementById("pic");
const wrap = document.getElementById("wrap");
const msg = document.getElementById("msg");

function reset() {
  pts = [];
  document.querySelectorAll(".dot").forEach(d => d.remove());
  msg.innerHTML = "Click <b>" + ORDER[0] + "</b>";
}

pic.addEventListener("click", ev => {
  if (pts.length >= 4) return;
  const r = pic.getBoundingClientRect();
  const x = ev.clientX - r.left, y = ev.clientY - r.top;

  const dot = document.createElement("div");
  dot.className = "dot";
  dot.style.left = x + "px";
  dot.style.top = y + "px";
  dot.textContent = pts.length + 1;
  wrap.appendChild(dot);

  // Back to pixels in the original photo, not the scaled-down view.
  const sx = pic.naturalWidth / r.width, sy = pic.naturalHeight / r.height;
  pts.push([Math.round(x * sx), Math.round(y * sy)]);

  if (pts.length < 4) {
    msg.innerHTML = "Click <b>" + ORDER[pts.length] + "</b>";
    return;
  }
  msg.textContent = "Sending...";
  fetch("corners", {method: "POST", body: JSON.stringify(pts)})
    .then(() => { msg.textContent = "Done. Go back to the terminal."; })
    .catch(e => { msg.textContent = "Could not send: " + e; });
});
</script>
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description="Pick corners in a browser.")
    ap.add_argument("photo")
    # new_bed.py passes the bed size as well. The window version draws
    # it on the overlay; here it is not needed, but the call has to be
    # accepted unchanged so one caller can use either picker.
    ap.add_argument("size", nargs="*", help=argparse.SUPPRESS)
    ap.add_argument("--out", default="corners.json")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)

    photo = pathlib.Path(args.photo)
    if not photo.exists():
        sys.exit(f"No photo at {photo}")
    blob = photo.read_bytes()
    kind = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
            ".webp": "image/webp"}.get(photo.suffix.lower(), "image/jpeg")

    picked = {}
    done = threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/").endswith("photo"):
                self._send(200, blob, kind)
            else:
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            try:
                pts = json.loads(self.rfile.read(n))
            except ValueError:
                self._send(400, b"bad json", "text/plain")
                return
            picked["pts"] = [[int(x), int(y)] for x, y in pts]
            self._send(200, b"ok", "text/plain")
            done.set()

        def log_message(self, *a):
            pass

    server = http.server.ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    host = socket.gethostbyname(socket.gethostname())
    print("")
    print(f"  {photo.name}  -  open this and click the four corners:")
    print("")
    print(f"      http://localhost:{args.port}        (VS Code forwards this)")
    print(f"      http://{host}:{args.port}     (same network)")
    print("")
    print("  Order: near-left, near-right, far-right, far-left.")
    print("  Waiting... Ctrl+C to give up.")

    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        done.wait()
    except KeyboardInterrupt:
        print("\n  Stopped. No corners picked.")
        return 1
    finally:
        server.shutdown()

    flat = [v for pt in picked["pts"] for v in pt]
    pathlib.Path(args.out).write_text(json.dumps({"corners": flat}))
    print("")
    print("  Corners: " + " ".join(str(v) for v in flat))
    return 0


if __name__ == "__main__":
    sys.exit(main())
