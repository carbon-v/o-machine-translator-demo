#!/usr/bin/env python3
"""
Tiny Interactive Web UI Server for HDC Interlingua Showcase
===========================================================
Zero-dependency HTTP server serving interactive dark-mode UI
and sub-millisecond translation API.

Usage:
    python app.py [port]
"""

import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from translator import HDCPolysemyTranslator

translator = HDCPolysemyTranslator(dim=10000)


class HDCShowcaseHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            html_path = os.path.join(os.path.dirname(__file__), "index.html")
            if os.path.exists(html_path):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                with open(html_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "index.html not found")
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/translate":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(body)
                sentence = data.get("sentence", "")
                result = translator.translate(sentence)
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        # Keep terminal log clean
        pass


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8088
    print("====================================================================")
    print(f"   O-MACHINE HDC SHOWCASE UI RUNNING AT: http://localhost:{port}")
    print("====================================================================")
    print("Press Ctrl+C to stop.")
    server = HTTPServer(("localhost", port), HDCShowcaseHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down showcase UI server.")
        server.server_close()


if __name__ == "__main__":
    main()
