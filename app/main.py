import os
from pathlib import Path

from flask import Flask, send_file, send_from_directory
from flask_cors import CORS

from api.routes import api_bp

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = Flask(__name__, static_folder=None)
CORS(app)

app.register_blueprint(api_bp, url_prefix="/api/v1")


@app.route("/")
def root():
    return send_file(STATIC_DIR / "index.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)


@app.route("/favicon.ico")
def favicon():
    return "", 204


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
