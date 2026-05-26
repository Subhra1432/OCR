"""
Lightweight Frontend Proxy App for Render
This serves the static UI and proxies /upload and /api/* to Hugging Face Spaces.
"""
import os
from flask import Flask, render_template, request, Response
import requests

app = Flask(__name__)
HF_BACKEND_URL = os.environ.get("HF_BACKEND_URL", "https://subhra1432-ai-ocr-mining.hf.space").rstrip('/')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def proxy_upload():
    if "image" not in request.files:
        return {"error": "No image"}, 400
    
    file = request.files["image"]
    files = {"image": (file.filename, file.stream, file.mimetype)}
    
    # Forward form data
    data = {k: v for k, v in request.form.items()}
    
    # Forward to HF
    resp = requests.post(f"{HF_BACKEND_URL}/upload", files=files, data=data)
    
    # Return response as-is
    excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
    headers = [(name, value) for (name, value) in resp.raw.headers.items()
               if name.lower() not in excluded_headers]
    return Response(resp.content, resp.status_code, headers)

@app.route("/api/status", methods=["GET"])
def proxy_status():
    headers = {"X-User-API-Key": request.headers.get("X-User-API-Key", "")}
    resp = requests.get(f"{HF_BACKEND_URL}/api/status", headers=headers)
    
    excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
    return_headers = [(name, value) for (name, value) in resp.raw.headers.items()
               if name.lower() not in excluded_headers]
    return Response(resp.content, resp.status_code, return_headers)

@app.route("/api/translate", methods=["POST"])
def proxy_translate():
    headers = {"X-Groq-Api-Key": request.headers.get("X-Groq-Api-Key", ""), "Content-Type": "application/json"}
    resp = requests.post(f"{HF_BACKEND_URL}/api/translate", json=request.get_json(), headers=headers)
    
    excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
    return_headers = [(name, value) for (name, value) in resp.raw.headers.items()
               if name.lower() not in excluded_headers]
    return Response(resp.content, resp.status_code, return_headers)

if __name__ == "__main__":
    print("=" * 60)
    print("  AI Picture Text Mining — Lightweight Frontend Proxy")
    print("=" * 60)
    port = int(os.environ.get("PORT", 9090))
    print(f"  Open in your browser: http://localhost:{port}")
    print(f"  Proxying backend to: {HF_BACKEND_URL}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=False)
