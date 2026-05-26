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
    # Read entire file into memory to avoid stream positioning issues across different environments
    file_content = file.read()
    files = {"image": (file.filename, file_content, file.mimetype)}
    
    # Forward form data and inject server-side API key if missing
    data = {k: v for k, v in request.form.items()}
    if not data.get("groq_api_key"):
        server_key = os.environ.get("GROQ_API_KEY", "")
        if server_key:
            data["groq_api_key"] = server_key
    
    # Forward to HF
    try:
        resp = requests.post(f"{HF_BACKEND_URL}/upload", files=files, data=data)
        
        # If the backend returned an error, make sure it's JSON so the frontend doesn't crash
        if resp.status_code >= 400:
            try:
                error_data = resp.json()
                return error_data, resp.status_code
            except:
                return {"error": f"Backend Error {resp.status_code}: {resp.text[:200]}"}, resp.status_code
                
        # Return success response as-is
        excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
        headers = [(name, value) for (name, value) in resp.raw.headers.items()
                   if name.lower() not in excluded_headers]
        return Response(resp.content, resp.status_code, headers)
    except Exception as e:
        import traceback
        return {"error": f"Proxy error: {str(e)}\n{traceback.format_exc()}"}, 500

@app.route("/api/status", methods=["GET"])
def proxy_status():
    # Inject server-side API key if client didn't provide one
    user_key = request.headers.get("X-User-API-Key", "")
    if not user_key:
        user_key = os.environ.get("GROQ_API_KEY", "")
        
    headers = {"X-User-API-Key": user_key}
    try:
        resp = requests.get(f"{HF_BACKEND_URL}/api/status", headers=headers)
        if resp.status_code >= 400:
            try:
                return resp.json(), resp.status_code
            except:
                return {"error": f"Backend Error {resp.status_code}: {resp.text[:200]}"}, resp.status_code
                
        excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
        return_headers = [(name, value) for (name, value) in resp.raw.headers.items()
                   if name.lower() not in excluded_headers]
        return Response(resp.content, resp.status_code, return_headers)
    except Exception as e:
        import traceback
        return {"error": f"Proxy error: {str(e)}\n{traceback.format_exc()}"}, 500

@app.route("/api/translate", methods=["POST"])
def proxy_translate():
    # Inject server-side API key if client didn't provide one
    user_key = request.headers.get("X-Groq-Api-Key", "")
    if not user_key:
        user_key = os.environ.get("GROQ_API_KEY", "")
        
    headers = {"X-Groq-Api-Key": user_key, "Content-Type": "application/json"}
    try:
        resp = requests.post(f"{HF_BACKEND_URL}/api/translate", json=request.get_json(), headers=headers)
        if resp.status_code >= 400:
            try:
                return resp.json(), resp.status_code
            except:
                return {"error": f"Backend Error {resp.status_code}: {resp.text[:200]}"}, resp.status_code
                
        excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
        return_headers = [(name, value) for (name, value) in resp.raw.headers.items()
                   if name.lower() not in excluded_headers]
        return Response(resp.content, resp.status_code, return_headers)
    except Exception as e:
        import traceback
        return {"error": f"Proxy error: {str(e)}\n{traceback.format_exc()}"}, 500

if __name__ == "__main__":
    print("=" * 60)
    print("  AI Picture Text Mining — Lightweight Frontend Proxy")
    print("=" * 60)
    port = int(os.environ.get("PORT", 9090))
    print(f"  Open in your browser: http://localhost:{port}")
    print(f"  Proxying backend to: {HF_BACKEND_URL}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=False)
