"""
Lightweight Frontend Proxy App for Render
This serves the static UI and proxies /upload and /api/* to Hugging Face Spaces.
"""
import os
from flask import Flask, render_template, request, Response
import requests

app = Flask(__name__)
HF_BACKEND_URL = os.environ.get("HF_BACKEND_URL", "https://subhra1432-ai-ocr-mining.hf.space").rstrip('/')

def _is_json_response(resp):
    content_type = resp.headers.get("Content-Type", "").lower()
    return "application/json" in content_type

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
        
        # If the backend is waking up or building, it might return a 200/503 HTML wrapper page
        if not _is_json_response(resp):
            return {
                "error": "The backend processing space is currently waking up or building. Please wait a moment and try again."
            }, 503
            
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
        
        if not _is_json_response(resp):
            return {
                "error": "The backend processing space is currently waking up or building. Please wait a moment and try again."
            }, 503

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
        
        if not _is_json_response(resp):
            return {
                "error": "The backend processing space is currently waking up or building. Please wait a moment and try again."
            }, 503

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

@app.route("/export/csv", methods=["GET"])
def proxy_export_csv():
    try:
        resp = requests.get(f"{HF_BACKEND_URL}/export/csv")
        if not resp.ok:
            return {"error": f"Failed to export CSV: {resp.text[:200]}"}, resp.status_code
        return Response(resp.content, resp.status_code, [
            ("Content-Type", resp.headers.get("Content-Type", "text/csv")),
            ("Content-Disposition", resp.headers.get("Content-Disposition", "attachment; filename=ocr_results.csv"))
        ])
    except Exception as e:
        return {"error": f"Proxy error: {str(e)}"}, 500

@app.route("/export/charts", methods=["GET"])
def proxy_export_charts():
    try:
        resp = requests.get(f"{HF_BACKEND_URL}/export/charts")
        if not resp.ok:
            return {"error": f"Failed to export charts: {resp.text[:200]}"}, resp.status_code
        return Response(resp.content, resp.status_code, [
            ("Content-Type", resp.headers.get("Content-Type", "application/zip")),
            ("Content-Disposition", resp.headers.get("Content-Disposition", "attachment; filename=ocr_charts.zip"))
        ])
    except Exception as e:
        return {"error": f"Proxy error: {str(e)}"}, 500

if __name__ == "__main__":
    print("=" * 60)
    print("  AI Picture Text Mining — Lightweight Frontend Proxy")
    print("=" * 60)
    port = int(os.environ.get("PORT", 9090))
    print(f"  Open in your browser: http://localhost:{port}")
    print(f"  Proxying backend to: {HF_BACKEND_URL}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=False)
