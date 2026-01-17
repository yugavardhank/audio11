from django.shortcuts import render
from django.http import FileResponse
from django.conf import settings
from pipeline.orchestrator import run_pipeline
import os

def index(request):
    if request.method == "POST":
        audio = request.FILES["audio"]
        
        # Create media directories if they don't exist
        media_input_dir = "media/input"
        os.makedirs(media_input_dir, exist_ok=True)
        
        path = f"{media_input_dir}/{audio.name}"

        with open(path, "wb+") as f:
            for chunk in audio.chunks():
                f.write(chunk)

        result = run_pipeline(path)
        
        # Convert file paths to relative URLs for template
        if result.get('export_formats'):
            for fmt, file_path in result['export_formats'].items():
                if file_path and os.path.exists(file_path):
                    # Make path relative to BASE_DIR for template, already has media/ prefix
                    rel_path = os.path.relpath(file_path, settings.BASE_DIR)
                    # Ensure we don't double the /media/ prefix
                    url_path = rel_path.replace(os.sep, '/')
                    if not url_path.startswith('media/'):
                        url_path = f"media/{url_path}"
                    result['export_formats'][fmt] = f"/{url_path}"

        return render(request, "result.html", result)

    return render(request, "upload.html")


def download_file(request, filepath):
    """Serve transcript files for download."""
    try:
        # Sanitize filepath to prevent directory traversal
        safe_path = os.path.abspath(filepath)
        base_dir = os.path.abspath(settings.BASE_DIR)
        
        if not safe_path.startswith(base_dir):
            return FileResponse(status=403)
        
        if os.path.exists(safe_path):
            response = FileResponse(open(safe_path, 'rb'))
            response['Content-Disposition'] = f'attachment; filename="{os.path.basename(safe_path)}"'
            return response
        else:
            return FileResponse(status=404)
    except Exception as e:
        print(f"Download error: {e}")
        return FileResponse(status=500)