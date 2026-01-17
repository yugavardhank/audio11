from django.urls import path
from .views import index, download_file

urlpatterns = [
    path("", index),
    path("download/<path:filepath>", download_file),
]