from django.urls import path
from .views import RetrieveImageView, UploadImageView, RetrieveResultView

urlpatterns = [
    path('fetch-image', RetrieveImageView.as_view()),
    path('upload', UploadImageView.as_view()),
    path('fetch-result', RetrieveResultView.as_view())
]