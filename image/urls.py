from django.urls import path
from .views import RetrieveImageView, UploadImageView 

urlpatterns = [
    path('retrieve-images', RetrieveImageView.as_view()),
    path('upload', UploadImageView.as_view()),
]