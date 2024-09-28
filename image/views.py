from django.shortcuts import render
from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView
from .serializers import ImageSerializer

# Create your views here.

class RetrieveImageView(APIView):
    def get(self, request, format=None):
        try:
            if Image.objects.all().exists():
                images = Image.objects.all()
                images = ImageSerializer(images, many=True)

                return Response(
                    {'images':images.data},
                    status=status.HTTP_200_OK
                )
            else:
                return Response (
                    {'error': 'Something went wrong when retrieving images'},
                    status=status.HTTP_400_BAD_REQUEST
                )
        except:
            return Response (
                {'error': 'Something went wrong when retrieving images'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class UploadImageView(APIView):
    def post(self, request):
        try:
            data = self.request.data

            image = data['image']
            alt_text = data['alt_text']

            Image.objects.create(
                image=image,
                alt_text=alt_text
            )

            return Response(
                {'success':"Image Uploaded successfully"},
                stats = 201
            )
        except:
            return Response (
                {'error': 'Something went wrong when uploading images'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )