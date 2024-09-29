from django.shortcuts import render
from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView
from .serializers import ImageSerializer
from .models import Image

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
            print(data)

            image = data['image']
            print(image)

            Image.objects.create(
                image = image
            )


            return Response(
                {'success':"Image Uploaded successfully"},
                status = 201
            )
        except Exception as err:
            print(err)
            return Response (
                {'error': 'Something went wrong when uploading images'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )