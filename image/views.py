from django.shortcuts import render
from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView
from .serializers import ImageSerializer
from .models import ImageM
import base64
from PIL import Image

# Create your views here.

class RetrieveImageView(APIView):
    def get(self, request, format=None):
        image = base64.b64encode(open("test.png", "rb").read())

        content_type = "image/png"
    
        resp = Response(image, content_type=content_type)
        resp["Cache-Control"] = "no-cache"


        return resp

class UploadImageView(APIView):
    def post(self, request):
        try:
            data = self.request.data
            print(data)

            image = data['image']
            print(image)

            ImageM.objects.create(
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