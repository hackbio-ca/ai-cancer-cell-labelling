from rest_framework import serializers
from .models import ImageM

class ImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImageM
        fields = '__all__'
