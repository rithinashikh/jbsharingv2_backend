from django.contrib import admin
from django.urls import path
from .views import MLModel

urlpatterns = [
    path('ml-model/<input>', MLModel.as_view(), name='MLModel'),
]