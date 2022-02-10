from django.contrib import admin
from django.urls import path
from home import views

urlpatterns = [
   
    path('',views.index, name='home'),
    path('form/',views.form, name='form1'),
    path('form/', views.form, name='form2'),
    path('predict/',views.predict,name='predict'),
]
