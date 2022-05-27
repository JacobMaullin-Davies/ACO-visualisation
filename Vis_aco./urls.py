"""Vis_aco URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from main import views as main_view
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('admin/', admin.site.urls), #admin
    path('', main_view.index, name='index'), # index page
    path('visual/', main_view.visualisation, name='visual'), # visual page
    path("locations_send/", main_view.locations), # location data send
    path("api_pointUpdate/", main_view.points_api), # update request for data for each iteration
    path("path_start/", main_view.vis_logic), # start run request
    path("path_finish/", main_view.vis_finish), # stop run request
    path("reset_path", main_view.reset_path_run), # reset data request

]

urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
