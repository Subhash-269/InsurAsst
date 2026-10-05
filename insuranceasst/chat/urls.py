from django.urls import path
from . import views


urlpatterns = [
    path('', views.index, name='index'),
    path('health/', views.health, name='health'),
    path('api/chat/', views.chat_api, name='chat_api'),
    path('api/chat/stream/', views.chat_stream, name='chat_stream'),
    path('api/files/', views.list_files, name='list_files'),
    path('api/files/upload/', views.upload_file, name='upload_file'),
    path('api/reindex/', views.reindex, name='reindex'),
    path('api/vision/analyze/', views.vision_analyze, name='vision_analyze'),
    path('api/files/delete/', views.delete_files, name='delete_files'),
    path('api/vectors/clear/', views.clear_vectors, name='clear_vectors'),

]
