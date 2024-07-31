from django.db import models

class Image(models.Model):
    image = models.ImageField(upload_to='images/')
    processed_image = models.ImageField(upload_to='processed_images/', blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

