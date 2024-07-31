from django.shortcuts import render, redirect
from .forms import ImageForm
from .models import Image
from django.conf import settings
from django.http import HttpResponse
from inference import get_model
import supervision as sv
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

def upload_image(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            image_instance = form.save()
            image_path = os.path.join(settings.MEDIA_ROOT, str(image_instance.image))
            processed_image_path = run_inference(image_path)
            image_instance.processed_image = os.path.relpath(processed_image_path, settings.MEDIA_ROOT)
            image_instance.save()
            return render(request, 'result.html', {'image': image_instance})
    else:
        form = ImageForm()
    return render(request, 'upload.html', {'form': form})

def run_inference(image_path):
    api_key = "OzoNQLQJXshsqlU6vM3n"
    if not api_key:
        raise ValueError("API key not set. Please set the ROBOFLOW_API_KEY environment variable.")

    model = get_model(model_id="test_dks/2", api_key=api_key)

    image = cv2.imread(image_path)
    results = model.infer(image)[0]

    detections = sv.Detections.from_inference(results)

    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    processed_image_path = os.path.splitext(image_path)[0] + "_result.jpg"
    cv2.imwrite(processed_image_path, annotated_image)

    return processed_image_path
