from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import requests
#import cv2.dnn
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "aquivanunaspalabrotas"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

RestURL = os.getenv('REST_URL')
infer_url = f'{RestURL}/v2/models/accident-detection-model/infer'

CLASSES = {
    0: "moderado",
    1: "severo"
}
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def preprocess(image_path):
    original_image: np.ndarray = cv2.imread(image_path)
    [height, width, _] = original_image.shape
 
    # Calculo de escalación
    scale = (height/640, width/640)
 
    # Preprocesamiento de imagen y preperacion para envio a modelo
    blob = cv2.dnn.blobFromImage(original_image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
 
    return blob, scale, original_image


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    """
    Draws bounding boxes on the input image based on the provided arguments.

    Args:
        img (numpy.ndarray): The input image to draw the bounding box on.
        class_id (int): Class ID of the detected object.
        confidence (float): Confidence score of the detected object.
        x (int): X-coordinate of the top-left corner of the bounding box.
        y (int): Y-coordinate of the top-left corner of the bounding box.
        x_plus_w (int): X-coordinate of the bottom-right corner of the bounding box.
        y_plus_h (int): Y-coordinate of the bottom-right corner of the bounding box.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX,
    text_color_bg = colors[class_id]
    label = f'{CLASSES[class_id]} {confidence:.2f}'
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), text_color_bg, 2) # Box
    cv2.rectangle(img, (x, y-label_height), (x+label_width, y), text_color_bg, cv2.FILLED)  # Background label
    cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

def postprocess(response, scale, original_image):
    outputs = np.array([cv2.transpose(response[0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []

    # Iteracion en la salida para obtener las predicciones
    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        if maxScore >= 0.25:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2], outputs[0][i][3]]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)
    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

    detections = []

    # Itereacion en la salida para obtener las cajas
    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]
        detection = {
            'class_id': class_ids[index],
            'class_name': CLASSES[class_ids[index]],
            'confidence': scores[index],
            'box': box,
            'scale': scale}
        detections.append(detection)
        draw_bounding_box(original_image, class_ids[index], scores[index], round(box[0] * scale[1]), round(box[1] * scale[0]),                           
                            round((box[0] + box[2]) * scale[1]), round((box[1] + box[3]) * scale[0]))
    return original_image


def _serialize(image):
    payload = {
        'inputs': [
            {
                'name': 'images',
                'shape': [1, 3, 640, 640],
                'datatype': 'FP32',
                'data': image.flatten().tolist(),
            },
        ]
    }
    return payload


def _unpack(response_item):
    return np.array(response_item['data']).reshape(response_item['shape'])


def send_request(image, endpoint):
    payload = _serialize(image)
    raw_response = requests.post(endpoint, json = payload)
    try:
        response = raw_response.json()
    except:
        print(f'Failed to deserialize service response.\n'
              f'Status code: {raw_response.status_code}\n'
              f'Response body: {raw_response.text}')
        raise

    try:
        model_output = response['outputs']
    except:
        print(f'Failed to extract model output from service response.\n'
              f'Service response: {response}')
        raise

    unpacked_output = [_unpack(item) for item in model_output]
    return unpacked_output


def process_image(image_path, endpoint):
    preprocessed, scale, original_image = preprocess(image_path)
    response = send_request(preprocessed, endpoint)
    new_image = postprocess(response[0], scale, original_image)

    return new_image

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No se ha seleccionado imagen para cargar')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img = process_image(os.path.join(app.config['UPLOAD_FOLDER'], filename), infer_url)
        filename_infered = "infered" + filename
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename_infered), img)
        flash('La imagen se ha cargado correctamente')
        return render_template('index.html', filename=filename_infered)
    else:
        flash('Los tipos de imagenes permitidos son - png, jpg, jpeg')
        return redirect(request.url)
 
@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
 
if __name__ == "__main__":
    app.run()
