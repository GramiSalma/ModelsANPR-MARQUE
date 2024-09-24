from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
import base64
from flask import send_file
from ultralytics import YOLO 

app = Flask(__name__)

# Charger les modèles YOLO et OCR
def load_yolo():
    net = cv2.dnn.readNet("models/yolov3_last.weights", 
                          "models/yolov3.cfg")
    classes = []
    with open("models/classes.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    layers_names = net.getLayerNames()
    unconnected_out_layers = net.getUnconnectedOutLayers().flatten()
    output_layers = [layers_names[i - 1] for i in unconnected_out_layers]
    return net, classes, output_layers

def load_yolo_ocr():
    net = cv2.dnn.readNet("models/yolov3_last_characters.weights", 
                          "models/yolov3_characters.cfg")
    classes = []
    with open("models/classes_characters.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layers_names = net.getLayerNames()
    unconnected_out_layers = net.getUnconnectedOutLayers().flatten()
    output_layers = [layers_names[i - 1] for i in unconnected_out_layers]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers

# Détection et extraction des plaques
def detect_objects(img, net, output_layers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    return blob, outputs

def get_box_dimensions(outputs, height, width, threshold):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > threshold:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids

def draw_labels(boxes, confs, class_ids, classes, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.1, 0.1)
    font = cv2.FONT_HERSHEY_PLAIN
    plats = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color_green = (0, 255, 0)
            color_red = (0, 0, 255)
            crop_img = img.copy()[y:y + h, x:x + w]
            cv2.rectangle(img, (x, y), (x + w, y + h), color_green, 2)
            confidence = round(confs[i], 3) * 100
            cv2.putText(img, str(confidence) + '%', (x, y - 5), font, 3.25, color_red, 2)
            plats.append(crop_img)
    return img, plats

# Reconnaissance du contenu des plaques
def detect_characters(img, net, output_layers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    return blob, outputs

def get_characters_dimensions(outputs, height, width, threshold):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > threshold:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids

def draw_characters(boxes, confs, colors, class_ids, classes, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.4, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    caracters = []
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i % len(colors)]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        confidence = round(confs[i], 3) * 100
        cv2.putText(img, str(confidence) + '%', (x, y - 5), font, 1, color, 2)
        caracters.append((label, x))
    caracters.sort(key=lambda x: x[1])
    plat = "".join([label for label, x in caracters])
    formatted_plat = format_plate(plat)
    return img, formatted_plat

def format_plate(plate_text):
    if len(plate_text) < 7:
        return plate_text
    part1 = plate_text[:-3]
    part2 = plate_text[-3]
    part3 = plate_text[-2:]
    return f"{part1}{part2}{part3}"

def split_plate(plate_text):
    num1 = ""
    letters = ""
    num2 = ""

    found_letter = False

    for char in plate_text:
        if char.isdigit() and not found_letter:
            num1 += char
        elif char.isalpha():
            found_letter = True
            letters += char
        elif char.isdigit() and found_letter:
            num2 += char

    return num1, letters, num2

def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

# Routes Flask
@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json()
    if not data or 'filePath' not in data:
        return jsonify({"error": "Invalid request data"}), 400

    file_path = data['filePath']
    print(f"Received file path: {file_path}")  # Debug: print the file path
    file_path = file_path.replace("\\", "/")  # Normalize the file path
    img = cv2.imread(file_path)
    if img is None:
        print(f"Error: Image not found at path {file_path}")  # Debug: print an error message
        return jsonify({"error": "Image not found"}), 400

    height, width, _ = img.shape
    model, classes, output_layers = load_yolo()
    blob, outputs = detect_objects(img, model, output_layers)
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width, 0.3)
    plat_detected, crop_plats = draw_labels(boxes, confs, class_ids, classes, img)
    if len(crop_plats) > 0:
        crop_img = crop_plats[0]
        ocr_model, ocr_classes, colors, ocr_output_layers = load_yolo_ocr()
        blob, outputs = detect_characters(crop_img, ocr_model, ocr_output_layers)
        height, width, _ = crop_img.shape
        boxes, confs, class_ids = get_characters_dimensions(outputs, height, width, 0.1)
        lp_image, lp_text = draw_characters(boxes, confs, colors, class_ids, ocr_classes, crop_img)
        encoded_lp_image = encode_image_to_base64(lp_image)
        num1, letters, num2 = split_plate(lp_text)
        return jsonify({
            "plate_text": lp_text,
            "plate_image": encoded_lp_image,
            "order_number": num1,
            "registration_series": letters,
            "prefecture_number": num2
        })
    else:
        return jsonify({"error": "No plate detected"}), 400
    
def detect_brands(image_path):
    model = YOLO("best.pt")
    results = model.predict(source=image_path)
    
    # Supposons que la marque soit le label de la première détection
    brand_name = results[0].names[int(results[0].boxes.cls[0])] if results[0].boxes else "Unknown"
    return brand_name

@app.route('/detect_brand', methods=['POST'])

def detect_brand():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    temp_image_path = 'temp_image.jpg'
    file.save(temp_image_path)

    # Détection des marques
    brand_name = detect_brands(temp_image_path)

    # Supprimer l'image temporaire
    os.remove(temp_image_path)

    # Retourner le nom de la marque détectée
    return jsonify({"brand": brand_name})

if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)
