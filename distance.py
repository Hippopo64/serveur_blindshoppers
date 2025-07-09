import pyrealsense2 as rs
import numpy as np
import cv2

# Charger le modèle de détection d'objets (YOLOv3)
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Configurer le pipeline RealSense
pipeline = rs.pipeline()
config = rs.config()

# Obtenir les informations sur le périphérique et les flux
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

# Vérifier si un type de caméra a été trouvé
found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("La démo nécessite une caméra de profondeur et une caméra couleur.")
    exit(0)

# Activer les flux de la caméra
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Démarrer le streaming
profile = pipeline.start(config)

# Obtenir l'échelle de profondeur
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Échelle de profondeur : ", depth_scale)

# Aligner le flux de profondeur sur le flux de couleur
align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:
        # Attendre une paire de trames cohérente : profondeur et couleur
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convertir les images en tableaux numpy
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        height, width, channels = color_image.shape

        # Détection d'objets
        blob = cv2.dnn.blobFromImage(color_image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Afficher les informations à l'écran
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and classes[class_id] == "person":
                    # Objet détecté
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Coordonnées du rectangle
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(color_image, (x, y), (x + w, y + h), color, 2)
                
                # Calculer le centre de la boîte de détection
                center_x = x + w // 2
                center_y = y + h // 2

                # S'assurer que le centre est dans les limites de l'image
                if 0 <= center_x < width and 0 <= center_y < height:
                    # Obtenir la distance de la personne
                    distance = depth_frame.get_distance(center_x, center_y)
                    if distance > 0:
                        distance_text = f"{label}: {distance:.2f}m"
                        cv2.putText(color_image, distance_text, (x, y + 30), font, 2, color, 3)

        # Afficher l'image résultante
        cv2.imshow("Image", color_image)
        
        # Quitter avec la touche 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Arrêter le streaming
    pipeline.stop()
    cv2.destroyAllWindows()