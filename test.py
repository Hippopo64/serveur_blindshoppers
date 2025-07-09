import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os

# --- Fonction de correction mathématique ---
def appliquer_correction_lineaire(distance_mesuree):
    """
    Corrige la distance en utilisant une fonction linéaire (y = mx + b)
    calibrée sur deux points connus.
    - Point 1 (x1, y1): (mesuré=1.0, réel=1.0)
    - Point 2 (x2, y2): (mesuré=4.6, réel=6.0)
    """
    # Calcul de la pente (m)
    m = (6.0 - 1.0) / (4.6 - 1.0)
    
    # Calcul de l'ordonnée à l'origine (b) en utilisant le point 1
    # y = mx + b  =>  b = y - mx
    b = 1.0 - m * 1.0
    
    # Application de la correction
    distance_corrigee = m * distance_mesuree + b
    
    # Sécurité pour éviter que la correction ne réduise la distance à courte portée
    return max(distance_mesuree, distance_corrigee)


# --- Configuration des chemins (recommandé) ---
script_dir = os.path.dirname(os.path.abspath(__file__))
# Assurez-vous que les chemins sont corrects pour votre configuration
weights_path = os.path.join(script_dir, "yolov3.weights")
config_path = os.path.join(script_dir, "yolov3.cfg")
names_path = os.path.join(script_dir, "coco.names")

# --- Initialisation du modèle YOLO ---
print("Chargement du modèle YOLO...")
net = cv2.dnn.readNet(weights_path, config_path)
classes = []
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
try:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
except AttributeError:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
print("Modèle chargé.")

# --- Initialisation de la caméra RealSense ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

print("Démarrage du pipeline de la caméra...")
profile = pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)
print("Pipeline démarré.")

# --- Boucle principale ---
last_print_time = 0
try:
    print("\nDébut de la détection. Appuyez sur Ctrl+C pour arrêter.")
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        current_time = time.time()
        if (current_time - last_print_time) < 0.5:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        height, width, channels = color_image.shape

        blob = cv2.dnn.blobFromImage(color_image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.6 and classes[class_id] == "person":
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        if len(boxes) > 0:
            min_distance_perçue = float('inf')
            
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            
            if isinstance(indexes, np.ndarray):
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    roi_x = max(0, int(x + w * 0.2))
                    roi_y = max(0, int(y + h * 0.2))
                    roi_w = int(w * 0.6)
                    roi_h = int(h * 0.6)
                    roi_x_end = min(width, roi_x + roi_w)
                    roi_y_end = min(height, roi_y + roi_h)

                    depth_roi = depth_image[roi_y:roi_y_end, roi_x:roi_x_end]
                    valid_depths = depth_roi[depth_roi > 0]
                    
                    if valid_depths.size > 0:
                        distance = np.median(valid_depths) / 1000.0
                        
                        if 0 < distance < min_distance_perçue:
                            min_distance_perçue = distance
            
            if min_distance_perçue != float('inf'):
                # ✅ APPLICATION DE LA CORRECTION
                distance_corrigee = appliquer_correction_lineaire(min_distance_perçue)
                
                # ✅ AFFICHAGE DES DEUX VALEURS
                print(f"Distance perçue: {min_distance_perçue:.2f}m -> Distance corrigée: {distance_corrigee:.2f}m")
                last_print_time = current_time

except KeyboardInterrupt:
    print("\nArrêt demandé par l'utilisateur.")
except Exception as e:
    print(f"Une erreur est survenue: {e}")
finally:
    print("Arrêt du pipeline de la caméra.")
    pipeline.stop()