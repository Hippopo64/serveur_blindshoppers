import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os

# ==============================================================================
# ======================== CONFIGURATION DU MAGASIN ============================
# ==============================================================================

# Distance RÉELLE en mètres entre la caméra et les points de référence sur le plan.
DISTANCE_REELLE_DEBUT_ALLEE_M = 0.65  # Distance réelle pour y=1
DISTANCE_REELLE_FIN_ALLEE_M   = 6.1 # Distance réelle pour y=14

# Coordonnées sur le plan
CAMERA_X_POS = 2
ALLEE_Y_DEBUT = 1
ALLEE_Y_FIN = 14

# ==============================================================================
# ======================== FONCTIONS DE CALCUL =================================
# ==============================================================================

def appliquer_correction_lineaire(distance_mesuree):
    """
    Corrige la distance mesurée en utilisant une fonction linéaire (y = mx + b)
    calibrée sur deux points connus.
    - Point 1 (x1, y1): (mesuré=1.0, réel=1.0)
    - Point 2 (x2, y2): (mesuré=4.6, réel=6.0)
    """
    # Votre calcul, qui est parfaitement correct :
    m = (6.0 - 1.0) / (4.6 - 1.0)  # Pente
    b = 1.0 - m * 1.0              # Ordonnée à l'origine
    
    distance_corrigee = m * distance_mesuree + b
    
    # ✅ LA CORRECTION : On retourne TOUJOURS le résultat du calcul.
    return distance_corrigee

def calculer_position_sur_grille(distance_corrigee_m):
    """Calcule la coordonnée y sur la grille à partir de la distance RÉELLE corrigée."""
    if DISTANCE_REELLE_FIN_ALLEE_M <= DISTANCE_REELLE_DEBUT_ALLEE_M:
        return None

    proportion = (distance_corrigee_m - DISTANCE_REELLE_DEBUT_ALLEE_M) / (DISTANCE_REELLE_FIN_ALLEE_M - DISTANCE_REELLE_DEBUT_ALLEE_M)
    pos_y = ALLEE_Y_DEBUT + proportion * (ALLEE_Y_FIN - ALLEE_Y_DEBUT)
    
    pos_y_arrondi = int(round(pos_y))
    pos_y_final = max(ALLEE_Y_DEBUT, min(pos_y_arrondi, ALLEE_Y_FIN))
    
    return (CAMERA_X_POS, pos_y_final)

# ==============================================================================
# ======================== INITIALISATION ======================================
# ==============================================================================

# --- Configuration des chemins ---
script_dir = os.path.dirname(os.path.abspath(__file__))
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
align = rs.align(rs.stream.color)
print("Pipeline démarré.")

# ==============================================================================
# ======================== BOUCLE PRINCIPALE ===================================
# ==============================================================================
last_print_time = 0
try:
    print("\nDébut de la localisation. Appuyez sur Ctrl+C pour arrêter.")
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
        height, width, _ = color_image.shape

        blob = cv2.dnn.blobFromImage(color_image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        boxes, confidences = [], []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                if scores[class_id] > 0.6 and classes[class_id] == "person":
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w, h = int(detection[2] * width), int(detection[3] * height)
                    boxes.append([int(center_x - w / 2), int(center_y - h / 2), w, h])
                    confidences.append(float(scores[class_id]))
        
        min_distance_perçue = float('inf')
        if len(boxes) > 0:
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            if isinstance(indexes, np.ndarray):
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    roi_x, roi_y = max(0, int(x + w * 0.2)), max(0, int(y + h * 0.2))
                    roi_w, roi_h = int(w * 0.6), int(h * 0.6)
                    
                    depth_roi = depth_image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
                    valid_depths = depth_roi[depth_roi > 0]
                    
                    if valid_depths.size > 0:
                        distance = np.median(valid_depths) / 1000.0
                        if 0 < distance < min_distance_perçue:
                            min_distance_perçue = distance
            
            if min_distance_perçue != float('inf'):
                distance_reelle = appliquer_correction_lineaire(min_distance_perçue)
                position = calculer_position_sur_grille(distance_reelle)
                
                if position:
                    print(f"Position: (x={position[0]}, y={position[1]}) | Distance Corrigée: {distance_reelle:.2f}m")
                
                last_print_time = current_time

except KeyboardInterrupt:
    print("\nArrêt demandé par l'utilisateur.")
except Exception as e:
    print(f"Une erreur est survenue: {e}")
finally:
    print("Arrêt du pipeline de la caméra.")
    pipeline.stop()