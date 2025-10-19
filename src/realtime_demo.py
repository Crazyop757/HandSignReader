import cv2
import numpy as np
from tensorflow import keras
from collections import deque

model = keras.models.load_model("asl_cnn_savedmodel.keras")

import json
with open("label_map.json") as f:
    label_map = json.load(f)

import mediapipe as mp
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, 
                       min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

prediction_buffer = deque(maxlen=10)
confidence_threshold = 40.0

def preprocess_hand_image(hand_roi_rgb):
    h, w = hand_roi_rgb.shape[:2]
    max_dim = max(h, w)
    
    square_img = np.ones((max_dim, max_dim, 3), dtype=np.uint8) * 200 
    
    y_offset = (max_dim - h) // 2
    x_offset = (max_dim - w) // 2
    square_img[y_offset:y_offset+h, x_offset:x_offset+w] = hand_roi_rgb
    
    resized = cv2.resize(square_img, (64, 64), interpolation=cv2.INTER_AREA)
    
    resized_lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
    resized_lab[:,:,0] = cv2.equalizeHist(resized_lab[:,:,0])
    resized = cv2.cvtColor(resized_lab, cv2.COLOR_LAB2RGB)
    
    normalized = resized.astype(np.float32) / 255.0
    
    return normalized, resized

def get_stable_prediction(pred, confidence):
    class_id = np.argmax(pred)
    
    if confidence > confidence_threshold:
        prediction_buffer.append(class_id)
    
    if len(prediction_buffer) > 0:
        unique, counts = np.unique(list(prediction_buffer), return_counts=True)
        most_common_idx = unique[np.argmax(counts)]
        return most_common_idx
    
    return class_id

cap = cv2.VideoCapture(0)

print("\nPress 'q' to quit, 'r' to reset prediction buffer")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            h, w, _ = frame.shape
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))

            margin = 50
            x_min, x_max = max(0, x_min - margin), min(w, x_max + margin)
            y_min, y_max = max(0, y_min - margin), min(h, y_max + margin)

            cropped_hand = frame_rgb[y_min:y_max, x_min:x_max]

            if cropped_hand.size > 0 and cropped_hand.shape[0] > 0 and cropped_hand.shape[1] > 0:
                
                processed_hand, display_hand = preprocess_hand_image(cropped_hand)
                input_img = np.expand_dims(processed_hand, axis=0)

                pred = model.predict(input_img, verbose=0)
                raw_class_id = np.argmax(pred)
                confidence = np.max(pred) * 100
                
                stable_class_id = get_stable_prediction(pred[0], confidence)
                letter = list(label_map.keys())[list(label_map.values()).index(stable_class_id)]
                
                if confidence > 70:
                    color = (0, 255, 0) 
                elif confidence > 50:
                    color = (0, 200, 200)
                else:
                    color = (0, 165, 255) 
                
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 3)
                
                text = f"Sign: {letter}"
                conf_text = f"Conf: {confidence:.1f}%"
                cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            2.0, color, 4, cv2.LINE_AA)
                cv2.putText(frame, conf_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, color, 2, cv2.LINE_AA)
                
                top3_idx = np.argsort(pred[0])[-3:][::-1]
                y_offset = 160
                cv2.putText(frame, "Top 3:", (10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                y_offset += 35
                
                for i, idx in enumerate(top3_idx, 1):
                    letter_name = list(label_map.keys())[list(label_map.values()).index(idx)]
                    conf = pred[0][idx] * 100
                    cv2.putText(frame, f"{i}. {letter_name}: {conf:.1f}%", (10, y_offset), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    y_offset += 30
                
                display_crop = cv2.resize(display_hand, (150, 150))
                display_crop = cv2.cvtColor(display_crop, cv2.COLOR_RGB2BGR)
                frame[10:160, w-160:w-10] = display_crop
                
                cv2.putText(frame, "Model View", (w-155, 180), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        prediction_buffer.clear()
        cv2.putText(frame, "No hand detected", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow("ASL Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        prediction_buffer.clear()
        print("Prediction buffer reset!")

cap.release()
cv2.destroyAllWindows()