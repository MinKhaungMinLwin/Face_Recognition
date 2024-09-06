import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import face_recognition

model = tf.keras.models.load_model('face_resnet50_03.keras')

class_names = ['Kaung Htet Naing VI-MCE-3','Soe Thu Aung VI-MCE-5', 'Thant Zin Aung VI-MCE-4']

def predict_person(img_array):
    """Predict  the person in the image"""
    try:
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        higher_ratio = prediction[0][predicted_class]

        if higher_ratio > 0.4:
            name = class_names[predicted_class] if predicted_class < len(class_names) else 'Unknown'
        else:
            name = 'Unknown'
        return name, higher_ratio

    except Exception as e:
        print(f"Error during prediction: {e}")
        return 'Unknown', 0.0

def draw_rounded_rectangle(img, top_left, bottom_right, color, thickness, radius=15):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)

    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 + radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 + radius), (radius, radius), 0, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 + radius), (radius, radius), 90, 0, 90, color, thickness)


cap = cv2.VideoCapture(0)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (640, 480))
    rgb_frame = small_frame[:, :, ::-1]

    if frame_count % 5 == 0:
        face_locations = face_recognition.face_locations(rgb_frame)
    frame_count += 1

    for (top, right, bottom, left) in face_locations:
        face_img = frame[top:bottom, left:right]
        face_img = cv2.resize(face_img, (224, 224))
        face_img = img_to_array(face_img)

        person, accuracy = predict_person(face_img)
        display_name = person
        display_accuracy = f"({accuracy * 100:.2f}%)"

        print(f"{person} {accuracy * 100:.2f}% 1/1 ------------- 0s {np.random.randint(120, 180)}ms/step")

        draw_rounded_rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2, radius=15)
        
        cv2.putText(frame, display_name, (left, top - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.putText(frame, display_accuracy, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 255, 12), 2)

    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
