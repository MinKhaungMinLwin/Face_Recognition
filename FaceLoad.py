import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import face_recognition


model = tf.keras.models.load_model('face_resnet50_1.keras')

class_names = ['Kaung Htet Htun\n - roll (001)', 'Kaung Htet Naing\n VI-MCE-3', 'MKML\n-code-001', 
               'Soe Thu Aung\n VI-MCE-5', 'Thant Zin Aung\n VI-MCE-4']

def predict_person(img_array):
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 #divide

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    higher_ratio =  prediction[0][predicted_class]

    if higher_ratio > 0.4:
        return class_names[predicted_class] if predicted_class < len(class_names) else 'Unknown'
    else:
        return 'Unknown'

cap = cv2.VideoCapture(0)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (640,480))
    rgb_frame = frame[:, :, ::-1]#convert to RGB

    if frame_count % 5 == 0:
        face_locations = face_recognition.face_locations(rgb_frame)
    frame_count += 1

    for (top, right, bottom, left) in face_locations:
        face_img = frame[top:bottom, left:right]
        face_img = cv2.resize(face_img, (224, 224))
        face_img = img_to_array(face_img)
        person = predict_person(face_img)
        cv2.rectangle(frame, (left, top), (right,    bottom), (255, 0, 0), 2)
        cv2.putText(frame, person, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
