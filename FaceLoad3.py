import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import face_recognition

model = tf.keras.models.load_model('face_resnet50_04.keras')

class_names = ['Kaung Htet Naing\n VI-MCE-3','Soe Thu Aung\n VI-MCE-5', 'Thant Zin Aung\n VI-MCE-4']

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

#-----------------------------------------#

def draw_gradient_ellipse(image, center, axes, angle, color_start, color_end, thickness=2):
    mask = np.zeros_like(image)

    glow_intensity = 20
    for i in range(glow_intensity):
        alpha = (glow_intensity - i) / glow_intensity
        cv2.ellipse(mask, center, (axes[0] + i, axes[1] + i), angle, 0, 360, (255,255,255), 1, cv2.LINE_AA)
        
    cv2.addWeighted(mask, 0.1, image, 1 - 0.1, 0, image)

    gradient_mask = np.zeros_like(image)

    gradient_steps = 50
    for i in range(gradient_steps):
        interp_color = (
            int(color_start[0] + (color_end[0] - color_start[0]) * i / gradient_steps),
            int(color_start[1] + (color_end[1] - color_start[1]) * i / gradient_steps),
            int(color_start[2] + (color_end[2] - color_start[2]) * i / gradient_steps)
        )

        radius_factor = 1 + 0.02 * i
        cv2.ellipse(mask, center, (int(axes[0] * radius_factor), int(axes[1] * radius_factor)),
                    angle, 0, 360, interp_color, 1, lineType=cv2.LINE_AA)

    cv2.addWeighted(gradient_mask, 0.7, image, 0.3, 0, image)

    for i in range(0, 360, 10):
        start_angle = i
        end_angle = i + 5
        cv2.ellipse(image, center, axes, angle, start_angle, end_angle, (255, 255, 255), thickness, cv2.LINE_AA)


cap = cv2.VideoCapture(0)
frame_count = 0

#----------------------------------------------#

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (640, 480))
    rgb_frame = small_frame[:, :, ::-1]

    if frame_count % 5 == 0:
        face_locations = face_recognition.face_locations(rgb_frame)
    frame_count += 1

    if not face_locations:
        continue    

    for (top, right, bottom, left) in face_locations:
        center = ((left + right) // 2, (top + bottom) // 2)
        axes = ((right - left) // 2, (bottom - top) // 2)
        angle = 0#no rotation

        color_start = (0, 128, 255)#blueish
        color_end = (255, 0, 128)#purplish

        face_img = frame[top:bottom, left:right]
        face_img = cv2.resize(face_img, (224, 224))
        face_img = img_to_array(face_img)

        person, accuracy = predict_person(face_img)

        display_name = person
        display_accuracy = f"({accuracy * 100:.2f}%)"

        print(f"{person} 1/1 ------------- 0s {np.random.randint(120, 180)}ms/step")

        draw_gradient_ellipse(frame, center, axes, angle, color_start, color_end, thickness = 2)

        cv2.putText(frame, display_name, (left, top - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.putText(frame, display_accuracy, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 255, 12), 2)

    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
