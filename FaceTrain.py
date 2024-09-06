import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

base_model = ResNet50(weights='imagenet', include_top = False, input_shape = (224, 224, 3))

x = base_model.output#custom layers on top
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation = 'relu')(x)
predictions = Dense(5, activation='softmax')(x)

#full model
model = Model(inputs=base_model.input, outputs= predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1./255,#Data Augmentation
    validation_split = 0.1,
    rotation_range = 30,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range= 0.2,
    horizontal_flip = True
)

train_generator = train_datagen.flow_from_directory(
    '/home/sinbad/Documents/AI intro/Face_roll_name_Detect/Dataset/Train/',
    target_size = (224, 224),
    batch_size = 32,
    class_mode = 'categorical',
    subset = 'training'
)

class_indices = train_generator.class_indices
class_names = sorted(class_indices, key=lambda x: class_indices[x])
print("class names :", class_names)
# print(train_generator.shape),
# print(validation_generator.shape)
validation_dir = '/home/sinbad/Documents/AI intro/Face_roll_name_Detect/Dataset/Test'

validation_generator = ImageDataGenerator(rescale=1./255)
validation_generator = validation_generator.flow_from_directory(
    validation_dir,
    target_size = (224, 224),
    batch_size = 32,
    class_mode = 'categorical'
)

call_back = ModelCheckpoint(
    filepath = 'face_resnet50.keras',
    monitor = 'val_accuracy',
    save_best_only = True,
    mode = 'max',#max the monitor metric
    save_weights_only = False,
    verbose = 1#show message when model saved
)

best_model = model.fit(train_generator, epochs = 56, validation_data = validation_generator, 
          callbacks = [call_back])

loss, accuracy = model.evaluate(validation_generator)

print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# model.save('face_recog_resnet50.keras')

