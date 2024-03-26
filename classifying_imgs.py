import os
import numpy as np
import pandas as pd
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout  # Import Dropout
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# Data paths
train_data_dir = '/Users/dgomaa/Downloads/karimooooo/project/stem-oct-cs-club-1/train/isolated_alphabets_per_alphabet'
test_data_dir = '/Users/dgomaa/Downloads/karimooooo/project/stem-oct-cs-club-1/test/test'

# Define the class names based on your dataset
class_names = ['ain_begin', 'ain_end', 'ain_middle', 'ain_regular', 'alif_end', 'alif_hamza', 'alif_regular', 'beh_begin', 'beh_end', 'beh_middle', 'beh_regular', 'dal_end', 'dal_regular', 'feh_begin', 'feh_end', 'feh_middle', 'feh_regular', 'heh_begin', 'heh_end', 'heh_middle', 'heh_regular', 'jeem_begin', 'jeem_end', 'jeem_middle', 'jeem_regular', 'kaf_begin', 'kaf_end', 'kaf_middle', 'kaf_regular', 'lam_alif', 'lam_begin', 'lam_end', 'lam_middle', 'lam_regular', 'meem_begin', 'meem_end', 'meem_middle', 'meem_regular', 'noon_begin', 'noon_end', 'noon_middle', 'noon_regular', 'qaf_begin', 'qaf_end', 'qaf_middle', 'qaf_regular', 'raa_end', 'raa_regular', 'sad_begin', 'sad_end', 'sad_middle', 'sad_regular', 'seen_begin', 'seen_end', 'seen_middle', 'seen_regular', 'tah_end', 'tah_middle', 'tah_regular', 'waw_end', 'waw_regular', 'yaa_begin', 'yaa_end', 'yaa_middle', 'yaa_regular']


# Data preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,  
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,  
    zoom_range=0.15,   
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=32,  
    class_mode='sparse',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',
    subset='validation'
)

# Calculate class weights to handle class imbalance
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights_dict = {i: class_weights[i] for i in range(len(class_names))}

# Load pre-trained ResNet50 model without top layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(512, activation='relu', kernel_regularizer='l2')(x)
x = Dropout(0.3)(x)  # Adding dropout layer for regularization
predictions = Dense(len(class_names), activation='softmax')(x)

# Compile the model
model = Model(inputs=base_model.input, outputs=predictions)
optimizer = Adam(learning_rate=0.0001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Early stopping to prevent overfitting and reduce learning rate on plateau
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-7, verbose=1)

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=1,
    class_weight=class_weights_dict,
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate the model on the validation dataset
validation_loss, validation_accuracy = model.evaluate(validation_generator)
print(f"Validation Loss: {validation_loss:.4f}")
print(f"Validation Accuracy: {validation_accuracy:.2%}")

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()



# Make predictions on the test dataset
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(248, 350),
    batch_size=1,
    shuffle=False,
    class_mode=None
)

# Get filenames without the directory path
test_filenames = [os.path.basename(filename) for filename in test_generator.filenames]

predictions = model.predict(test_generator)
predicted_labels = [class_names[np.argmax(pred)] for pred in predictions]

# Create submission CSV
submission_df = pd.DataFrame({'ID': test_filenames, 'Letter': predicted_labels})
submission_df.to_csv('Submission.csv', index=False)