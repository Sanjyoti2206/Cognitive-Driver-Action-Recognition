
def train_model():

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from keras.models import Sequential
    from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
    from tensorflow.keras.optimizers import SGD
    from keras.preprocessing.image import ImageDataGenerator
    
    # Base directory (Make sure this path is correct)
    basepath = r"C:/Users/ajink/OneDrive/Desktop/pranali/100% Code Exam Video/Driver Code/Final Code"
    
    # Check if dataset paths exist
    train_path = os.path.join(basepath, "training_set")
    test_path = os.path.join(basepath, "testing_set")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
     raise FileNotFoundError("Dataset folders not found! Please check the paths.")
    
    # Ensure datasets are not empty
    if len(os.listdir(train_path)) == 0 or len(os.listdir(test_path)) == 0:
     raise ValueError("Dataset folders are empty! Please add images to the respective class folders.")
    
    # Initializing the CNN
    classifier = Sequential()
    
    # Step 1 - Convolution Layer
    classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
    
    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Adding second convolution layer
    classifier.add(Convolution2D(32, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Adding third convolution layer
    classifier.add(Convolution2D(64, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Step 3 - Flattening
    classifier.add(Flatten())
    
    # Step 4 - Fully Connected Layer
    classifier.add(Dense(256, activation='relu'))
    classifier.add(Dropout(0.8))
    classifier.add(Dense(5, activation='softmax'))  # Change 5 to the number of classes in your dataset
    
    # Compiling The CNN
    classifier.compile(
    optimizer=SGD(learning_rate=0.01),  # Fixed deprecated 'lr' argument
    loss='categorical_crossentropy',
    metrics=['accuracy']
    )
    
    # Data Augmentation & Preprocessing
    train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    training_set = train_datagen.flow_from_directory(
    train_path,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
    )
    
    test_set = test_datagen.flow_from_directory(
    test_path,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
    )
    
    steps_per_epoch = int(np.ceil(training_set.samples / 32))
    val_steps = int(np.ceil(test_set.samples / 32))
    
    # Training the CNN
    history = classifier.fit(
    training_set,
    steps_per_epoch=steps_per_epoch,
    epochs=300,
    validation_data=test_set,
    validation_steps=val_steps
    )
    
    # Save the trained model
    classifier.save(os.path.join(basepath, 'drivermodel.h5'))
    
    # Evaluate the model
    train_scores = classifier.evaluate(training_set, verbose=1)
    test_scores = classifier.evaluate(test_set, verbose=1)
    
    train_acc = f"Training Accuracy: {train_scores[1] * 100:.2f}%"
    test_acc = f"Testing Accuracy: {test_scores[1] * 100:.2f}%"
    print(train_acc)
    print(test_acc)
    
    # Plot Accuracy
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Test Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(os.path.join(basepath, "accuracy.png"), bbox_inches='tight')
    plt.show()
    
    # Plot Loss
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(os.path.join(basepath, "loss.png"), bbox_inches='tight')
    plt.show()
    
    # Return results as a message
    msg = train_acc + '\n' + test_acc
    print(msg)
#train()