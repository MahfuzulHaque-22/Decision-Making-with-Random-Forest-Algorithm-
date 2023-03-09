import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('my_dataset.csv')

# Preprocess the data
X = data.drop('label', axis=1).values.astype('float32') / 255.0
y = data['label'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a random forest classifier on the training set
rfc = RandomForestClassifier(n_estimators=100, max_depth=10)
rfc.fit(X_train, y_train)

# Evaluate the random forest classifier on the testing set
y_pred_rfc = rfc.predict(X_test)
acc_rfc = accuracy_score(y_test, y_pred_rfc)
print(f"Random forest classifier accuracy: {acc_rfc:.2f}")

# Train a convolutional neural network on the training set
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
X_train_cnn = X_train.reshape(-1, 28, 28, 1)
X_test_cnn = X_test.reshape(-1, 28, 28, 1)
model.fit(X_train_cnn, y_train, epochs=5, validation_data=(X_test_cnn, y_test))

# Evaluate the convolutional neural network on the testing set
y_pred_cnn = np.argmax(model.predict(X_test_cnn), axis=-1)
acc_cnn = accuracy_score(y_test, y_pred_cnn)
print(f"Convolutional neural network accuracy: {acc_cnn:.2f}")

# Make a decision based on the highest accuracy
if acc_rfc > acc_cnn:
    print("The random forest classifier is the best model.")
else:
    print("The convolutional neural network is the best model.")
