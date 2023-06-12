import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Reading the data and dropping extra column
data = pd.read_csv('Iris.csv')
data.drop(columns=['Id'], inplace=True)

# Encoding and splitting the data
X = data.drop(['Species'], axis=1)
y = data['Species']
y_enc = LabelEncoder().fit_transform(y)
y_label = keras.utils.to_categorical(y_enc)

X_train, X_test, y_train, y_test = train_test_split(X, y_label, test_size=0.3, stratify=y_label)

# Creating a model
def get_model():
    model = keras.Sequential([
        keras.layers.Input(shape=X_train.shape[1:]),
        keras.layers.Dense(units=1000, activation='relu'),
        keras.layers.Dense(units=500, activation='relu'),
        keras.layers.Dense(units=300, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(units=3, activation='softmax')
    ])
    return model

model = get_model()

# Training the model
model.compile(optimizer='adam',
              loss=keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=25, validation_data=(X_test, y_test), verbose=1)
model.evaluate(X_test, y_test)

# Plotting the loss 
pd.DataFrame(history.history).plot(figsize=(10,6))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()