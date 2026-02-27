import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random
import time

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from streamlit_drawable_canvas import st_canvas

st.title("Clasificador de Dígitos MNIST - Proyecto Completo")

# ==============================
# CARGA Y ENTRENAMIENTO
# ==============================

@st.cache_resource
def load_and_train():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train_flat = X_train.reshape(-1, 28*28) / 255.0
    X_test_flat = X_test.reshape(-1, 28*28) / 255.0

    # Modelos clásicos
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "KNN": KNeighborsClassifier()
    }

    trained_models = {}
    metrics = {}

    for name, model in models.items():
        model.fit(X_train_flat, y_train)
        preds = model.predict(X_test_flat)
        acc = accuracy_score(y_test, preds)
        trained_models[name] = model
        metrics[name] = acc

    # ================= CNN =================
    X_train_cnn = X_train.reshape(-1,28,28,1)/255.0
    X_test_cnn = X_test.reshape(-1,28,28,1)/255.0

    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    cnn = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    cnn.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    cnn.fit(X_train_cnn, y_train_cat, epochs=2, verbose=0)

    loss, acc = cnn.evaluate(X_test_cnn, y_test_cat, verbose=0)

    trained_models["CNN"] = cnn
    metrics["CNN"] = acc

    return trained_models, metrics, X_test, y_test


models, metrics, X_test, y_test = load_and_train()

# ==============================
# COMPARACIÓN DE MODELOS
# ==============================

st.subheader("Comparación de Modelos")

fig, ax = plt.subplots()
ax.bar(metrics.keys(), metrics.values())
ax.set_ylabel("Accuracy")
ax.set_ylim([0.8,1])
plt.xticks(rotation=45)
st.pyplot(fig)

# ==============================
# SELECCIÓN MODELO
# ==============================

model_name = st.selectbox("Seleccione modelo:", list(models.keys()))
model = models[model_name]

st.write(f"Accuracy del modelo: {metrics[model_name]:.4f}")

# ==============================
# MATRIZ DE CONFUSIÓN
# ==============================

if model_name != "CNN":
    X_test_flat = X_test.reshape(-1,28*28)/255.0
    preds = model.predict(X_test_flat)
else:
    X_test_cnn = X_test.reshape(-1,28,28,1)/255.0
    preds = np.argmax(model.predict(X_test_cnn), axis=1)

cm = confusion_matrix(y_test, preds)

st.subheader("Matriz de Confusión")

fig2, ax2 = plt.subplots()
ax2.imshow(cm, cmap='Blues')
ax2.set_xlabel("Predicho")
ax2.set_ylabel("Real")
st.pyplot(fig2)

# ==============================
# IMAGEN ALEATORIA
# ==============================

if st.button("Probar con imagen de validación"):
    idx = random.randint(0,len(X_test)-1)
    img = X_test[idx]
    label = y_test[idx]

    st.image(img, width=150, caption="Imagen MNIST")

    if model_name != "CNN":
        pred = model.predict(img.reshape(1,-1)/255.0)
        prediction = pred[0]
    else:
        pred = model.predict(img.reshape(1,28,28,1)/255.0)
        prediction = np.argmax(pred)

    st.write(f"Predicción: {prediction}")
    st.write(f"Etiqueta real: {label}")

# ==============================
# DIBUJAR NÚMERO
# ==============================

st.subheader("Dibuja un número")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Clasificar dibujo"):
    if canvas_result.image_data is not None:
        img = canvas_result.image_data[:,:,0]
        img = img[::10, ::10]  # reducir tamaño
        img = img.reshape(1,28,28,1)/255.0

        if model_name != "CNN":
            pred = model.predict(img.reshape(1,-1))
            prediction = pred[0]
        else:
            pred = model.predict(img)
            prediction = np.argmax(pred)

        st.write(f"Predicción del dibujo: {prediction}")