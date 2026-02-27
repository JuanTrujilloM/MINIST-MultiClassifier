import streamlit as st
import numpy as np
import random
from tensorflow.keras.datasets import mnist
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

st.title("Clasificador de Dígitos MNIST")

@st.cache_resource
def load_and_train_models():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(-1, 28*28) / 255.0
    X_test = X_test.reshape(-1, 28*28) / 255.0

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "KNN": KNeighborsClassifier()
    }

    trained_models = {}
    metrics = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        trained_models[name] = model
        metrics[name] = acc

    return trained_models, metrics, X_test, y_test

models, metrics, X_test, y_test = load_and_train_models()

# Selección del modelo
model_name = st.selectbox("Seleccione el modelo:", list(models.keys()))
model = models[model_name]

st.subheader("Métrica del modelo")
st.write(f"Accuracy: {metrics[model_name]:.4f}")

if st.button("Generar imagen de validación"):
    idx = random.randint(0, len(X_test)-1)
    image = X_test[idx]
    label = y_test[idx]

    st.image(image.reshape(28,28), caption="Imagen a clasificar", width=150)

    start = time.time()
    prediction = model.predict(image.reshape(1,-1))
    end = time.time()

    st.write(f"Predicción: {prediction[0]}")
    st.write(f"Etiqueta real: {label}")
    st.write(f"Tiempo de inferencia: {(end-start)*1000:.2f} ms")