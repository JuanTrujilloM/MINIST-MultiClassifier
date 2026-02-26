import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# -------------------------
# CONFIGURACIÓN
# -------------------------
st.set_page_config(page_title="Clasificador MNIST", layout="wide")
st.title("🔢 Clasificador de Dígitos - MNIST")

# -------------------------
# CARGA DE DATOS
# -------------------------
@st.cache_data
def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_data()

# Normalización
X_train_flat = X_train.reshape(-1, 28*28) / 255.0
X_test_flat = X_test.reshape(-1, 28*28) / 255.0

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.header("⚙ Configuración")

model_option = st.sidebar.selectbox(
    "Seleccione el modelo",
    ["Logistic Regression", "Random Forest", "SVM", "Neural Network"]
)

# -------------------------
# ENTRENAMIENTO
# -------------------------
st.subheader("📊 Entrenamiento del Modelo")

if model_option == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_flat, y_train)
    y_pred = model.predict(X_test_flat)

elif model_option == "Random Forest":
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train_flat, y_train)
    y_pred = model.predict(X_test_flat)

elif model_option == "SVM":
    model = SVC()
    model.fit(X_train_flat[:10000], y_train[:10000])  # reducir para rapidez
    y_pred = model.predict(X_test_flat)

else:
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train_cat, epochs=3, verbose=0)
    loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
    y_pred = np.argmax(model.predict(X_test), axis=1)

# -------------------------
# MÉTRICAS
# -------------------------
st.subheader("📈 Métricas de Desempeño")

if model_option != "Neural Network":
    accuracy = accuracy_score(y_test, y_pred)
else:
    accuracy = acc

st.metric("Accuracy", f"{accuracy:.4f}")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6,6))
sns.heatmap(cm, annot=False, cmap="Blues")
plt.title("Matriz de Confusión")
st.pyplot(fig)

st.text("Reporte de Clasificación")
st.text(classification_report(y_test, y_pred))

# -------------------------
# PRUEBA CON IMAGEN DEL DATASET
# -------------------------
st.subheader("🖼 Probar Imagen del Dataset")

index = st.slider("Seleccione índice de imagen", 0, len(X_test)-1, 0)

image = X_test[index]

col1, col2 = st.columns(2)

with col1:
    st.image(image, width=200, caption="Imagen seleccionada")

with col2:
    if model_option != "Neural Network":
        prediction = model.predict([X_test_flat[index]])[0]
    else:
        prediction = np.argmax(model.predict(image.reshape(1,28,28)), axis=1)[0]

    st.success(f"Predicción: {prediction}")
    st.info(f"Etiqueta real: {y_test[index]}")

# -------------------------
# SUBIR IMAGEN PERSONAL
# -------------------------
st.subheader("⬆ Subir Imagen Propia")

uploaded_file = st.file_uploader("Sube una imagen 28x28 en escala de grises", type=["png","jpg","jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("L").resize((28,28))
    img_array = np.array(img) / 255.0

    st.image(img, caption="Imagen cargada", width=200)

    if model_option != "Neural Network":
        pred = model.predict([img_array.flatten()])[0]
    else:
        pred = np.argmax(model.predict(img_array.reshape(1,28,28)), axis=1)[0]

    st.success(f"Predicción del modelo: {pred}")