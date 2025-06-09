import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Diccionario para traducir el color al número
color_codes = {
    "negro": 0, "marrón": 1, "rojo": 2, "naranja": 3, "amarillo": 4,
    "verde": 5, "azul": 6, "violeta": 7, "gris": 8, "blanco": 9
}

# Diccionario LAB mejorado con tolerancias individuales
lab_reference = {
    "negro": {"val": np.array([25, 128, 128]), "tol": 30},
    "marrón": {"val": np.array([40, 145, 140]), "tol": 35},
    "rojo": {"val": np.array([55, 180, 135]), "tol": 25},
    "naranja": {"val": np.array([70, 190, 160]), "tol": 20},
    "amarillo": {"val": np.array([90, 110, 170]), "tol": 25},
    "verde": {"val": np.array([65, 125, 125]), "tol": 40},
    "azul": {"val": np.array([35, 155, 180]), "tol": 35},
    "violeta": {"val": np.array([45, 195, 150]), "tol": 30},
    "gris": {"val": np.array([110, 128, 128]), "tol": 20},
    "blanco": {"val": np.array([240, 128, 128]), "tol": 10}
}

# Función mejorada con tolerancias individuales
def detectar_color_lab(lab_color):
    min_dist = float('inf')
    color_detectado = None
    debug = {}
    
    for color, data in lab_reference.items():
        ref_lab = data["val"]
        tol = data["tol"]
        dist = np.linalg.norm(lab_color - ref_lab)
        debug[color] = dist
        
        if dist < min_dist and dist < tol:
            min_dist = dist
            color_detectado = color
    
    st.write(f"[DEBUG] Distancias: { {k: round(v, 1) for k, v in debug.items()} }")
    return color_detectado

# Streamlit app
st.title("Lector automático de bandas de resistencias")

uploaded_file = st.file_uploader("Sube una imagen de una resistencia", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    st.image(image_np, caption="Imagen cargada", use_column_width=True)

    # Convertimos la imagen a LAB
    lab_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    y = lab_image.shape[0] // 2  # Línea central horizontal

    # --- DETECCIÓN AUTOMÁTICA DE BANDAS ---
    l_channel = lab_image[:, :, 0]
    _, mask = cv2.threshold(l_channel, 80, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bandas_x = []
    for cnt in contornos:
        x, y_c, w, h = cv2.boundingRect(cnt)
        aspect_ratio = h / float(w)
        if aspect_ratio > 2 and h > 20:
            bandas_x.append(x + w // 2)

    bandas_x = sorted(bandas_x)[:5]  # Limitar a las 5 más a la izquierda

    # Visualización de líneas detectadas
    vis = image_np.copy()
    for x in bandas_x:
        cv2.line(vis, (x, 0), (x, vis.shape[0]), (0, 255, 0), 2)
    st.image(vis, caption="Bandas detectadas automáticamente", use_column_width=True)

    # --- DETECCIÓN DE COLORES ---
    colores_detectados = []
    for i, x in enumerate(bandas_x):
        ventana = lab_image[y - 10:y + 11, x - 10:x + 11, :]
        lab_val = ventana.reshape(-1, 3).mean(axis=0)
        color = detectar_color_lab(lab_val)
        colores_detectados.append(color)
        st.write(f"[DEBUG] Banda {i + 1} en x={x} → LAB promedio: {lab_val.astype(int)} → {color}")

    # --- CÁLCULO DEL VALOR DE RESISTENCIA ---
    if len(colores_detectados) == 5 and all(c is not None for c in colores_detectados):
        try:
            d1 = color_codes[colores_detectados[0]]
            d2 = color_codes[colores_detectados[1]]
            d3 = color_codes[colores_detectados[2]]
            multiplicador = 10 ** color_codes[colores_detectados[3]]
            resistencia = (d1 * 100 + d2 * 10 + d3) * multiplicador
            
            # Formateo para valores grandes
            if resistencia >= 1e6:
                resistencia = f"{resistencia / 1e6:.2f} MΩ"
            elif resistencia >= 1e3:
                resistencia = f"{resistencia / 1e3:.2f} KΩ"
            else:
                resistencia = f"{resistencia} Ω"
                
            st.success(f"Valor de la resistencia: {resistencia}")
        except KeyError:
            st.error("Error al convertir los colores a números. ¿Alguno es inválido?")
    else:
        st.warning("No se detectaron 5 bandas válidas. Revisa la imagen o cambia el umbral.")