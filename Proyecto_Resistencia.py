import streamlit as st
import numpy as np
from PIL import Image, ImageDraw  # Importar ImageDraw para dibujar sobre la imagen
import Libreria as lib

st.title("🔍 Escáner de Resistencias")
# ----------- Interfaz Streamlit -----------

uploaded_file = st.file_uploader("Sube una imagen de una resistencia", type=["jpg", "png", "jpeg"])

# Sliders para ajustar los parámetros de detección
st.sidebar.header("Parámetros de Detección")
gamma_value = st.sidebar.slider(
    "Valor Gamma (aclarar/oscurecer imagen)",
    min_value=0.1, max_value=2.0, value=1.0, step=0.1,
    help="Valores menores a 1.0 aclaran la imagen, valores mayores la oscurecen."
)
sigma_factor = st.sidebar.slider(
    "Factor para el Umbral de Detección (sensibilidad)",
    min_value=1.0, max_value=5.0, value=2.5, step=0.1,
    help="Un valor más bajo detecta más cambios, uno más alto es más estricto."
)
distancia_agrupacion = st.sidebar.slider(
    "Distancia máxima para agrupar bordes (píxeles)",
    min_value=5, max_value=30, value=10, step=1,
    help="Distancia máxima entre bordes para considerarlos parte de la misma banda."
)

if uploaded_file:
    # Abrir y convertir la imagen a formato RGB
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    st.subheader("📷 Imagen original (sin procesar)")
    st.image(img_np, use_column_width=True)

    # Redimensionar si es muy grande para un procesamiento más eficiente
    if max(img_np.shape[:2]) > 300:
        img_np = lib.resize_image(img_np)
        st.subheader("📏 Imagen redimensionada")
        st.image(img_np, use_column_width=True)
    
    # Aplicar corrección gamma para aclarar/oscurecer la imagen
    img_np_gamma_corrected = lib.gama(img_np, gamma=gamma_value)
    st.subheader(f"☀️ Imagen con Corrección Gamma (gamma={gamma_value})")
    st.image(img_np_gamma_corrected, use_column_width=True)

    # Convertir la imagen a escala de grises (usando la imagen gamma corregida)
    gray_img = lib.rgb2gray(img_np_gamma_corrected)
    st.subheader("⚫ Imagen en escala de grises")
    gray_img_norm = (gray_img - gray_img.min()) / (gray_img.max() - gray_img.min())
    st.image(gray_img_norm, use_column_width=True, clamp=True)

    # Aplicar filtro bilateral para suavizar la imagen manteniendo los bordes
    gray_suavizada = lib.filtro_bilateral(gray_img, sigma_s=5, sigma_r=20, tamaño_ventana=5)
    st.subheader("🪄 Imagen suavizada (filtro bilateral)")
    gray_suavizada_norm = (gray_suavizada - gray_suavizada.min()) / (gray_suavizada.max() - gray_suavizada.min())
    st.image(gray_suavizada_norm, use_column_width=True, clamp=True)

    # ----------- Paso 3: Detección de bandas -----------

    st.subheader("📊 Perfil de intensidad por columna")
    perfil_columnas = np.mean(gray_suavizada, axis=0)
    st.line_chart(perfil_columnas)

    gradiente = np.abs(np.diff(perfil_columnas))
    umbral = np.mean(gradiente) + sigma_factor * np.std(gradiente)
    posibles_bordes = np.where(gradiente > umbral)[0]
    st.text(f"Posibles bordes detectados en columnas: {posibles_bordes}")

    st.subheader("🧠 Agrupando bandas detectadas")
    agrupados = []
    grupo_actual = [posibles_bordes[0]] if len(posibles_bordes) > 0 else []

    for idx in posibles_bordes[1:]:
        if idx - grupo_actual[-1] < distancia_agrupacion:
            grupo_actual.append(idx)
        else:
            agrupados.append(grupo_actual)
            grupo_actual = [idx]
    if grupo_actual:
        agrupados.append(grupo_actual)

    posiciones_bandas_original_detection = [int(np.mean(g)) for g in agrupados]
    st.text(f"Columnas centrales de bandas detectadas (orden inicial): {posiciones_bandas_original_detection}")

    st.subheader("🎨 Lectura de colores de bandas (detectadas originalmente)")

    bandas_info = []
    alto, ancho, _ = img_np_gamma_corrected.shape

    for col in posiciones_bandas_original_detection:
        fila_central = alto // 2
        fila_izq = max(0, fila_central - 2)
        fila_der = min(alto, fila_central + 3)
        col_izq = max(0, col - 2)
        col_der = min(ancho, col + 3)

        franja = img_np_gamma_corrected[fila_izq:fila_der, col_izq:col_der, :]
        color_promedio_rgb = np.mean(franja.reshape(-1, 3), axis=0).astype(int)
        
        # Usar función de tu librería para encontrar el color más cercano usando HSV
        nombre_color, valor_numerico, tolerancia_porcentaje = lib.color_mas_cercano_hsv_rango(color_promedio_rgb)
        
        bandas_info.append({
            'posicion': col,
            'rgb': color_promedio_rgb,
            'nombre': nombre_color,
            'valor': valor_numerico,
            'tolerancia': tolerancia_porcentaje
        })

    for i, band_data in enumerate(bandas_info):
        r, g, b = band_data['rgb']
        hex_color = '#%02x%02x%02x' % (r, g, b)
        st.markdown(f"**Banda {i+1}** (detectada en columna {band_data['posicion']}): Color promedio RGB = ({r}, {g}, {b})", unsafe_allow_html=True)
        st.markdown(f'<div style="width:100px;height:30px;background-color:{hex_color};border:1px solid #000;"></div>', unsafe_allow_html=True)

    st.markdown("---")

    # --- Invertir automáticamente el orden si detecta una banda dorada o plateada al final ---
    if len(bandas_info) > 0:
        last_band_name = bandas_info[-1]['nombre']
        if last_band_name in ["Dorado", "Plateado"]:
            st.warning("Se detectó una banda de tolerancia (Dorado/Plateado) al final. Invirtiendo el orden de las bandas para la lectura correcta.")
            bandas_info.reverse()

    st.subheader("🔢 Interpretación de bandas (orden ajustado)")

    numeric_values_for_calc = []
    resistencia_tolerancia = None

    for i, band_data in enumerate(bandas_info):
        if band_data['nombre'] in ["Dorado", "Plateado"]:
            st.markdown(f"**Banda {i+1}**: {band_data['nombre']} → Tolerancia: $\pm {band_data['tolerancia']*100}\%$")
            resistencia_tolerancia = band_data['tolerancia']
        else:
            st.markdown(f"**Banda {i+1}**: {band_data['nombre']} → Valor: {band_data['valor']}")
            if band_data['valor'] is not None:
                numeric_values_for_calc.append(band_data['valor'])

    if len(numeric_values_for_calc) >= 3:
        valor_base = int(str(numeric_values_for_calc[0]) + str(numeric_values_for_calc[1]))
        multiplicador = 10 ** numeric_values_for_calc[2]
        val_ohmios = valor_base * multiplicador
        
        if resistencia_tolerancia is not None:
            st.success(f"💡 Valor estimado de la resistencia: **{val_ohmios} Ω** con tolerancia de $\pm {resistencia_tolerancia*100}\\%$")
        else:
            st.success(f"💡 Valor estimado de la resistencia: **{val_ohmios} Ω**")
    else:
        st.warning("No se detectaron suficientes bandas numéricas para estimar el valor (se requieren al menos 3 para el cálculo actual).")

    st.subheader("✨ Visualización de bandas detectadas")
    img_with_bands = Image.fromarray(img_np_gamma_corrected.copy().astype(np.uint8))
    draw = ImageDraw.Draw(img_with_bands)

    for band_data in bandas_info:
        col = band_data['posicion']
        draw.line((col, 0, col, img_np_gamma_corrected.shape[0]), fill=(255, 0, 0), width=3)

    st.image(img_with_bands, use_column_width=True, caption="Imagen con bandas detectadas resaltadas en rojo (orden ajustado)")