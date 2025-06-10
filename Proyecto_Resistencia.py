import streamlit as st
import numpy as np
import math
from PIL import Image, ImageDraw # Importar ImageDraw para dibujar sobre la imagen
import colorsys # Para convertir RGB a HSV

st.title("🔍 Escáner de Resistencias")

# ----------- Funciones auxiliares -----------

def rgb2gray(img):
    """Convierte una imagen RGB a escala de grises."""
    alto, ancho, canales = img.shape
    gray = np.zeros((alto, ancho), dtype=float)
    for i in range(alto):
        for j in range(ancho):
            r, g, b = img[i, j]
            gray[i, j] = 0.299 * r + 0.587 * g + 0.114 * b
    return gray

def resize_image(img, max_dim=300):
    """Escala la imagen manteniendo la relación de aspecto para un procesamiento más rápido."""
    factor = max_dim / max(img.shape[0], img.shape[1])
    new_size = (int(img.shape[1]*factor), int(img.shape[0]*factor))  # ancho, alto
    resized_img = np.array(Image.fromarray(img).resize(new_size, Image.BILINEAR))
    return resized_img

def gama(img, c=1.0, gamma=0.5):
    """
    Aplica la transformación gamma a una imagen.
    gamma < 1.0 aclara la imagen (ilumina áreas oscuras); gamma > 1.0 la oscurece.
    c es una constante de escala, usualmente 1.0.
    Esta función está optimizada para NumPy y maneja imágenes RGB.
    """
    # Convertir la imagen a tipo flotante y normalizar a [0, 1]
    img_float = img.astype(float) / 255.0
    
    # Aplicar la fórmula de corrección gamma a todos los píxeles y canales simultáneamente
    # s = c * (rk ** gamma)
    output_img_float = c * (img_float ** gamma)
    
    # Desnormalizar a [0, 255] y asegurar que los valores estén dentro del rango uint8
    output_img_uint8 = np.clip(output_img_float * 255.0, 0, 255).astype(np.uint8)
    
    return output_img_uint8

def filtro_bilateral(imagen, sigma_s, sigma_r, tamaño_ventana):
    """
    Aplica un filtro bilateral a una imagen.
    sigma_s: Desviación estándar del filtro espacial.
    sigma_r: Desviación estándar del filtro de rango (intensidad).
    tamaño_ventana: Tamaño del kernel del filtro.
    """
    alto, ancho = imagen.shape
    resultado = np.zeros_like(imagen)
    offset = tamaño_ventana // 2
    eps = 1e-8 # Pequeño valor para evitar división por cero

    for x in range(alto):
        for y in range(ancho):
            suma_pesos = 0
            suma_intensidades = 0
            for i in range(-offset, offset + 1):
                for j in range(-offset, offset + 1):
                    xi = x + i
                    yj = y + j
                    if 0 <= xi < alto and 0 <= yj < ancho:
                        distancia_espacial = math.sqrt(i**2 + j**2)
                        diferencia_intensidad = imagen[xi][yj] - imagen[x][y]
                        peso_espacial = math.exp(- (distancia_espacial ** 2) / (2 * sigma_s ** 2))
                        peso_rango = math.exp(- (diferencia_intensidad ** 2) / (2 * sigma_r ** 2))
                        peso_total = peso_espacial * peso_rango
                        suma_pesos += peso_total
                        suma_intensidades += imagen[xi][yj] * peso_total
            resultado[x][y] = suma_intensidades / (suma_pesos + eps)
    return resultado

# ----------- Definiciones de colores y funciones de interpretación -----------

# Diccionario de colores con RANGOS HSV más realistas para resistencias
# Formato: "Nombre": [ (min_H, min_S, min_V), (max_H, max_S, max_V), Valor numérico de banda, Tolerancia/Multiplicador ]
# H (Tono): 0-360 grados (aquí normalizado a 0-1.0 para colorsys)
# S (Saturación): 0-1.0 (pureza del color, 0 es gris)
# V (Valor/Brillo): 0-1.0 (brillo, 0 es negro)
# NOTA: Los rangos H se manejan con cuidado para colores que cruzan el 0/360 (rojo)
COLORES_RESISTENCIA_HSV = {
    "Negro":    [(0.0, 0.0, 0.0), (1.0, 1.0, 0.1), 0, None], # Muy bajo brillo, ajustado para no confundir con rojos muy oscuros
    "Marrón":   [(0.0, 0.3, 0.1), (0.1, 1.0, 0.4), 1, None], # Rojos/Naranjas oscuros
    "Rojo":     [(0.95, 0.4, 0.1), (1.0, 1.0, 0.8), 2, None], # Rojos (incluye cruce de 0), min_V ajustado
    "Naranja":  [(0.05, 0.5, 0.4), (0.15, 1.0, 0.9), 3, None],
    "Amarillo": [(0.15, 0.4, 0.3), (0.22, 1.0, 1.0), 4, None], # min_V ajustado
    "Verde":    [(0.25, 0.4, 0.2), (0.40, 1.0, 0.8), 5, None],
    "Azul":     [(0.55, 0.4, 0.2), (0.70, 1.0, 0.8), 6, None],
    "Violeta":  [(0.70, 0.3, 0.2), (0.85, 1.0, 0.7), 7, None],
    "Gris":     [(0.0, 0.0, 0.2), (1.0, 0.2, 0.8), 8, None], # Baja saturación
    "Blanco":   [(0.0, 0.0, 0.8), (1.0, 0.2, 1.0), 9, None], # Alta luminosidad, baja saturación
    "Dorado":   [(0.12, 0.4, 0.3), (0.18, 0.8, 1.0), None, 0.05], # Entre amarillo y naranja, brillante, min_V ajustado
    "Plateado": [(0.0, 0.0, 0.5), (1.0, 0.1, 0.9), None, 0.10] # Más claro que gris, muy baja saturación
}

def color_rgb_a_hsv(r, g, b):
    """Convierte un color RGB (0-255) a HSV (0.0-1.0)."""
    # Normalizar RGB a 0-1.0 antes de la conversión
    r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0
    h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
    return (h, s, v)

def color_mas_cercano_hsv_rango(color_rgb):
    """
    Encuentra el color de resistencia más cercano a un color RGB dado,
    usando rangos HSV.
    """
    h_detectado, s_detectado, v_detectado = color_rgb_a_hsv(*color_rgb)

    color_nombre = "Desconocido"
    valor_banda = -1
    tolerancia_banda = None
    
    # Primero, buscar coincidencias exactas dentro de los rangos
    for nombre, (min_hsv, max_hsv, val, tol) in COLORES_RESISTENCIA_HSV.items():
        min_h, min_s, min_v = min_hsv
        max_h, max_s, max_v = max_hsv

        # Manejo especial para el tono rojo que cruza el 0/360
        if min_h > max_h: # El rango de tono cruza el 0/360 grados
            h_in_range = (h_detectado >= min_h or h_detectado <= max_h)
        else:
            h_in_range = (h_detectado >= min_h and h_detectado <= max_h)

        if (h_in_range and
            s_detectado >= min_s and s_detectado <= max_s and
            v_detectado >= min_v and v_detectado <= max_v):
            return nombre, val, tol
            
    # Si no hay una coincidencia de rango perfecta, se puede añadir una lógica de fallback
    # (por ejemplo, volver a la distancia euclidiana RGB o HSV si no se encuentra nada)
    # Por ahora, si no coincide con ningún rango, retorna "Desconocido"
    return color_nombre, valor_banda, tolerancia_banda


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
        img_np = resize_image(img_np)
        st.subheader("📏 Imagen redimensionada")
        st.image(img_np, use_column_width=True)
    
    # Aplicar corrección gamma para aclarar/oscurecer la imagen
    # Usando la función gama del usuario
    img_np_gamma_corrected = gama(img_np, gamma=gamma_value)
    st.subheader(f"☀️ Imagen con Corrección Gamma (gamma={gamma_value})")
    st.image(img_np_gamma_corrected, use_column_width=True)


    # Convertir la imagen a escala de grises (usando la imagen gamma corregida)
    gray_img = rgb2gray(img_np_gamma_corrected)
    st.subheader("⚫ Imagen en escala de grises")
    # Normalizar a [0,1] para mostrarla correctamente en Streamlit
    gray_img_norm = (gray_img - gray_img.min()) / (gray_img.max() - gray_img.min())
    st.image(gray_img_norm, use_column_width=True, clamp=True)

    # Aplicar filtro bilateral para suavizar la imagen manteniendo los bordes
    gray_suavizada = filtro_bilateral(gray_img, sigma_s=5, sigma_r=20, tamaño_ventana=5)
    st.subheader("🪄 Imagen suavizada (filtro bilateral)")
    # Normalizar la imagen suavizada para mostrarla bien
    gray_suavizada_norm = (gray_suavizada - gray_suavizada.min()) / (gray_suavizada.max() - gray_suavizada.min())
    st.image(gray_suavizada_norm, use_column_width=True, clamp=True)

    # ----------- Paso 3: Detección de bandas -----------

    st.subheader("📊 Perfil de intensidad por columna")

    # Promediar intensidades por columna (eje vertical) para obtener un perfil horizontal
    perfil_columnas = np.mean(gray_suavizada, axis=0)

    # Mostrar el perfil como un gráfico de líneas
    st.line_chart(perfil_columnas)

    # Calculamos la diferencia absoluta entre columnas adyacentes (gradiente)
    # Esto ayuda a detectar cambios abruptos de intensidad que indican bordes de banda.
    gradiente = np.abs(np.diff(perfil_columnas))

    # Umbral empírico para detectar cambios significativos (ajustable por slider)
    umbral = np.mean(gradiente) + sigma_factor * np.std(gradiente)
    posibles_bordes = np.where(gradiente > umbral)[0]

    st.text(f"Posibles bordes detectados en columnas: {posibles_bordes}")

    st.subheader("🧠 Agrupando bandas detectadas")

    # Agrupamos bordes cercanos para formar bandas cohesivas (distancia ajustable por slider)
    agrupados = []
    grupo_actual = [posibles_bordes[0]] if len(posibles_bordes) > 0 else []

    for idx in posibles_bordes[1:]:
        if idx - grupo_actual[-1] < distancia_agrupacion: # Usar el slider aquí
            grupo_actual.append(idx)
        else:
            agrupados.append(grupo_actual)
            grupo_actual = [idx]
    if grupo_actual:
        agrupados.append(grupo_actual)

    # Tomamos el centro de cada grupo como la "posición de la banda"
    posiciones_bandas_original_detection = [int(np.mean(g)) for g in agrupados]
    st.text(f"Columnas centrales de bandas detectadas (orden inicial): {posiciones_bandas_original_detection}")

    st.subheader("🎨 Lectura de colores de bandas (detectadas originalmente)")

    # Lista para almacenar la información completa de cada banda
    bandas_info = []
    alto, ancho, _ = img_np_gamma_corrected.shape # Usar la imagen gamma corregida para leer colores

    # Iterar sobre las posiciones detectadas para extraer el color promedio de cada banda
    for col in posiciones_bandas_original_detection:
        fila_central = alto // 2
        # Tomamos una pequeña franja alrededor del píxel central para promediar el color
        fila_izq = max(0, fila_central - 2)
        fila_der = min(alto, fila_central + 3)
        col_izq = max(0, col - 2)
        col_der = min(ancho, col + 3)

        franja = img_np_gamma_corrected[fila_izq:fila_der, col_izq:col_der, :] # Leer de la imagen gamma corregida
        color_promedio_rgb = np.mean(franja.reshape(-1, 3), axis=0).astype(int) # Promedio de RGB en la franja
        
        # Encontrar el color de resistencia más cercano usando la nueva función de rango HSV
        nombre_color, valor_numerico, tolerancia_porcentaje = color_mas_cercano_hsv_rango(color_promedio_rgb)
        
        bandas_info.append({
            'posicion': col, # Posición de la banda en la imagen
            'rgb': color_promedio_rgb, # Color RGB detectado
            'nombre': nombre_color, # Nombre del color de resistencia
            'valor': valor_numerico, # Valor numérico de la banda
            'tolerancia': tolerancia_porcentaje # Tolerancia si es banda de tolerancia
        })

    # Mostrar los colores detectados en su orden original
    for i, band_data in enumerate(bandas_info):
        r, g, b = band_data['rgb']
        hex_color = '#%02x%02x%02x' % (r, g, b)
        st.markdown(f"**Banda {i+1}** (detectada en columna {band_data['posicion']}): Color promedio RGB = ({r}, {g}, {b})", unsafe_allow_html=True)
        st.markdown(f'<div style="width:100px;height:30px;background-color:{hex_color};border:1px solid #000;"></div>', unsafe_allow_html=True)

    st.markdown("---") # Separador visual

    # --- Invertir automáticamente el orden si detecta una banda dorada o plateada al final ---
    if len(bandas_info) > 0:
        # Se verifica si la última banda es de tolerancia (Dorado o Plateado)
        last_band_name = bandas_info[-1]['nombre']
        if last_band_name in ["Dorado", "Plateado"]:
            st.warning("Se detectó una banda de tolerancia (Dorado/Plateado) al final. Invirtiendo el orden de las bandas para la lectura correcta.")
            bandas_info.reverse() # Invertir el orden de toda la lista de información de bandas

    st.subheader("🔢 Interpretación de bandas (orden ajustado)")

    # Mostrar las bandas interpretadas en su orden final (potencialmente invertido)
    numeric_values_for_calc = [] # Solo para los valores numéricos usados en el cálculo de la resistencia
    resistencia_tolerancia = None # Variable para almacenar la tolerancia si existe

    for i, band_data in enumerate(bandas_info):
        if band_data['nombre'] in ["Dorado", "Plateado"]:
            st.markdown(f"**Banda {i+1}**: {band_data['nombre']} → Tolerancia: $\pm {band_data['tolerancia']*100}\%$")
            resistencia_tolerancia = band_data['tolerancia']
        else:
            st.markdown(f"**Banda {i+1}**: {band_data['nombre']} → Valor: {band_data['valor']}")
            if band_data['valor'] is not None: # Solo añadir valores numéricos válidos
                numeric_values_for_calc.append(band_data['valor'])

    # Calcular el valor de la resistencia si se detectaron suficientes bandas numéricas
    if len(numeric_values_for_calc) >= 3:
        # La fórmula asume que las primeras dos bandas son dígitos y la tercera es el multiplicador.
        # Esto es adecuado para resistencias de 3 bandas de valor + 1 de tolerancia o más.
        valor_base = int(str(numeric_values_for_calc[0]) + str(numeric_values_for_calc[1]))
        multiplicador = 10 ** numeric_values_for_calc[2]
        val_ohmios = valor_base * multiplicador
        
        # Mostrar el valor con la tolerancia si se detectó
        if resistencia_tolerancia is not None:
            st.success(f"💡 Valor estimado de la resistencia: **{val_ohmios} Ω** con tolerancia de $\pm {resistencia_tolerancia*100}\\%$")
        else:
            st.success(f"💡 Valor estimado de la resistencia: **{val_ohmios} Ω**")
    else:
        st.warning("No se detectaron suficientes bandas numéricas para estimar el valor (se requieren al menos 3 para el cálculo actual).")

    # --- Visualización de bandas detectadas sobre la imagen original ---
    st.subheader("✨ Visualización de bandas detectadas")

    # Crear una copia de la imagen numpy (la gamma corregida) y convertirla a un objeto PIL Image para dibujar
    img_with_bands = Image.fromarray(img_np_gamma_corrected.copy().astype(np.uint8))
    draw = ImageDraw.Draw(img_with_bands)

    # Dibujar líneas rojas en las posiciones de las bandas detectadas (usando el orden FINAL)
    for band_data in bandas_info:
        col = band_data['posicion']
        # Dibujar una línea vertical roja para cada banda detectada
        draw.line((col, 0, col, img_np_gamma_corrected.shape[0]), fill=(255, 0, 0), width=3) # Ancho 3 para mayor visibilidad

    st.image(img_with_bands, use_column_width=True, caption="Imagen con bandas detectadas resaltadas en rojo (orden ajustado)")
