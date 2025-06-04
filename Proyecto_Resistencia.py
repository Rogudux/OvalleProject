import streamlit as st
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Configuración de la página
st.set_page_config(page_title="Detector de Resistencias (5 Bandas)", page_icon="🔬")
st.title("🔬 Detector de Código de Colores para Resistencias (5 Bandas)")
st.write("Carga una imagen de una resistencia de 5 bandas para identificar su valor")

# Diccionario de colores y valores (actualizado para 5 bandas)
COLOR_CODES = {
    "black": (0, 0, 0),
    "brown": (101, 69, 31),
    "red": (255, 0, 0),
    "orange": (255, 165, 0),
    "yellow": (255, 255, 0),
    "green": (0, 128, 0),
    "blue": (0, 0, 255),
    "violet": (238, 130, 238),
    "gray": (128, 128, 128),
    "white": (255, 255, 255),
    "gold": (212, 175, 55),
    "silver": (192, 192, 192)
}

COLOR_VALUES = {
    "black": 0,
    "brown": 1,
    "red": 2,
    "orange": 3,
    "yellow": 4,
    "green": 5,
    "blue": 6,
    "violet": 7,
    "gray": 8,
    "white": 9
}

MULTIPLIER = {
    "black": 1,
    "brown": 10,
    "red": 100,
    "orange": 1000,
    "yellow": 10000,
    "green": 100000,
    "blue": 1000000,
    "gold": 0.1,
    "silver": 0.01
}

TOLERANCE = {
    "brown": "±1%",
    "red": "±2%",
    "green": "±0.5%",
    "blue": "±0.25%",
    "violet": "±0.1%",
    "gray": "±0.05%",
    "gold": "±5%",
    "silver": "±10%"
}

# Rangos de colores en LAB (Lightness, A, B)
COLOR_RANGES_LAB = {
    "black": ([0, -20, -20], [50, 20, 20]),
    "brown": ([30, 10, 40], [70, 40, 80]),
    "red": ([40, 60, 30], [80, 100, 60]),
    "orange": ([60, 20, 60], [90, 50, 90]),
    "yellow": ([80, -20, 70], [95, 0, 90]),
    "green": ([50, -70, 0], [80, -30, 40]),
    "blue": ([40, -10, -60], [70, 10, -20]),
    "violet": ([50, 30, -40], [80, 60, -10]),
    "gray": ([50, -5, -5], [90, 5, 5]),
    "white": ([90, -5, -5], [100, 5, 5]),
    "gold": ([70, 0, 30], [85, 15, 50]),
    "silver": ([75, -5, -5], [90, 5, 5])
}

def preprocess_image(image):
    """Aplica preprocesamiento fijo a la imagen"""
    # Reducir ruido con filtro de mediana
    processed = cv2.medianBlur(image, 5)
    
    # Ajustar contraste con CLAHE
    lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return processed

def crop_resistance_area(image):
    """Recorta el área central donde se espera la resistencia"""
    height, width = image.shape[:2]
    
    # Parámetros basados en cámara iPhone 13 Pro Max a 30cm con zoom 3.6x
    crop_width = int(width * 0.7)   # 70% del ancho
    crop_height = int(height * 0.4) # 40% del alto
    
    # Calcular coordenadas de recorte (centrado)
    x = (width - crop_width) // 2
    y = (height - crop_height) // 2
    
    return image[y:y+crop_height, x:x+crop_width]

def detect_bands(image):
    """Detecta 5 bandas de colores en la resistencia"""
    # Convertir a espacio de color LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Proyección horizontal para detectar bandas
    projection = np.mean(lab, axis=0)
    
    # Calcular la derivada para encontrar transiciones
    diff = np.diff(projection, axis=0)
    diff_norm = np.linalg.norm(diff, axis=1)
    
    # Suavizar la señal
    kernel_size = max(5, int(image.shape[1] * 0.05))
    kernel = np.ones(kernel_size) / kernel_size
    smoothed_diff = np.convolve(diff_norm, kernel, mode='same')
    
    # Encontrar picos significativos (umbral adaptativo)
    mean_val = np.mean(smoothed_diff)
    std_val = np.std(smoothed_diff)
    threshold = mean_val + 3 * std_val  # Más estricto para 5 bandas
    peaks = np.where(smoothed_diff > threshold)[0]
    
    # Agrupar picos cercanos
    band_edges = []
    if len(peaks) > 0:
        band_edges.append(peaks[0])
        for i in range(1, len(peaks)):
            if peaks[i] - band_edges[-1] > image.shape[1] * 0.04:  # Separación mínima reducida
                band_edges.append(peaks[i])
    
    # Filtrar para obtener exactamente 5 bandas
    if len(band_edges) >= 5:
        # Tomar las 5 bandas con mayores transiciones
        peak_vals = smoothed_diff[peaks]
        sorted_indices = np.argsort(peak_vals)[::-1][:5]
        band_positions = sorted([peaks[i] for i in sorted_indices])
    else:
        # Usar posiciones equidistantes forzadas para 5 bandas
        st.warning("No se detectaron 5 bandas claramente. Usando aproximación.")
        band_positions = np.linspace(image.shape[1]*0.1, image.shape[1]*0.9, 5, dtype=int)
    
    return band_positions

def classify_color(lab_value):
    """Clasifica un color basado en el valor LAB"""
    L, A, B = lab_value
    min_dist = float('inf')
    color_name = "unknown"
    
    for color, (lower, upper) in COLOR_RANGES_LAB.items():
        # Calcular el centro del rango
        center = np.array([
            (lower[0] + upper[0]) / 2, 
            (lower[1] + upper[1]) / 2, 
            (lower[2] + upper[2]) / 2
        ])
        
        # Ponderar más el canal de luminosidad
        weights = np.array([0.5, 0.25, 0.25])
        # Calcular la diferencia ponderada
        diferencia = np.array([L, A, B]) - center
        diferencia_ponderada = diferencia * weights
        dist = np.linalg.norm(diferencia_ponderada)
        
        if dist < min_dist:
            min_dist = dist
            color_name = color
            
    return color_name

def calculate_resistance(colors):
    """Calcula el valor de la resistencia basado en 5 bandas"""
    if len(colors) < 5:
        return "Error: No suficientes bandas detectadas"
    
    try:
        # Primeras tres bandas: dígitos
        digit1 = COLOR_VALUES.get(colors[0], 0)
        digit2 = COLOR_VALUES.get(colors[1], 0)
        digit3 = COLOR_VALUES.get(colors[2], 0)
        
        # Cuarta banda: multiplicador
        multiplier_val = MULTIPLIER.get(colors[3], 1)
        
        # Quinta banda: tolerancia
        tolerance_val = TOLERANCE.get(colors[4], "±20%")
        
        # Calcular valor
        value = (digit1 * 100 + digit2 * 10 + digit3) * multiplier_val
        
        # Formatear resultado
        if value >= 1e6:
            result = f"{value/1e6:.2f} MΩ {tolerance_val}"
        elif value >= 1e3:
            result = f"{value/1e3:.2f} KΩ {tolerance_val}"
        else:
            result = f"{value:.2f} Ω {tolerance_val}"
            
        return result
    except Exception as e:
        return f"Error: {str(e)}"

# Interfaz de usuario
uploaded_file = st.file_uploader("Sube una imagen de la resistencia de 5 bandas", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Leer y procesar imagen
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Paso 1: Recorte inteligente del área de resistencia
    cropped_image = crop_resistance_area(original_image)
    
    # Paso 2: Preprocesamiento fijo
    processed_image = preprocess_image(cropped_image)
    
    # Paso 3: Detectar bandas
    band_positions = detect_bands(processed_image)
    
    # Paso 4: Clasificar colores
    detected_colors = []
    lab_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2LAB)
    
    for pos in band_positions:
        # Obtener muestra de color en la posición de la banda
        sample_region = lab_image[:, max(0, pos-5):min(pos+5, lab_image.shape[1]-1)]
        avg_color = np.median(sample_region, axis=(0, 1))
        color_name = classify_color(avg_color)
        detected_colors.append(color_name)
    
    # Calcular valor de resistencia
    resistance_value = calculate_resistance(detected_colors)
    
    # Mostrar resultados
    st.subheader("Imagen Original")
    st.image(original_image, channels="BGR", use_container_width=True)
    
    st.subheader("Área de Resistencia Recortada")
    st.image(cropped_image, channels="BGR", use_container_width=True)
    
    st.subheader("Imagen Procesada")
    st.image(processed_image, channels="BGR", use_container_width=True)
    
    # Dibujar bandas detectadas
    result_image = processed_image.copy()
    for pos in band_positions:
        cv2.line(result_image, (pos, 0), (pos, result_image.shape[0]), (0, 255, 0), 2)
    
    st.subheader("Bandas Detectadas")
    st.image(result_image, channels="BGR", use_container_width=True)
    
    # Mostrar colores detectados
    st.subheader("Resultados de Análisis")
    fig, ax = plt.subplots(figsize=(10, 2))
    for i, color_name in enumerate(detected_colors):
        color_rgb = COLOR_CODES.get(color_name, (128, 128, 128))
        color_rgb = tuple(c / 255 for c in color_rgb)
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color_rgb))
        ax.text(i + 0.5, 0.5, color_name, ha='center', va='center', fontsize=12, 
                color='white' if color_name in ['black', 'blue', 'brown'] else 'black')
    
    ax.set_xlim(0, len(detected_colors))
    ax.set_ylim(0, 1)
    ax.axis('off')
    st.pyplot(fig)
    
    # Mostrar valor de resistencia
    st.success(f"**Valor de la resistencia:** {resistance_value}")
    
    # Mostrar detalles técnicos
    with st.expander("Detalles de detección"):
        st.write(f"**Posiciones de bandas detectadas:** {band_positions}")
        st.write("**Colores identificados:**")
        for i, color in enumerate(detected_colors, 1):
            st.write(f"Banda {i}: {color}")
        
        st.write("**Espacio de color LAB promedio:**")
        for i, pos in enumerate(band_positions):
            sample_region = lab_image[:, max(0, pos-5):min(pos+5, lab_image.shape[1]-1)]
            avg_lab = np.median(sample_region, axis=(0, 1))
            st.write(f"Banda {i+1}: L={avg_lab[0]:.1f}, A={avg_lab[1]:.1f}, B={avg_lab[2]:.1f}")

# Plantilla para guiar al usuario
st.subheader("Instrucciones de Captura (iPhone 13 Pro Max)")
st.write("Para obtener los mejores resultados, siga estas indicaciones:")

# Crear una plantilla visual
template = np.zeros((300, 700, 3), dtype=np.uint8)
template[:] = (240, 240, 240)  # Fondo gris claro

# Dibujar guía de captura
cv2.putText(template, "DISTANCIA: 30 cm", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
cv2.putText(template, "ZOOM: 3.6x", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
cv2.putText(template, "ORIENTACION: Horizontal", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

# Dibujar marco de captura
cv2.rectangle(template, (100, 150), (600, 250), (0, 0, 0), 3)
cv2.putText(template, "Zona de captura", (250, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

# Dibujar resistencia de ejemplo dentro del marco
cv2.rectangle(template, (150, 180), (550, 220), (200, 200, 200), -1)
colors = [(0, 0, 0), (101,69,31), (255,0,0), (255,165,0), (0,0,255)]  # Negro, marrón, rojo, naranja, azul
positions = [180, 240, 300, 360, 420]
for color, pos in zip(colors, positions):
    cv2.rectangle(template, (pos-8, 175), (pos+8, 225), color, -1)

st.image(template, use_container_width=True)

# Instrucciones en el sidebar
st.sidebar.header("Configuración Recomendada")
st.sidebar.info("""
**Cámara iPhone 13 Pro Max:**
- Usar lente teleobjetivo (3x)
- Distancia: 30 cm
- Zoom digital: 3.6x
- Enfoque manual en la resistencia
- Flash: Solo si necesario (evitar reflejos)
""")

st.sidebar.header("Posicionamiento")
st.sidebar.info("""
1. **Orientación:**
   - Resistencia horizontal
   - Bandas verticales

2. **Fondo:**
   - Preferiblemente blanco
   - Sin patrones complejos

3. **Iluminación:**
   - Luz natural preferible
   - Evitar sombras sobre la resistencia
""")

st.sidebar.header("Codificación de 5 Bandas")
st.sidebar.markdown("""
1. **1ra Banda:** 1er dígito  
2. **2da Banda:** 2do dígito  
3. **3ra Banda:** 3er dígito  
4. **4ta Banda:** Multiplicador  
5. **5ta Banda:** Tolerancia  
""")
st.sidebar.image("https://www.electronics-tutorials.ws/wp-content/uploads/2018/05/resistor-res9.png", 
                 use_container_width=True)