import colorsys # Para convertir RGB a HSV
import math
import numpy as np
from PIL import Image

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
