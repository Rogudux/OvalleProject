🔍 Escáner de Resistencias
Este proyecto implementa un escáner de resistencias interactivo utilizando Streamlit y procesamiento de imágenes con NumPy y PIL. Permite a los usuarios subir una imagen de una resistencia, detectar sus bandas de color y calcular su valor en ohmios, incluyendo la tolerancia.

✨ Características Principales
Procesamiento de Imagen:

Redimensionamiento automático de imágenes para un procesamiento eficiente.

Conversión a escala de grises.

Aplicación de filtro bilateral para suavizado y preservación de bordes.

Corrección Gamma: Ajuste del brillo de la imagen para mejorar la visibilidad de los colores oscuros.

Detección de Bandas:

Análisis del perfil de intensidad de la resistencia por columna para identificar cambios bruscos que indican bandas.

Agrupamiento de bordes cercanos para delimitar las bandas.

Visualización de las bandas detectadas sobre la imagen original mediante líneas rojas.

Lectura e Interpretación de Colores:

Extracción del color promedio de cada banda.

Detección de Colores basada en Rangos HSV: Clasificación de los colores de las bandas utilizando rangos definidos en el espacio de color HSV, lo que mejora la precisión bajo diversas condiciones de iluminación.

Inversión automática del orden de las bandas si se detecta una banda de tolerancia (Dorado o Plateado) al final.

Cálculo del valor de la resistencia y su tolerancia.

Interfaz Interactiva con Streamlit:

Interfaz amigable para subir imágenes.

Gráfico del perfil de intensidad para visualización del proceso.

Parámetros Ajustables: Sliders en la barra lateral para afinar el factor del umbral de detección, la distancia de agrupación de bandas y el valor gamma, permitiendo al usuario optimizar los resultados.

🚀 Cómo Usar
Guarda el código Python proporcionado (por ejemplo, resistor_scanner.py).

Abre una terminal o línea de comandos.

Navega al directorio donde guardaste el archivo.

Ejecuta la aplicación Streamlit con el siguiente comando:

streamlit run resistor_scanner.py

Se abrirá una pestaña en tu navegador web con la aplicación.

Sube una imagen de una resistencia utilizando el botón "Sube una imagen de una resistencia".

Observa los resultados en pantalla: la imagen original, la imagen procesada, el perfil de intensidad, los colores detectados y el valor de la resistencia.

Ajusta los "Parámetros de Detección" en la barra lateral (deslizable desde la izquierda) para mejorar la precisión de la detección de bandas y colores.

📸 Guía para Tomar Imágenes de Resistencias
Para obtener los mejores resultados con el escáner, se recomienda seguir estas directrices al tomar las fotografías de las resistencias:

Distancia y Zoom: Tomar la imagen a aproximadamente 13 cm de distancia de la resistencia, preferiblemente con un zoom de 5x. Esto ayuda a capturar la resistencia con suficiente detalle y resolución.

Perspectiva Perpendicular: Asegurar que la imagen se tome desde una perspectiva perpendicular a la resistencia. Evitar ángulos oblicuos que puedan distorsionar las bandas.

Iluminación Uniforme: Es crucial garantizar una iluminación uniforme sobre la resistencia. Evitar sombras fuertes, reflejos directos o puntos de luz excesivos que puedan alterar los colores reales de las bandas.

Claridad de la Imagen: Las imágenes deben ser claras y nítidas. Evitar imágenes borrosas o aquellas en las que las bandas estén parcialmente oscurecidas o cortadas.
