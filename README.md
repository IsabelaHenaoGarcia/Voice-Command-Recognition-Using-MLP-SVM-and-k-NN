# Reconocimiento de Comandos de Voz mediante MLP, SVM y kNN

Este proyecto implementa y compara tres modelos de aprendizaje automático (MLP, SVM y kNN) para la clasificación de comandos de voz simples ("yes", "no", "up", "down"). La red neuronal (MLP) está implementada desde cero en NumPy (incluyendo el paso hacia adelante, propagación hacia atrás y descenso de gradiente), compartiendo un mismo pipeline de extracción de características espectrales (MFCC).

---

## Estructura y Pipeline del Proyecto

1. Dataset (Google Speech Commands v2):
   - Subconjunto de 4 clases: yes, no, up, down (250 muestras por clase, 1,000 archivos .wav en total).
   - División de datos: 70% Entrenamiento, 15% Validación, 15% Evaluación (Test).

2. Preprocesamiento y Extracción de Características:
   - Filtro de pre-énfasis (alpha = 0.97) para realzar las frecuencias altas del habla.
   - Coeficientes MFCC: 40 coeficientes Mel, ventana FFT de 512, paso de trama (hop) de 160 y longitud de ventana de 400.
   - Relleno/Truncado (Padding/Truncating): Normalizado a 98 tramas temporales (~1s a 16kHz).
   - Estandarización y Aplanamiento (Flatten): Vector estandarizado por coeficiente de dimensión 40 x 98 = 3,920 características por muestra.

3. Modelos Evaluados:
   - MLP (Perceptrón Multicapa):
     - Arquitectura: Entrada (3,920) -> FC (256, ReLU) -> FC (128, ReLU) -> FC (4, Softmax).
     - Pérdida: Entropía Cruzada (Cross-Entropy) con Regularización L2 (lambda = 1e-4).
     - Optimización: Descenso de Gradiente Estocástico por minibaches (SGD, batch size = 128, learning rate = 5e-3).
     - Inicialización: Inicialización He para capas con ReLU.
   - SVM (Support Vector Machine):
     - Kernel RBF con optimización de hiperparámetros vía Búsqueda en Malla (Grid Search sobre C y gamma).
   - kNN (k-Nearest Neighbors):
     - Búsqueda de k óptimo (k en [3, 5, 7, 9, 11, 15]) con métrica de distancia euclidiana.

---

## Resultados y Comparativa

Los modelos se evaluaron sobre el mismo conjunto de prueba (Test set):

| Modelo | Accuracy (%) | Tiempo de Entrenamiento | Ventajas | Desventajas |
| :--- | :---: | :---: | :--- | :--- |
| SVM (RBF) | 74.67% | ~8.45 s | Robusto con conjuntos reducidos y alta dimensión. | Tiempo de inferencia depende de los vectores de soporte. |
| MLP (NumPy) | 60.67% | ~15.23 s | Inferencia matricial ultrarrápida; aprende no linealidades. | Requiere más datos, épocas y ajuste de hiperparámetros. |
| kNN | 49.33% | 0.52 s | Cero tiempo de entrenamiento (Lazy Learning). | Sufre la maldición de la dimensionalidad; inferencia lenta. |

---

## Requisitos e Instalación

### Requisitos previos
- Python 3.8+

### Librerías necesarias
Puedes instalar las dependencias exactas ejecutando:

```bash
pip install librosa==0.10.2.post1 soundfile==0.12.1 scikit-learn==1.5.2 numpy matplotlib
