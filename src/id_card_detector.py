"""
ID Card Detector Module
Detecta si una imagen contiene un documento de identidad válido usando OpenCV.

Características verificadas:
- Presencia de rectángulos con proporciones de documento ID
- Detección de bordes y contornos característicos
- Análisis de características texturales (presencia de texto)
- Validación de color y contraste
"""

import cv2
import numpy as np
import logging


class IDCardDetector:
    """Detector de documentos de identidad en imágenes."""

    def __init__(self):
        """Inicializa el detector de cédulas."""
        self.logger = logging.getLogger(__name__)

        # Proporciones típicas de documentos ID (ancho/alto)
        # ISO/IEC 7810 ID-1: 85.60 × 53.98 mm ≈ 1.586
        # Cédulas pueden variar entre 1.4 - 1.8
        self.min_aspect_ratio = 1.3
        self.max_aspect_ratio = 1.9

        # Tamaño mínimo del documento en la imagen (% del área total)
        self.min_area_ratio = 0.05

        # Umbral de confianza para considerar válido
        self.confidence_threshold = 0.4

        self.logger.info("IDCardDetector inicializado correctamente")

    def detect(self, image):
        """
        Detecta si la imagen contiene un documento de identidad válido.

        Args:
            image: numpy array (BGR) de la imagen o ruta al archivo

        Returns:
            dict: {
                'is_valid': bool,
                'confidence': float (0-1),
                'details': dict con información adicional
            }
        """
        # Cargar imagen si es una ruta
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                return {
                    'is_valid': False,
                    'confidence': 0.0,
                    'details': {'error': 'No se pudo cargar la imagen'}
                }
        else:
            img = image.copy()

        # Validaciones básicas
        if img is None or img.size == 0:
            return {
                'is_valid': False,
                'confidence': 0.0,
                'details': {'error': 'Imagen vacía o inválida'}
            }

        height, width = img.shape[:2]
        total_area = height * width

        # Lista de características detectadas
        features = {
            'rectangular_shape': 0.0,
            'correct_proportions': 0.0,
            'text_presence': 0.0,
            'edges_quality': 0.0,
            'color_consistency': 0.0
        }

        try:
            # 1. Detección de contornos rectangulares
            rect_score = self._detect_rectangular_shape(img)
            features['rectangular_shape'] = rect_score

            # 2. Verificar proporciones de documento
            prop_score = self._check_proportions(img)
            features['correct_proportions'] = prop_score

            # 3. Detectar presencia de texto (característico de IDs)
            text_score = self._detect_text_regions(img)
            features['text_presence'] = text_score

            # 4. Calidad de bordes (documentos tienen bordes definidos)
            edge_score = self._analyze_edge_quality(img)
            features['edges_quality'] = edge_score

            # 5. Consistencia de color (documentos tienen regiones uniformes)
            color_score = self._analyze_color_consistency(img)
            features['color_consistency'] = color_score

            # Calcular confianza ponderada
            weights = {
                'rectangular_shape': 0.30,
                'correct_proportions': 0.25,
                'text_presence': 0.20,
                'edges_quality': 0.15,
                'color_consistency': 0.10
            }

            confidence = sum(features[k] * weights[k] for k in features)
            is_valid = confidence >= self.confidence_threshold

            self.logger.info(f"ID Card Detection - Confidence: {confidence:.2f}, Valid: {is_valid}")

            # Convertir todos los valores a tipos JSON serializables
            return {
                'is_valid': bool(is_valid),
                'confidence': float(confidence),
                'details': {
                    'features': {k: float(v) for k, v in features.items()},
                    'threshold': float(self.confidence_threshold)
                }
            }

        except Exception as e:
            self.logger.error(f"Error en detección de ID card: {str(e)}")
            return {
                'is_valid': False,
                'confidence': 0.0,
                'details': {'error': str(e)}
            }

    def _detect_rectangular_shape(self, img):
        """Detecta formas rectangulares característicos de documentos."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Encontrar contornos
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0.0

        height, width = img.shape[:2]
        total_area = height * width
        max_score = 0.0

        for contour in contours:
            # Aproximar contorno a polígono
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # Verificar si es un rectángulo (4 vértices)
            if len(approx) == 4:
                area = cv2.contourArea(contour)
                area_ratio = area / total_area

                # Verificar que ocupe un tamaño razonable
                if area_ratio >= self.min_area_ratio:
                    # Calcular rectángulo delimitador
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = w / h if h > 0 else 0

                    # Verificar proporciones de documento
                    if self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio:
                        score = min(1.0, area_ratio * 2)  # Mayor área = mayor score
                        max_score = max(max_score, score)

        return max_score

    def _check_proportions(self, img):
        """Verifica si la imagen tiene proporciones de documento ID."""
        height, width = img.shape[:2]
        aspect_ratio = width / height if height > 0 else 0

        # Verificar si está dentro del rango aceptable
        if self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio:
            # Score basado en qué tan cerca está del ratio ideal (1.586)
            ideal_ratio = 1.586
            deviation = abs(aspect_ratio - ideal_ratio) / ideal_ratio
            score = max(0.0, 1.0 - deviation)
            return score

        return 0.0

    def _detect_text_regions(self, img):
        """Detecta regiones con texto, característico de documentos ID."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detectar gradientes horizontales (típicos de líneas de texto)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Magnitud del gradiente
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        magnitude = np.uint8(magnitude / magnitude.max() * 255)

        # Umbralizar para detectar regiones con texto
        _, thresh = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)

        # Contar píxeles activos (posibles regiones de texto)
        text_pixels = np.sum(thresh > 0)
        total_pixels = thresh.size
        text_ratio = text_pixels / total_pixels

        # Normalizar score (documentos típicamente tienen 10-30% de texto)
        if 0.05 <= text_ratio <= 0.40:
            score = min(1.0, text_ratio * 3)
        else:
            score = 0.3  # Penalizar pero no descartar completamente

        return score

    def _analyze_edge_quality(self, img):
        """Analiza la calidad de los bordes (documentos tienen bordes definidos)."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Contar píxeles de borde
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.size
        edge_ratio = edge_pixels / total_pixels

        # Documentos bien iluminados tienen bordes definidos (2-10%)
        if 0.01 <= edge_ratio <= 0.15:
            score = min(1.0, edge_ratio * 8)
        else:
            score = 0.3

        return score

    def _analyze_color_consistency(self, img):
        """Analiza la consistencia de color (documentos tienen regiones uniformes)."""
        # Convertir a HSV para mejor análisis de color
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Calcular desviación estándar de matiz y saturación
        h_std = np.std(hsv[:, :, 0])
        s_std = np.std(hsv[:, :, 1])

        # Documentos tienen regiones con color consistente
        # Penalizar variaciones extremas
        consistency_score = 1.0 - min(1.0, (h_std + s_std) / 200)

        return max(0.0, consistency_score)


# Función de utilidad para testing
def test_id_card_detector(image_path):
    """Función de prueba para el detector."""
    detector = IDCardDetector()
    result = detector.detect(image_path)

    print(f"\n=== ID Card Detection Result ===")
    print(f"Valid: {result['is_valid']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"\nFeature Scores:")
    for feature, score in result['details']['features'].items():
        print(f"  {feature}: {score:.2%}")

    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_id_card_detector(sys.argv[1])
    else:
        print("Uso: python id_card_detector.py <ruta_imagen>")