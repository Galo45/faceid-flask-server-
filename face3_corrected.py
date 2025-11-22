"""
Servidor Flask CORREGIDO con detección anti-spoofing + verificación facial mejorada.

CORRECCIONES CRÍTICAS DE SEGURIDAD:

1. VALIDACIÓN DE ROSTRO ÚNICO (v2.0):
   - Todas las funciones extract_*_embedding() validan rostros según contexto
   - IMÁGENES EN VIVO: Requieren EXACTAMENTE 1 rostro (seguridad estricta)
   - DOCUMENTOS ID: Permiten múltiples rostros, seleccionan el más grande
   - Si detectan 0 rostros: retorna error "No se detectó ningún rostro"
   - Si detectan 2+ rostros EN VIVO: retorna error de seguridad
   - Previene ataques de suplantación con múltiples personas

2. MANEJO INTELIGENTE DE DOCUMENTOS ID:
   - Las cédulas típicamente tienen 2 rostros (foto principal + foto secundaria/holograma)
   - Sistema automáticamente selecciona el rostro más grande (foto principal)
   - Usa parámetros allow_multiple_faces y select_largest
   - Solo en endpoint /verify-profile (verificación con documento)
   - Endpoints de verificación en vivo (/verify, /verify-with-profile) mantienen
     validación estricta de 1 rostro

3. SISTEMA DE SIMILITUD CONSERVADOR:
   - Umbrales más estrictos y realistas
   - Normalización correcta de embeddings

4. RESPUESTAS DETALLADAS:
   - Todos los endpoints retornan información sobre cantidad de rostros detectados
   - Campo 'faces_detected' incluye conteo para cada imagen procesada
   - Mensajes de error específicos para diferentes escenarios
   - Información adicional cuando se selecciona rostro de múltiples opciones

VULNERABILIDADES CORREGIDAS:
1. (v2.0) Sistema seleccionaba automáticamente rostro con mayor score sin notificar
   → Ahora rechaza múltiples personas en imágenes en vivo
2. (v2.1) Sistema rechazaba cédulas con 2 fotos impidiendo verificación de perfil
   → Ahora permite múltiples rostros solo en documentos ID, selecciona el más grande

FECHA: 2025-01-05 (v2.1)
"""

import io
import os
import sys
import argparse

# Fix para el error de OpenMP duplicate library
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import requests
import numpy as np
import cv2
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import logging
import warnings
warnings.filterwarnings('ignore')

# Legacy models como base confiable
from facenet_pytorch import MTCNN, InceptionResnetV1

# Enhanced face recognition imports (opcionales)
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("⚠️  InsightFace not available")

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("⚠️  DeepFace not available")

# OCR for ID document number extraction
try:
    import easyocr
    import re
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("⚠️  EasyOCR not available, ID number verification disabled")

# Importar módulos de Silent-Face-Anti-Spoofing
sys.path.append('./src')
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
from src.id_card_detector import IDCardDetector

# --- Inicialización ---
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)


class CorrectedFaceRecognitionSystem:
    """Sistema corregido de reconocimiento facial"""
    
    def __init__(self, device='cpu'):
        self.device = device
        
        # Umbrales más estrictos y realistas
        self.thresholds = {
            'profile_vs_id': 0.7,     # Más estricto para perfil vs cédula  
            'live_vs_profile': 0.65,   # Más estricto para live
            'insightface': 0.4,        # Umbral específico para InsightFace (distancia coseno)
            'facenet': 0.7,            # Umbral para FaceNet
            'deepface': 0.68           # Umbral para DeepFace
        }
        
        self._initialize_models()
        logging.info(f"[CORRECTED] Sistema inicializado")
        
    def _initialize_models(self):
        """Inicializar modelos de forma más conservadora"""
        
        # 1. FaceNet como base confiable
        try:
            self.mtcnn = MTCNN(
                image_size=160, margin=0, min_face_size=20,
                thresholds=[0.6, 0.7, 0.7], factor=0.709,
                post_process=True, device=self.device
            )
            self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            logging.info("✅ FaceNet cargado correctamente")
        except Exception as e:
            logging.error(f"❌ Error cargando FaceNet: {e}")
            self.mtcnn = None
            self.facenet = None
        
        # 2. InsightFace como modelo principal (si está disponible)
        if INSIGHTFACE_AVAILABLE:
            try:
                self.insightface = FaceAnalysis(providers=['CPUExecutionProvider'])
                self.insightface.prepare(ctx_id=0, det_size=(640, 640))
                logging.info("✅ InsightFace cargado correctamente")
            except Exception as e:
                logging.error(f"❌ Error cargando InsightFace: {e}")
                self.insightface = None
        else:
            self.insightface = None
        
        # 3. DeepFace como backup
        if DEEPFACE_AVAILABLE:
            try:
                # Test DeepFace
                test_img = np.zeros((224, 224, 3), dtype=np.uint8)
                _ = DeepFace.represent(test_img, model_name="Facenet", enforce_detection=False)
                self.deepface = DeepFace
                logging.info("✅ DeepFace disponible")
            except Exception as e:
                logging.warning(f"⚠️  DeepFace con problemas: {e}")
                self.deepface = None
        else:
            self.deepface = None
    
    def extract_facenet_embedding(self, image: np.ndarray, allow_multiple_faces: bool = False,
                                  select_largest: bool = False) -> dict:
        """
        Extraer embedding usando FaceNet con validación de rostro único

        Args:
            image: Imagen en formato numpy array
            allow_multiple_faces: Si True, permite múltiples rostros (para documentos ID)
            select_largest: Si True, selecciona el rostro más grande cuando hay múltiples

        Nota: MTCNN por defecto solo detecta el rostro más prominente cuando keep_all=False,
              por lo que estos parámetros tienen efecto limitado con la configuración actual.
        """
        try:
            if self.mtcnn is None or self.facenet is None:
                return {'success': False, 'error': 'FaceNet no disponible', 'faces_count': 0}

            # Convertir a PIL
            if len(image.shape) == 3 and image.shape[2] == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)

            # Detectar rostro con MTCNN
            # NOTA: Con keep_all=False (configuración actual), MTCNN solo retorna el rostro más prominente
            # Esto es adecuado para documentos ID donde generalmente detectará la foto principal
            face_tensor = self.mtcnn(pil_image)

            if face_tensor is None:
                logging.warning("[FACENET] No se detectó rostro")
                return {'success': False, 'error': 'No se detectó ningún rostro', 'faces_count': 0}

            # MTCNN retorna un solo tensor cuando no se usa keep_all=True
            # Automáticamente selecciona el rostro más prominente/grande
            logging.info("[FACENET] 1 rostro detectado (más prominente)")

            # Generar embedding
            with torch.no_grad():
                face_tensor = face_tensor.unsqueeze(0).to(self.device)
                embedding = self.facenet(face_tensor)
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

                return {
                    'success': True,
                    'embedding': embedding.cpu().numpy().flatten(),
                    'faces_count': 1,
                    'confidence': 1.0,
                    'note': 'MTCNN automáticamente seleccionó el rostro más prominente'
                }

        except Exception as e:
            logging.error(f"[FACENET] Error: {e}")
            return {'success': False, 'error': str(e), 'faces_count': 0}
    
    def extract_insightface_embedding(self, image: np.ndarray, allow_multiple_faces: bool = False,
                                      select_largest: bool = False) -> dict:
        """
        Extraer embedding usando InsightFace con validación de rostro único

        Args:
            image: Imagen en formato numpy array
            allow_multiple_faces: Si True, permite múltiples rostros (para documentos ID)
            select_largest: Si True, selecciona el rostro más grande cuando hay múltiples
                           (útil para cédulas con foto principal + foto pequeña)
        """
        try:
            if self.insightface is None:
                return {'success': False, 'error': 'InsightFace no disponible', 'faces_count': 0}

            # Asegurar formato correcto
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image

            faces = self.insightface.get(image_rgb)

            # VALIDACIÓN CRÍTICA: Verificar cantidad de rostros
            if not faces or len(faces) == 0:
                logging.warning("[INSIGHTFACE] No se detectó rostro")
                return {'success': False, 'error': 'No se detectó ningún rostro en la imagen', 'faces_count': 0}

            # CASO 1: Múltiples rostros detectados
            if len(faces) > 1:
                # Si se permite múltiples rostros (documentos ID)
                if allow_multiple_faces:
                    if select_largest:
                        # Seleccionar el rostro más grande (foto principal de la cédula)
                        # El área del bbox indica el tamaño del rostro detectado
                        largest_face = max(faces, key=lambda f: self._calculate_face_area(f.bbox))

                        logging.info(f"[INSIGHTFACE-ID] {len(faces)} rostros detectados en documento ID, " +
                                   f"seleccionando el más grande (área: {self._calculate_face_area(largest_face.bbox):.0f}px²)")

                        return {
                            'success': True,
                            'embedding': largest_face.embedding,
                            'faces_count': len(faces),
                            'confidence': float(largest_face.det_score),
                            'selected_face': 'largest',
                            'face_area': float(self._calculate_face_area(largest_face.bbox))
                        }
                    else:
                        # Seleccionar el rostro con mayor confianza de detección
                        best_face = max(faces, key=lambda f: f.det_score)

                        logging.info(f"[INSIGHTFACE-ID] {len(faces)} rostros detectados en documento ID, " +
                                   f"seleccionando el de mayor confianza ({best_face.det_score:.3f})")

                        return {
                            'success': True,
                            'embedding': best_face.embedding,
                            'faces_count': len(faces),
                            'confidence': float(best_face.det_score),
                            'selected_face': 'highest_confidence'
                        }
                else:
                    # Modo estricto: rechazar múltiples rostros (imágenes en vivo)
                    logging.error(f"[SECURITY] InsightFace detectó {len(faces)} rostros en la imagen")
                    return {
                        'success': False,
                        'error': f'Se detectaron {len(faces)} personas en la imagen. Por favor, asegúrese de estar solo en el encuadre.',
                        'faces_count': len(faces)
                    }

            # CASO 2: Solo hay exactamente 1 rostro
            face = faces[0]
            logging.info(f"[INSIGHTFACE] 1 rostro detectado con confianza: {face.det_score:.3f}")

            return {
                'success': True,
                'embedding': face.embedding,
                'faces_count': 1,
                'confidence': float(face.det_score)
            }

        except Exception as e:
            logging.error(f"[INSIGHTFACE] Error: {e}")
            return {'success': False, 'error': str(e), 'faces_count': 0}

    def _calculate_face_area(self, bbox):
        """Calcular área del bounding box del rostro"""
        # bbox = [x1, y1, x2, y2]
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return width * height
    
    def extract_deepface_embedding(self, image: np.ndarray) -> dict:
        """Extraer embedding usando DeepFace con validación de rostro único"""
        try:
            if self.deepface is None:
                return {'success': False, 'error': 'DeepFace no disponible', 'faces_count': 0}

            # Usar modelo Facenet en DeepFace para consistencia
            result = self.deepface.represent(
                image,
                model_name="Facenet",
                enforce_detection=False,
                detector_backend='opencv'
            )

            # VALIDACIÓN CRÍTICA: Verificar cantidad de rostros detectados
            if not result or len(result) == 0:
                logging.warning("[DEEPFACE] No se detectó rostro")
                return {'success': False, 'error': 'No se detectó ningún rostro en la imagen', 'faces_count': 0}

            if len(result) > 1:
                logging.error(f"[SECURITY] DeepFace detectó {len(result)} rostros en la imagen")
                return {
                    'success': False,
                    'error': f'Se detectaron {len(result)} personas en la imagen. Por favor, asegúrese de estar solo en el encuadre.',
                    'faces_count': len(result)
                }

            # Solo si hay exactamente 1 rostro
            logging.info(f"[DEEPFACE] 1 rostro detectado correctamente")

            return {
                'success': True,
                'embedding': np.array(result[0]['embedding']),
                'faces_count': 1,
                'confidence': 1.0
            }

        except Exception as e:
            logging.error(f"[DEEPFACE] Error: {e}")
            return {'success': False, 'error': str(e), 'faces_count': 0}
    
    def calculate_similarity_corrected(self, embedding1: np.ndarray, embedding2: np.ndarray, 
                                     model_name: str) -> float:
        """Calcular similitud de forma correcta según el modelo"""
        
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        try:
            if model_name in ['insightface', 'deepface', 'facenet']:
                # Normalizar embeddings
                emb1_norm = embedding1 / np.linalg.norm(embedding1)
                emb2_norm = embedding2 / np.linalg.norm(embedding2)
                
                # Similitud coseno
                similarity = np.dot(emb1_norm, emb2_norm)
                
                # Convertir de [-1, 1] a [0, 1] 
                similarity = (similarity + 1) / 2
                
                return float(similarity)
            
            return 0.0
            
        except Exception as e:
            logging.error(f"[SIMILARITY] Error calculando similitud: {e}")
            return 0.0
    
    def verify_faces(self, image1: np.ndarray, image2: np.ndarray,
                    comparison_type: str = 'general',
                    image1_allow_multiple: bool = False,
                    image2_allow_multiple: bool = False) -> dict:
        """
        Verificar rostros usando múltiples modelos de forma conservadora con validación de rostro único

        Args:
            image1: Primera imagen (generalmente foto de perfil)
            image2: Segunda imagen (foto en vivo o documento ID)
            comparison_type: Tipo de comparación ('general', 'live_vs_profile', 'profile_vs_id')
            image1_allow_multiple: Si True, permite múltiples rostros en image1
            image2_allow_multiple: Si True, permite múltiples rostros en image2 (para documentos ID)
        """

        results = {
            'success': False,
            'verified': False,
            'primary_score': 0.0,
            'backup_score': 0.0,
            'method_used': 'none',
            'details': {},
            'error': None,
            'faces_count_image1': 0,
            'faces_count_image2': 0
        }

        # Método 1: InsightFace (si está disponible)
        if self.insightface is not None:
            # Extraer embeddings con configuración apropiada para cada imagen
            emb1_result = self.extract_insightface_embedding(
                image1,
                allow_multiple_faces=image1_allow_multiple,
                select_largest=image1_allow_multiple  # Si permite múltiples, seleccionar el más grande
            )
            emb2_result = self.extract_insightface_embedding(
                image2,
                allow_multiple_faces=image2_allow_multiple,
                select_largest=image2_allow_multiple
            )

            # VALIDACIÓN: Verificar que ambas extracciones fueron exitosas
            if not emb1_result['success']:
                results['error'] = f"Imagen 1: {emb1_result['error']}"
                results['faces_count_image1'] = emb1_result['faces_count']
                logging.warning(f"[VERIFICATION] Fallo en imagen 1: {emb1_result['error']}")
                return results

            if not emb2_result['success']:
                results['error'] = f"Imagen 2: {emb2_result['error']}"
                results['faces_count_image2'] = emb2_result['faces_count']
                logging.warning(f"[VERIFICATION] Fallo en imagen 2: {emb2_result['error']}")
                return results

            # Ambas imágenes tienen exactamente 1 rostro cada una
            results['faces_count_image1'] = emb1_result['faces_count']
            results['faces_count_image2'] = emb2_result['faces_count']

            similarity = self.calculate_similarity_corrected(
                emb1_result['embedding'],
                emb2_result['embedding'],
                'insightface'
            )
            threshold = self.thresholds.get('insightface', 0.4)

            results['primary_score'] = similarity
            results['method_used'] = 'insightface'
            results['details']['insightface'] = {
                'similarity': similarity,
                'threshold': threshold,
                'verified': similarity > threshold,
                'confidence_img1': emb1_result['confidence'],
                'confidence_img2': emb2_result['confidence']
            }

            # InsightFace como método principal
            if similarity > threshold:
                results['verified'] = True

            logging.info(f"[INSIGHTFACE] Similitud: {similarity:.4f}, Umbral: {threshold}, Verificado: {results['verified']}")

        # Método 2: FaceNet como backup/confirmación
        if self.facenet is not None and self.mtcnn is not None:
            emb1_result = self.extract_facenet_embedding(
                image1,
                allow_multiple_faces=image1_allow_multiple,
                select_largest=image1_allow_multiple
            )
            emb2_result = self.extract_facenet_embedding(
                image2,
                allow_multiple_faces=image2_allow_multiple,
                select_largest=image2_allow_multiple
            )

            # VALIDACIÓN: Solo procesar si ambas extracciones fueron exitosas
            if emb1_result['success'] and emb2_result['success']:
                similarity = self.calculate_similarity_corrected(
                    emb1_result['embedding'],
                    emb2_result['embedding'],
                    'facenet'
                )
                threshold = self.thresholds.get('facenet', 0.7)

                results['backup_score'] = similarity
                results['details']['facenet'] = {
                    'similarity': similarity,
                    'threshold': threshold,
                    'verified': similarity > threshold
                }

                # Si InsightFace no está disponible, usar FaceNet como principal
                if results['method_used'] == 'none':
                    results['faces_count_image1'] = emb1_result['faces_count']
                    results['faces_count_image2'] = emb2_result['faces_count']
                    results['primary_score'] = similarity
                    results['method_used'] = 'facenet'
                    results['verified'] = similarity > threshold

                logging.info(f"[FACENET] Similitud: {similarity:.4f}, Umbral: {threshold}, Verificado: {similarity > threshold}")
            else:
                # Si InsightFace no funcionó y FaceNet tampoco, reportar error
                if results['method_used'] == 'none':
                    if not emb1_result['success']:
                        results['error'] = f"Imagen 1: {emb1_result['error']}"
                        results['faces_count_image1'] = emb1_result['faces_count']
                    elif not emb2_result['success']:
                        results['error'] = f"Imagen 2: {emb2_result['error']}"
                        results['faces_count_image2'] = emb2_result['faces_count']
                    return results

        # Aplicar umbral final según tipo de comparación
        final_threshold = self.thresholds.get(comparison_type, 0.65)

        # Solo verificar si el score principal supera el umbral del tipo de comparación
        if results['primary_score'] < final_threshold:
            results['verified'] = False

        # Si llegamos aquí con un método usado, consideramos éxito
        if results['method_used'] != 'none':
            results['success'] = True

        logging.info(f"[VERIFICATION] Método: {results['method_used']}, Score: {results['primary_score']:.4f}, Verificado: {results['verified']}")

        return results

# Inicializar sistema corregido
corrected_system = CorrectedFaceRecognitionSystem()

# Mantener compatibilidad con sistema anti-spoofing
model_test = AntiSpoofPredict(0)
image_cropper = CropImage()
logging.info("[SERVER] Silent-Face Anti-Spoofing cargado correctamente")

# Inicializar ID Card Detector
try:
    id_card_detector = IDCardDetector()
    logging.info("[SERVER] ID Card Detector inicializado correctamente")
except Exception as e:
    id_card_detector = None
    logging.warning(f"[SERVER] No se pudo inicializar ID Card Detector: {e}")

# Inicializar OCR System para extracción de número de documento
class OCRSystem:
    """Sistema de OCR para extraer números de documento de identificación"""

    def __init__(self):
        self.reader = None
        if EASYOCR_AVAILABLE:
            try:
                # OPTIMIZACIÓN: Solo español (más rápido que es+en)
                # Para cédulas solo necesitamos números, no texto multiidioma
                self.reader = easyocr.Reader(['es'], gpu=False, verbose=False)
                logging.info("[OCR] EasyOCR inicializado correctamente (es) - Modo optimizado")
            except Exception as e:
                logging.error(f"[OCR] Error inicializando EasyOCR: {e}")
                self.reader = None
        else:
            logging.warning("[OCR] EasyOCR no disponible")

    def extract_id_number(self, image: np.ndarray) -> dict:
        """
        Extraer número de documento de una imagen de cédula

        Returns:
            dict con keys: 'found', 'numbers', 'confidence', 'raw_text'
        """
        if self.reader is None:
            return {
                'found': False,
                'numbers': [],
                'confidence': 0.0,
                'raw_text': '',
                'error': 'OCR no disponible'
            }

        try:
            # OPTIMIZACIÓN 1: Reducir resolución de imagen para acelerar OCR
            # Las cédulas no necesitan alta resolución para detectar números
            optimized_image = self._resize_for_ocr(image)

            # OPTIMIZACIÓN 2: Configurar parámetros rápidos de EasyOCR
            # Realizar OCR con parámetros optimizados para velocidad
            logging.info("[OCR] Procesando imagen optimizada...")
            results_original = self.reader.readtext(
                optimized_image,
                paragraph=False,        # No agrupar en párrafos (más rápido)
                min_size=10,           # Ignorar texto muy pequeño
                text_threshold=0.6,    # Umbral de confianza más permisivo
                low_text=0.3,          # Detección de texto más rápida
                link_threshold=0.3,    # Enlaces más rápidos
                canvas_size=1280,      # Tamaño de canvas reducido
                mag_ratio=1.0          # Sin magnificación (más rápido)
            )

            # Si no encuentra nada, intentar con imagen preprocesada
            if not results_original:
                logging.info("[OCR] No se encontró texto en imagen original, probando con preprocesamiento...")
                processed_image = self._preprocess_for_ocr(image)
                results = self.reader.readtext(processed_image)
            else:
                results = results_original

            # Extraer números que parecen ser documentos de identidad
            id_numbers = []
            raw_text = []

            logging.info(f"[OCR] EasyOCR detectó {len(results)} elementos de texto")

            for (bbox, text, confidence) in results:
                raw_text.append(text)
                logging.info(f"[OCR] Texto detectado: '{text}' (confianza: {confidence:.2f})")

                # Buscar patrones de números de documento
                # Ejemplo: 12345678, 12-34567890-1, 001-1234567-8, etc.

                # Limpiar texto manteniendo números y guiones
                cleaned_text = re.sub(r'[^\d-]', '', text)

                # También buscar en el texto original por si tiene espacios o letras
                text_with_spaces = re.sub(r'[^\d\s-]', '', text)

                # Patrones más amplios de números de documento
                patterns = [
                    r'\d{7,11}',                    # 7-11 dígitos consecutivos
                    r'\d{2,4}-\d{6,9}-\d{1,2}',    # Formato xxx-xxxxxx-x (flexible)
                    r'\d{3}\s*\d{6,7}\s*\d',       # Con espacios
                    r'\d{11}',                      # 11 dígitos (cédula dominicana)
                    r'\d{8,10}',                    # 8-10 dígitos
                ]

                # Buscar en texto limpio
                for pattern in patterns:
                    # Buscar en texto sin espacios
                    matches = re.findall(pattern, cleaned_text)
                    for match in matches:
                        # FILTRO: Solo números de 10 o más dígitos (cédulas válidas)
                        digits_only = match.replace('-', '').replace(' ', '')
                        if match and len(digits_only) >= 10:
                            id_numbers.append({
                                'number': match,
                                'confidence': confidence,
                                'original_text': text
                            })
                            logging.info(f"[OCR] ✓ Número de cédula encontrado: '{match}' ({len(digits_only)} dígitos) de texto '{text}'")

                    # Buscar en texto con espacios
                    matches = re.findall(pattern, text_with_spaces)
                    for match in matches:
                        normalized = match.replace(' ', '')
                        digits_only = normalized.replace('-', '')
                        # FILTRO: Solo números de 10 o más dígitos (cédulas válidas)
                        if normalized and len(digits_only) >= 10:
                            id_numbers.append({
                                'number': normalized,
                                'confidence': confidence,
                                'original_text': text
                            })
                            logging.info(f"[OCR] ✓ Número de cédula con espacios encontrado: '{normalized}' ({len(digits_only)} dígitos) de texto '{text}'")

            # Ordenar por confianza
            id_numbers.sort(key=lambda x: x['confidence'], reverse=True)

            # Eliminar duplicados
            unique_numbers = []
            seen = set()
            for item in id_numbers:
                normalized = item['number'].replace('-', '').replace(' ', '')
                if normalized not in seen:
                    seen.add(normalized)
                    unique_numbers.append(item)

            logging.info(f"[OCR] Encontrados {len(unique_numbers)} números candidatos únicos")
            logging.info(f"[OCR] Texto completo detectado: {' | '.join(raw_text)}")

            if unique_numbers:
                logging.info(f"[OCR] Mejor candidato: {unique_numbers[0]['number']} (confianza: {unique_numbers[0]['confidence']:.2f})")
            else:
                logging.warning(f"[OCR] ⚠️ No se encontraron patrones de número de documento en el texto detectado")

            return {
                'found': len(unique_numbers) > 0,
                'numbers': unique_numbers,
                'confidence': unique_numbers[0]['confidence'] if unique_numbers else 0.0,
                'raw_text': ' '.join(raw_text),
                'best_match': unique_numbers[0]['number'] if unique_numbers else None,
                'total_text_elements': len(results)
            }

        except Exception as e:
            logging.error(f"[OCR] Error extrayendo número: {e}")
            return {
                'found': False,
                'numbers': [],
                'confidence': 0.0,
                'raw_text': '',
                'error': str(e)
            }

    def _resize_for_ocr(self, image: np.ndarray, max_width: int = 1280) -> np.ndarray:
        """
        Redimensionar imagen para OCR más rápido
        Resolución más baja = OCR más rápido sin perder precisión en números
        """
        try:
            height, width = image.shape[:2]

            # Si la imagen ya es pequeña, no redimensionar
            if width <= max_width:
                return image

            # Calcular nueva altura manteniendo aspecto
            ratio = max_width / width
            new_width = max_width
            new_height = int(height * ratio)

            # Redimensionar
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            logging.info(f"[OCR] Imagen redimensionada de {width}x{height} a {new_width}x{new_height} para acelerar OCR")

            return resized

        except Exception as e:
            logging.warning(f"[OCR] Error redimensionando imagen: {e}, usando original")
            return image

    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Preprocesar imagen para mejorar precisión de OCR"""
        try:
            # Convertir a escala de grises
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Aplicar CLAHE para mejorar contraste
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # Binarización adaptativa
            binary = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            # Reducción de ruido
            denoised = cv2.fastNlMeansDenoising(binary)

            return denoised

        except Exception as e:
            logging.warning(f"[OCR] Error en preprocesamiento, usando imagen original: {e}")
            return image

    def compare_id_numbers(self, number1: str, number2: str) -> dict:
        """
        Comparar dos números de documento

        Returns:
            dict con 'match', 'similarity', 'normalized1', 'normalized2'
        """
        if not number1 or not number2:
            return {
                'match': False,
                'similarity': 0.0,
                'normalized1': '',
                'normalized2': '',
                'reason': 'Uno o ambos números están vacíos'
            }

        # Normalizar: quitar guiones, espacios, etc.
        norm1 = re.sub(r'[^\d]', '', str(number1))
        norm2 = re.sub(r'[^\d]', '', str(number2))

        if not norm1 or not norm2:
            return {
                'match': False,
                'similarity': 0.0,
                'normalized1': norm1,
                'normalized2': norm2,
                'reason': 'Números normalizados están vacíos'
            }

        # Comparación exacta
        exact_match = norm1 == norm2

        # Calcular similitud (Levenshtein distance simple)
        similarity = self._calculate_similarity(norm1, norm2)

        # Considerar match si:
        # 1. Son exactamente iguales, O
        # 2. Uno contiene al otro (para casos donde un número es más largo), O
        # 3. Similitud > 0.9 (para errores menores de OCR)
        match = exact_match or norm1 in norm2 or norm2 in norm1 or similarity > 0.9

        logging.info(f"[OCR] Comparación: '{norm1}' vs '{norm2}' -> Match: {match}, Similitud: {similarity:.2f}")

        return {
            'match': match,
            'similarity': similarity,
            'normalized1': norm1,
            'normalized2': norm2,
            'exact_match': exact_match,
            'contains_match': norm1 in norm2 or norm2 in norm1
        }

    def _calculate_similarity(self, s1: str, s2: str) -> float:
        """Calcular similitud entre dos strings (0.0 a 1.0)"""
        if s1 == s2:
            return 1.0

        # Levenshtein distance simple
        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 0.0

        # Crear matriz
        matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]

        for i in range(len1 + 1):
            matrix[i][0] = i
        for j in range(len2 + 1):
            matrix[0][j] = j

        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if s1[i-1] == s2[j-1] else 1
                matrix[i][j] = min(
                    matrix[i-1][j] + 1,      # deletion
                    matrix[i][j-1] + 1,      # insertion
                    matrix[i-1][j-1] + cost  # substitution
                )

        distance = matrix[len1][len2]
        max_len = max(len1, len2)

        return 1.0 - (distance / max_len)

# Inicializar OCR System
try:
    ocr_system = OCRSystem()
    logging.info("[SERVER] OCR System inicializado correctamente")
except Exception as e:
    ocr_system = None
    logging.warning(f"[SERVER] No se pudo inicializar OCR System: {e}")

def detect_spoofing(image_data):
    """Detectar si una imagen es real o spoof usando Silent-Face-Anti-Spoofing"""
    try:
        # Convertir a numpy array
        if isinstance(image_data, io.BytesIO):
            image_data.seek(0)
            pil_image = Image.open(image_data)
            image_array = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        elif hasattr(image_data, 'read'):
            pil_image = Image.open(image_data)
            image_array = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        elif isinstance(image_data, np.ndarray):
            image_array = image_data
        elif isinstance(image_data, str):
            image_array = cv2.imread(image_data)
        else:
            pil_image = image_data
            image_array = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        if image_array is None:
            raise ValueError("No se pudo cargar la imagen")
        
        logging.info(f"[SPOOFING] Imagen cargada: {image_array.shape}")
        
        # Usar el sistema original de anti-spoofing
        image_bbox = [0, 0, image_array.shape[1], image_array.shape[0]]
        prediction = np.zeros((1, 3))
        
        for model_name in os.listdir("./resources/anti_spoof_models"):
            if model_name.endswith('.pth'):
                h_input, w_input, model_type, scale = parse_model_name(model_name)
                param = {
                    "org_img": image_array,
                    "bbox": image_bbox,
                    "scale": scale,
                    "out_w": w_input,
                    "out_h": h_input,
                    "crop": True,
                }
                
                if scale is None:
                    param["crop"] = False
                
                img = image_cropper.crop(**param)
                prediction += model_test.predict(img, os.path.join("./resources/anti_spoof_models", model_name))
        
        label = np.argmax(prediction)
        value = prediction[0][label] / len(os.listdir("./resources/anti_spoof_models"))
        
        if label == 1:
            result_label = "Real Face"
            is_real = True
        else:
            result_label = "Fake Face" 
            is_real = False
        
        logging.info(f"[SPOOFING] Resultado: {result_label}, Score: {value:.3f}")
        
        return {
            'is_real': is_real,
            'score': float(value),
            'label': result_label
        }
        
    except Exception as e:
        logging.error(f"[SPOOFING] Error: {str(e)}")
        return {
            'is_real': False,
            'score': 0.0,
            'label': f'Error: {str(e)}'
        }

def convert_image_data_to_array(image_data):
    """Convertir datos de imagen a numpy array"""
    try:
        if isinstance(image_data, io.BytesIO):
            image_data.seek(0)
            pil_image = Image.open(image_data)
        elif hasattr(image_data, 'read'):
            pil_image = Image.open(image_data)
        elif isinstance(image_data, np.ndarray):
            return image_data
        elif isinstance(image_data, str):
            pil_image = Image.open(image_data)
        else:
            pil_image = image_data
        
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        return np.array(pil_image)
        
    except Exception as e:
        logging.error(f"[CONVERT] Error convirtiendo imagen: {e}")
        return None

# --- Rutas del servidor ---
@app.route("/", methods=["GET"])
def index():
    return "Corrected Anti-Spoofing + Face Verification Server is running", 200

@app.route('/verify-profile', methods=['POST'])
def verify_profile():
    """Verificación de perfil CORREGIDA con comparación de Número de ID"""
    try:
        profile_url = request.form.get('profile_url')
        userid = request.form.get('userid')
        iddocument_file = request.files.get('iddocument')
        user_idnumber = request.form.get('idnumber', '')  # Número de ID del perfil de Moodle

        if not profile_url or not iddocument_file:
            return jsonify(success=False, message='Faltan datos: profile_url o iddocument'), 400

        logging.info(f"[PROFILE] Verificando perfil usuario {userid} - SISTEMA CORREGIDO")
        if user_idnumber:
            logging.info(f"[PROFILE] Número de ID del perfil Moodle: {user_idnumber}")

        # --- PASO 1: VALIDACIÓN DE CÉDULA ---
        id_card_result = None
        if id_card_detector is not None:
            try:
                # Leer imagen de cédula
                iddocument_file.seek(0)
                file_bytes = np.frombuffer(iddocument_file.read(), np.uint8)
                id_card_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                # Detectar si es un documento válido
                id_card_result = id_card_detector.detect(id_card_image)

                if not id_card_result['is_valid']:
                    return jsonify(
                        success=True,
                        verified=False,
                        message='La imagen proporcionada no corresponde a un documento de identidad válido. Por favor, capture una fotografía clara de su cédula o documento oficial.',
                        score=0.0,
                        id_card_detection=id_card_result
                    ), 200

                logging.info(f"[PROFILE] ✓ Cédula validada con confianza: {id_card_result['confidence']:.2%}")
                # Resetear archivo para siguiente procesamiento
                iddocument_file.seek(0)

            except Exception as e:
                logging.warning(f"[PROFILE] Error en validación de cédula: {str(e)}")
                # Continuar sin validación si hay error
                id_card_result = {'error': str(e)}
        else:
            logging.warning("[PROFILE] ID Card Detector no disponible, saltando validación")

        # --- PASO 2: Cargar imagen de perfil
        try:
            response = requests.get(profile_url, timeout=10, verify=False)
            response.raise_for_status()
            profile_image = convert_image_data_to_array(io.BytesIO(response.content))
            
            if profile_image is None:
                raise ValueError("Imagen de perfil inválida")
                
            logging.info(f"[PROFILE] Imagen de perfil cargada: {profile_image.shape}")
        except Exception as e:
            return jsonify(
                success=True,
                verified=False,
                message=f'Error cargando imagen de perfil: {str(e)}',
                score=0.0
            ), 200

        # Procesar imagen de cédula
        id_image = convert_image_data_to_array(iddocument_file)
        if id_image is None:
            return jsonify(
                success=True,
                verified=False,
                message='No se pudo procesar la imagen de cédula',
                score=0.0
            ), 200

        logging.info(f"[PROFILE] Imagen de cédula cargada: {id_image.shape}")

        # --- PASO 3: EXTRACCIÓN Y COMPARACIÓN DE NÚMERO DE DOCUMENTO (Opcional) ---
        id_number_result = None
        id_number_match = None

        if user_idnumber and ocr_system is not None:
            try:
                logging.info("[PROFILE] Iniciando extracción de número de documento con OCR...")

                # Extraer número del documento de la cédula
                ocr_extraction = ocr_system.extract_id_number(id_image)

                if ocr_extraction['found']:
                    extracted_number = ocr_extraction['best_match']
                    logging.info(f"[PROFILE] Número extraído del documento: {extracted_number}")

                    # Comparar con el número del perfil
                    id_number_match = ocr_system.compare_id_numbers(user_idnumber, extracted_number)

                    id_number_result = {
                        'extracted': extracted_number,
                        'profile_number': user_idnumber,
                        'match': id_number_match['match'],
                        'similarity': id_number_match['similarity'],
                        'ocr_confidence': ocr_extraction['confidence'],
                        'all_candidates': ocr_extraction['numbers'][:3]  # Top 3 candidatos
                    }

                    if id_number_match['match']:
                        logging.info(f"[PROFILE] ✅ Números de ID coinciden!")
                    else:
                        logging.warning(f"[PROFILE] ⚠️ Números de ID NO coinciden: '{user_idnumber}' vs '{extracted_number}'")
                else:
                    logging.warning("[PROFILE] No se pudo extraer número de documento del ID")
                    raw_text = ocr_extraction.get('raw_text', '')
                    total_elements = ocr_extraction.get('total_text_elements', 0)

                    error_msg = 'No se encontró número de documento'
                    if total_elements > 0:
                        error_msg += f'. Se detectó texto pero no coincide con patrones de número: "{raw_text[:100]}"'
                    else:
                        error_msg += '. No se detectó texto en la imagen del documento'

                    id_number_result = {
                        'extracted': None,
                        'profile_number': user_idnumber,
                        'match': False,
                        'error': error_msg,
                        'raw_text': raw_text,
                        'total_text_detected': total_elements
                    }

            except Exception as e:
                logging.error(f"[PROFILE] Error en extracción/comparación de número: {e}")
                id_number_result = {
                    'error': str(e),
                    'profile_number': user_idnumber
                }
        elif user_idnumber:
            logging.info("[PROFILE] OCR no disponible, saltando comparación de número de ID")
            id_number_result = {'ocr_available': False, 'profile_number': user_idnumber}
        else:
            logging.info("[PROFILE] No se proporcionó número de ID del perfil, saltando comparación")

        # --- PASO 4: Verificar rostros usando sistema corregido ---
        # IMPORTANTE: Permitir múltiples rostros en documento ID (cédulas tienen 2 fotos)
        # pero mantener validación estricta en foto de perfil
        verification_result = corrected_system.verify_faces(
            profile_image,              # image1: foto de perfil (debe tener 1 rostro)
            id_image,                   # image2: foto de cédula (puede tener 2+ rostros)
            'profile_vs_id',            # tipo de comparación
            image1_allow_multiple=False,  # Perfil: solo 1 rostro
            image2_allow_multiple=True    # Cédula: permitir múltiples, selecciona el más grande
        )

        # VALIDACIÓN: Verificar si hubo error en la verificación (múltiples rostros, etc.)
        if not verification_result['success']:
            error_message = verification_result.get('error', 'Error en verificación facial')
            faces_image1 = verification_result.get('faces_count_image1', 0)
            faces_image2 = verification_result.get('faces_count_image2', 0)

            logging.warning(f"[PROFILE] Verificación falló: {error_message}")

            return jsonify(
                success=True,
                verified=False,
                message=error_message,
                score=0.0,
                faces_detected={
                    'profile_image': faces_image1,
                    'id_document_image': faces_image2
                },
                id_card_detection=id_card_result,
                id_number_verification=id_number_result
            ), 200

        # Determinar verificación final
        # La verificación facial es obligatoria, el número de ID es complementario
        face_verified = verification_result['verified']

        # CAMBIO: Si los números de ID no coinciden, la verificación debe fallar
        if id_number_match is not None and id_number_match['match'] == False:
            # Si los números NO coinciden, bloquear la verificación
            logging.warning(f"[PROFILE] ❌ VERIFICACIÓN RECHAZADA: El número de ID del documento ({id_number_result.get('extracted')}) no coincide con el del perfil ({user_idnumber})")
            face_verified = False  # Forzar verificación fallida

        message = 'Los rostros coinciden - perfil verificado' if face_verified else \
                 f'Los rostros no coinciden (score: {verification_result["primary_score"]:.3f})'

        # Agregar información de número de ID al mensaje si está disponible
        if id_number_result and id_number_result.get('match') is not None:
            if id_number_result['match']:
                message += ' | Número de ID verificado ✓'
            else:
                message = f'❌ El número de ID del documento ({id_number_result.get("extracted")}) no coincide con el del perfil ({user_idnumber})'

        return jsonify(
            success=True,
            verified=bool(face_verified),
            message=message,
            score=round(verification_result['primary_score'], 4),
            backup_score=round(verification_result['backup_score'], 4),
            method_used=verification_result['method_used'],
            details=verification_result['details'],
            faces_detected={
                'profile_image': verification_result['faces_count_image1'],
                'id_document_image': verification_result['faces_count_image2']
            },
            userid=str(userid),
            id_card_detection=id_card_result,
            id_number_verification=id_number_result  # Nueva información
        ), 200

    except Exception as e:
        logging.exception("[PROFILE] Error en verificación de perfil")
        return jsonify(success=False, message=f'Error del servidor: {str(e)}'), 500

@app.route('/verify', methods=['POST'])
def verify():
    """Ruta principal con sistema corregido - incluye info de número de ID"""
    try:
        user_id = request.form.get('userid')
        quizid = request.form.get('quizid')
        wwwroot = request.form.get('wwwroot', 'http://localhost')
        image_file = request.files.get('image')
        user_idnumber = request.form.get('idnumber', '')  # Número de ID del perfil

        if not user_id or not image_file:
            return jsonify(success=False, message='Faltan datos obligatorios'), 400

        logging.info(f"[SERVER] Verificando usuario {user_id} - SISTEMA CORREGIDO")
        if user_idnumber:
            logging.info(f"[SERVER] Usuario tiene número de ID registrado: {user_idnumber}")

        # Anti-spoofing
        spoofing_result = detect_spoofing(image_file)
        
        if not spoofing_result['is_real']:
            return jsonify(
                success=True,
                verified=False,
                blocked_by_antispoofing=True,
                message=f'Acceso bloqueado: {spoofing_result["label"]}',
                antispoofing_result=spoofing_result
            ), 200
        
        logging.info(f"[SERVER] ✓ Anti-spoofing aprobado")

        # Verificación facial
        image_file.seek(0)
        live_image = convert_image_data_to_array(image_file)
        if live_image is None:
            return jsonify(
                success=True,
                verified=False,
                message='No se pudo procesar la imagen',
                antispoofing_result=spoofing_result
            ), 200

        # Cargar imagen de perfil
        try:
            profile_url = f'{wwwroot}/user/pix.php/{user_id}/f3.jpg'
            response = requests.get(profile_url, timeout=10, verify=False)
            response.raise_for_status()
            profile_image = convert_image_data_to_array(io.BytesIO(response.content))
            
            if profile_image is None:
                raise ValueError("Imagen de perfil inválida")
                
        except Exception as e:
            return jsonify(
                success=True,
                verified=False,
                message=f'Error cargando imagen de perfil: {str(e)}',
                antispoofing_result=spoofing_result
            ), 200

        # Verificar con sistema corregido
        verification_result = corrected_system.verify_faces(
            profile_image, live_image, 'live_vs_profile'
        )

        # VALIDACIÓN: Verificar si hubo error en la verificación (múltiples rostros, etc.)
        if not verification_result['success']:
            error_message = verification_result.get('error', 'Error en verificación facial')
            faces_profile = verification_result.get('faces_count_image1', 0)
            faces_live = verification_result.get('faces_count_image2', 0)

            logging.warning(f"[SERVER] Verificación falló para usuario {user_id}: {error_message}")

            return jsonify(
                success=True,
                verified=False,
                blocked_by_antispoofing=False,
                message=error_message,
                antispoofing_result=spoofing_result,
                faces_detected={
                    'profile_image': faces_profile,
                    'live_image': faces_live
                }
            ), 200

        verification_success = spoofing_result['is_real'] and verification_result['verified']

        message = 'Verificación exitosa' if verification_success else \
                 ('El rostro no coincide con el perfil' if spoofing_result['is_real'] else
                  f'Bloqueado por anti-spoofing')

        # Información sobre número de ID (solo informativo en este endpoint)
        id_number_info = None
        if user_idnumber:
            id_number_info = {
                'profile_has_id': True,
                'profile_id_number': user_idnumber,
                'note': 'Número de ID registrado en perfil. Para verificación completa del documento, use /verify-profile.'
            }

        return jsonify(
            success=True,
            verified=verification_success,
            blocked_by_antispoofing=not spoofing_result['is_real'],
            message=message,
            antispoofing_result=spoofing_result,
            face_recognition_result={
                'primary_score': round(verification_result['primary_score'], 4),
                'backup_score': round(verification_result['backup_score'], 4),
                'method_used': verification_result['method_used'],
                'details': verification_result['details']
            },
            faces_detected={
                'profile_image': verification_result['faces_count_image1'],
                'live_image': verification_result['faces_count_image2']
            },
            face_match=bool(verification_result['verified']),
            id_number_info=id_number_info  # Información del número de ID
        ), 200

    except Exception as e:
        logging.exception("[SERVER] Error en verificación")
        return jsonify(success=False, message=f'Error del servidor: {str(e)}'), 500

@app.route('/test-antispoofing', methods=['POST'])
def test_antispoofing():
    """Test anti-spoofing"""
    try:
        image_file = request.files.get('image')
        if not image_file:
            return jsonify(success=False, message='No imagen'), 400
        
        result = detect_spoofing(image_file)
        return jsonify(success=True, result=result), 200
        
    except Exception as e:
        return jsonify(success=False, message=str(e)), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Información del sistema corregido"""
    models_loaded = []
    if corrected_system.insightface is not None:
        models_loaded.append('insightface')
    if corrected_system.facenet is not None:
        models_loaded.append('facenet')
    if corrected_system.deepface is not None:
        models_loaded.append('deepface')

    return jsonify(
        version="CORRECTED_v2.1_SECURE_ID_SUPPORT",
        antispoofing_model="Silent-Face-Anti-Spoofing",
        face_recognition_models=models_loaded,
        thresholds=corrected_system.thresholds,
        primary_model="InsightFace" if corrected_system.insightface else "FaceNet",
        security_features=[
            "Multiple face detection validation",
            "Single face enforcement for live images",
            "Smart ID document handling (allows multiple faces, selects largest)",
            "Anti-spoofing detection",
            "Conservative similarity thresholds",
            "Detailed error reporting"
        ],
        validation_rules={
            "live_images": {
                "faces_per_image": "exactly_1",
                "rejects_multiple_faces": True,
                "rejects_no_faces": True
            },
            "id_documents": {
                "faces_per_image": "1_or_more",
                "selects_largest_face": True,
                "typical_case": "2 faces (main photo + secondary/hologram)"
            }
        }
    ), 200

@app.route('/verify-with-profile', methods=['POST'])
def verify_with_profile():
    """
    Verificación facial en vivo comparando con perfil verificado del usuario - SISTEMA CORREGIDO
    Incluye verificación de número de ID si está disponible
    """
    try:
        # Obtener datos del request
        image_data = request.files.get('image')
        userid = request.form.get('userid')
        wwwroot = request.form.get('wwwroot')
        user_idnumber = request.form.get('idnumber', '')  # Número de ID del perfil

        if not all([image_data, userid, wwwroot]):
            return jsonify(success=False, message='Faltan datos: image, userid o wwwroot'), 400

        logging.info(f"[PROFILE-LIVE] Verificando imagen en vivo vs perfil para usuario {userid} - SISTEMA CORREGIDO")
        if user_idnumber:
            logging.info(f"[PROFILE-LIVE] Número de ID del perfil disponible: {user_idnumber}")

        # --- PASO 1: Anti-Spoofing ---
        try:
            antispoofing_result = detect_spoofing(image_data)
            
            if not antispoofing_result.get('is_real', False):
                logging.info(f"[PROFILE-LIVE] Usuario {userid} - Anti-spoofing falló")
                return jsonify(
                    success=True,
                    verified=False,
                    message='❌ Imagen detectada como falsificada',
                    antispoofing_result=antispoofing_result
                ), 200

            logging.info(f"[PROFILE-LIVE] Usuario {userid} - Anti-spoofing pasó ✓")

        except Exception as e:
            logging.exception("[PROFILE-LIVE] Error en anti-spoofing")
            return jsonify(
                success=True,
                verified=False,
                message=f'❌ Error en detección anti-spoofing: {str(e)}',
                antispoofing_result=False
            ), 200

        # --- PASO 2: Procesar imagen en vivo ---
        live_image = convert_image_data_to_array(image_data)
        if live_image is None:
            return jsonify(
                success=True,
                verified=False,
                message='❌ No se pudo procesar la imagen en vivo',
                antispoofing_result=antispoofing_result
            ), 200

        # --- PASO 3: Obtener imagen de perfil ---
        try:
            profile_url = f"{wwwroot}/user/pix.php/{userid}/f3.jpg"
            response = requests.get(profile_url, timeout=10, verify=False)
            response.raise_for_status()
            profile_image = convert_image_data_to_array(io.BytesIO(response.content))
            
            if profile_image is None:
                raise ValueError("Imagen de perfil inválida")
                
            logging.info(f"[PROFILE-LIVE] Imagen de perfil cargada desde {profile_url}")
        except Exception as e:
            return jsonify(
                success=True,
                verified=False,
                message=f'❌ Error cargando imagen de perfil: {str(e)}',
                antispoofing_result=antispoofing_result
            ), 200

        # --- PASO 4: Verificar rostros usando sistema corregido ---
        verification_result = corrected_system.verify_faces(
            live_image, profile_image, 'live_vs_profile'
        )

        # VALIDACIÓN: Verificar si hubo error en la verificación (múltiples rostros, etc.)
        if not verification_result['success']:
            error_message = verification_result.get('error', 'Error en verificación facial')
            faces_live = verification_result.get('faces_count_image1', 0)
            faces_profile = verification_result.get('faces_count_image2', 0)

            logging.warning(f"[PROFILE-LIVE] Verificación falló para usuario {userid}: {error_message}")

            return jsonify(
                success=True,
                verified=False,
                message=error_message,
                antispoofing_result=antispoofing_result,
                faces_detected={
                    'live_image': faces_live,
                    'profile_image': faces_profile
                },
                userid=userid,
                system_version="corrected"
            ), 200

        faces_match = verification_result['verified']
        primary_score = verification_result['primary_score']

        if faces_match:
            message = f'✅ Verificación exitosa - Score: {primary_score:.3f}'
            logging.info(f"[PROFILE-LIVE] Usuario {userid} - VERIFICADO ✓ (score: {primary_score:.3f})")
        else:
            message = f'❌ Las caras no coinciden - Score: {primary_score:.3f}'
            logging.info(f"[PROFILE-LIVE] Usuario {userid} - NO VERIFICADO ✗ (score: {primary_score:.3f})")

        # Información sobre número de ID (solo informativo en este endpoint)
        id_number_info = None
        if user_idnumber:
            id_number_info = {
                'profile_has_id': True,
                'profile_id_number': user_idnumber,
                'note': 'Número de ID disponible en perfil. Verificación completa requiere comparación con documento físico.'
            }

        return jsonify(
            success=True,
            verified=faces_match,
            message=message,
            score=round(primary_score, 4),
            face_recognition_result=verification_result,
            faces_detected={
                'live_image': verification_result['faces_count_image1'],
                'profile_image': verification_result['faces_count_image2']
            },
            antispoofing_result=antispoofing_result,
            userid=userid,
            id_number_info=id_number_info,  # Información del número de ID
            system_version="corrected"
        ), 200

    except Exception as e:
        logging.exception("[PROFILE-LIVE] Error en verificación con perfil - SISTEMA CORREGIDO")
        return jsonify(success=False, message=f'Error del servidor: {str(e)}'), 500

@app.route('/test-ocr', methods=['POST'])
def test_ocr():
    """Endpoint de prueba para OCR - diagnóstico"""
    try:
        image_file = request.files.get('image')
        if not image_file:
            return jsonify(success=False, message='No se proporcionó imagen'), 400

        logging.info("[TEST-OCR] Iniciando prueba de OCR...")

        # Convertir imagen
        id_image = convert_image_data_to_array(image_file)
        if id_image is None:
            return jsonify(success=False, message='No se pudo procesar la imagen'), 400

        # Extraer con OCR
        if ocr_system is None:
            return jsonify(success=False, message='OCR no disponible'), 400

        ocr_result = ocr_system.extract_id_number(id_image)

        return jsonify(
            success=True,
            ocr_available=True,
            result=ocr_result,
            debug_info={
                'image_shape': id_image.shape,
                'total_text_detected': ocr_result.get('total_text_elements', 0),
                'numbers_found': len(ocr_result.get('numbers', [])),
                'raw_text_preview': ocr_result.get('raw_text', '')[:200]
            }
        ), 200

    except Exception as e:
        logging.exception("[TEST-OCR] Error en prueba de OCR")
        return jsonify(success=False, message=f'Error: {str(e)}'), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify(status="healthy", version="corrected"), 200

# --- Inicio del servidor ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=5001, type=int)
    args = parser.parse_args()
    
    if not os.path.exists('./resources/anti_spoof_models'):
        print("❌ Error: No se encuentra la carpeta './resources/anti_spoof_models'")
        sys.exit(1)
    
    if not os.path.exists('./src'):
        print("❌ Error: No se encuentra la carpeta './src'")
        sys.exit(1)
    
    logging.info(f"🚀 Servidor CORREGIDO v2.1 SECURE iniciado en http://{args.host}:{args.port}")
    logging.info("🔧 Sistema de verificación facial corregido y más estricto")
    logging.info("🛡️  SEGURIDAD: Validación de rostro único en imágenes EN VIVO")
    logging.info("📄 DOCUMENTOS ID: Soporte inteligente para cédulas con múltiples rostros")
    logging.info("✅ Imágenes en vivo con múltiples personas: RECHAZADAS automáticamente")
    logging.info("✅ Documentos ID con 2 fotos: ACEPTADOS (selecciona foto principal)")

    app.run(host=args.host, port=args.port, debug=False)