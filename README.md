# Face ID Flask Server - AI-Powered Verification Backend

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0%2B-green)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-GPLv3-green)](https://www.gnu.org/licenses/gpl-3.0.html)

Servidor Flask para verificación biométrica facial con **detección anti-spoofing** y **reconocimiento facial de alta precisión**. Backend de inteligencia artificial para el [plugin de Moodle Face ID](https://github.com/Galo45/moodle-quizaccess-faceid-).

---

## 🎯 Características Principales

### Reconocimiento Facial Multi-Modelo

✅ **InsightFace (ArcFace)** - Modelo principal de última generación
✅ **FaceNet (MTCNN + InceptionResnetV1)** - Modelo base confiable
✅ **DeepFace** - Modelo de respaldo opcional
✅ **Consenso entre modelos** para mayor precisión

### Detección Anti-Spoofing

✅ **Silent-Face-Anti-Spoofing** con múltiples modelos MiniFASNet
✅ **Análisis multi-escala** en patches de 80x80
✅ **Supervisión auxiliar de espectro de Fourier**
✅ **Detección de fotos, videos y pantallas**

### Validación de Documentos de Identidad

✅ **Detector de tarjetas ID** con análisis de bordes y texto
✅ **Extracción OCR** con EasyOCR
✅ **Patrones múltiples** para diferentes formatos de cédula
✅ **Comparación fuzzy** con Levenshtein distance

### Seguridad Reforzada (v2.1)

✅ **Validación estricta de rostro único** en imágenes en vivo
✅ **Manejo inteligente de documentos ID** con múltiples rostros
✅ **Selección automática del rostro principal** en cédulas
✅ **Prevención de suplantación grupal**

---

## 📋 Requisitos del Sistema

### Software Base

- **Python:** 3.8 o superior
- **Sistema operativo:** Windows, Linux, macOS
- **RAM:** Mínimo 4 GB (8 GB recomendado)
- **Espacio en disco:** 2 GB para modelos de IA

### Hardware Recomendado

| Componente | Mínimo | Recomendado |
|------------|--------|-------------|
| **CPU** | Dual-core 2.0 GHz | Quad-core 3.0 GHz+ |
| **RAM** | 4 GB | 8 GB+ |
| **GPU** | No requerida | NVIDIA CUDA compatible |
| **Red** | 10 Mbps | 100 Mbps+ |

**Nota:** GPU no es necesaria pero acelera el procesamiento significativamente.

---

## 🚀 Instalación

### 1️⃣ Clonar el Repositorio

```bash
git clone https://github.com/Galo45/faceid-flask-server-.git
cd faceid-flask-server-
```

### 2️⃣ Crear Entorno Virtual

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Instalar Dependencias

```bash
pip install -r requirements.txt
```

**Contenido de `requirements.txt`:**
```txt
flask==2.3.0
flask-cors==4.0.0
numpy==1.24.3
opencv-python==4.8.0.74
torch==2.0.1
torchvision==0.15.2
facenet-pytorch==2.5.3
insightface==0.7.3
onnxruntime==1.15.1
deepface==0.0.79
easyocr==1.7.0
Pillow==10.0.0
requests==2.31.0
```

### 4️⃣ Descargar Modelos

Los modelos se descargan automáticamente en el primer uso, pero puedes pre-descargarlos:

**Anti-Spoofing Models (incluidos):**
```
resources/anti_spoof_models/
├── 4_0_0_80x80_MiniFASNetV1SE.pth
└── 2.7_80x80_MiniFASNetV2.pth
```

**Detection Model (incluido):**
```
resources/detection_model/
├── Widerface-RetinaFace.caffemodel
└── deploy.prototxt
```

**InsightFace/DeepFace (descarga automática):**
- InsightFace → `~/.insightface/models/`
- DeepFace → `~/.deepface/weights/`

### 5️⃣ Verificar Instalación

```bash
python face3_corrected.py --help
```

Deberías ver:
```
usage: face3_corrected.py [-h] [--host HOST] [--port PORT]

Face Recognition Server with Anti-Spoofing
```

---

## 🎮 Uso

### Iniciar el Servidor

**Modo desarrollo (localhost):**
```bash
python face3_corrected.py --host 127.0.0.1 --port 5001
```

**Modo producción (acceso en red):**
```bash
python face3_corrected.py --host 0.0.0.0 --port 5001
```

**Con logging detallado:**
```bash
python -u face3_corrected.py --host 127.0.0.1 --port 5001 2>&1 | tee server.log
```

### Parámetros de Línea de Comandos

| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `--host` | Dirección IP del servidor | `127.0.0.1` |
| `--port` | Puerto del servidor | `5001` |
| `-h, --help` | Muestra ayuda | - |

### Verificar Estado del Servidor

```bash
# Health check
curl http://127.0.0.1:5001/health

# Información de modelos
curl http://127.0.0.1:5001/model-info
```

---

## 📡 API Endpoints

### 1. Health Check

**Endpoint:** `GET /`

**Respuesta:**
```json
{
  "status": "Face Recognition Server is running",
  "version": "2.1",
  "timestamp": "2025-01-05T10:30:00"
}
```

### 2. Verificar Imagen en Vivo vs Perfil

**Endpoint:** `POST /verify`

**Descripción:** Verifica una imagen en vivo contra la foto de perfil del usuario.

**Request:**
```bash
curl -X POST http://127.0.0.1:5001/verify \
  -F "image=@face_live.jpg" \
  -F "userid=123" \
  -F "quizid=456" \
  -F "wwwroot=http://moodle.example.com"
```

**Parámetros:**
| Campo | Tipo | Descripción |
|-------|------|-------------|
| `image` | File | Imagen en vivo capturada (JPEG/PNG) |
| `userid` | String | ID del usuario en Moodle |
| `quizid` | String | ID del quiz |
| `wwwroot` | String | URL raíz de Moodle |

**Respuesta exitosa:**
```json
{
  "success": true,
  "verified": true,
  "score": 0.872,
  "message": "Identidad verificada correctamente",
  "faces_detected": {
    "live_image": 1,
    "profile_image": 1
  },
  "antispoofing": {
    "is_real": true,
    "confidence": 0.95
  },
  "models_used": ["insightface", "facenet"]
}
```

**Respuesta fallida:**
```json
{
  "success": false,
  "verified": false,
  "score": 0.45,
  "message": "No se pudo verificar la identidad. Score: 0.45",
  "faces_detected": {
    "live_image": 1,
    "profile_image": 1
  }
}
```

**Errores comunes:**
```json
{
  "success": false,
  "verified": false,
  "message": "Se detectaron 2 personas en la imagen. Por favor, asegúrese de estar solo en el encuadre.",
  "faces_detected": {
    "live_image": 2
  }
}
```

### 3. Verificar Perfil vs Documento ID

**Endpoint:** `POST /verify-profile`

**Descripción:** Verifica la foto de perfil del usuario contra su documento de identidad.

**Request:**
```bash
curl -X POST http://127.0.0.1:5001/verify-profile \
  -F "iddocument=@cedula.jpg" \
  -F "profile_url=http://moodle.example.com/user/pix.php/123/f3.jpg" \
  -F "userid=123" \
  -F "idnumber=001-1234567-8"
```

**Parámetros:**
| Campo | Tipo | Descripción |
|-------|------|-------------|
| `iddocument` | File | Foto del documento de identidad |
| `profile_url` | String | URL de la foto de perfil |
| `userid` | String | ID del usuario |
| `idnumber` | String (opcional) | Número de documento para OCR |

**Respuesta exitosa:**
```json
{
  "success": true,
  "verified": true,
  "score": 0.785,
  "message": "Perfil verificado exitosamente",
  "faces_detected": {
    "profile_image": 1,
    "id_document": 2
  },
  "id_document_info": {
    "is_valid_id": true,
    "selected_face": "largest",
    "face_area": 15360.5
  },
  "id_number_verification": {
    "extracted_id": "001-1234567-8",
    "profile_number": "001-1234567-8",
    "match": true,
    "similarity": 1.0,
    "confidence": 0.92
  }
}
```

**Respuesta con número no coincidente:**
```json
{
  "success": true,
  "verified": true,
  "score": 0.78,
  "message": "Rostro verificado pero número de ID no coincide",
  "id_number_verification": {
    "extracted_id": "001-9876543-2",
    "profile_number": "001-1234567-8",
    "match": false,
    "similarity": 0.33
  }
}
```

### 4. Verificar Imagen en Vivo con Perfil Verificado

**Endpoint:** `POST /verify-with-profile`

**Descripción:** Verifica imagen en vivo usando perfil previamente verificado.

**Request:**
```bash
curl -X POST http://127.0.0.1:5001/verify-with-profile \
  -F "image=@face_live.jpg" \
  -F "userid=123" \
  -F "wwwroot=http://moodle.example.com"
```

**Parámetros:**
| Campo | Tipo | Descripción |
|-------|------|-------------|
| `image` | File | Imagen en vivo |
| `userid` | String | ID del usuario |
| `wwwroot` | String | URL raíz de Moodle |

**Respuesta:** Similar a `/verify`

### 5. Test Anti-Spoofing

**Endpoint:** `POST /test-antispoofing`

**Descripción:** Solo prueba detección anti-spoofing.

**Request:**
```bash
curl -X POST http://127.0.0.1:5001/test-antispoofing \
  -F "image=@test_image.jpg"
```

**Respuesta:**
```json
{
  "success": true,
  "is_real": true,
  "confidence": 0.98,
  "label": 1,
  "message": "La imagen parece ser real (confianza: 98.5%)"
}
```

### 6. Test OCR

**Endpoint:** `POST /test-ocr`

**Descripción:** Solo prueba extracción OCR de número de documento.

**Request:**
```bash
curl -X POST http://127.0.0.1:5001/test-ocr \
  -F "image=@cedula.jpg"
```

**Respuesta:**
```json
{
  "success": true,
  "found": true,
  "extracted_numbers": [
    {
      "number": "001-1234567-8",
      "confidence": 0.95,
      "original_text": "001-1234567-8"
    }
  ],
  "raw_text": "REPÚBLICA DOMINICANA | CÉDULA | 001-1234567-8 | JUAN PÉREZ",
  "total_text_elements": 15
}
```

### 7. Model Info

**Endpoint:** `GET /model-info`

**Respuesta:**
```json
{
  "models": {
    "insightface": true,
    "facenet": true,
    "deepface": false,
    "antispoofing": true,
    "ocr": true
  },
  "thresholds": {
    "profile_vs_id": 0.7,
    "live_vs_profile": 0.65,
    "insightface": 0.4,
    "facenet": 0.7,
    "deepface": 0.68
  },
  "version": "2.1",
  "device": "cpu"
}
```

### 8. Health Status

**Endpoint:** `GET /health`

**Respuesta:**
```json
{
  "status": "ok",
  "uptime": 3600,
  "models_loaded": {
    "insightface": true,
    "facenet": true,
    "antispoofing": true,
    "ocr": true
  }
}
```

---

## 🏗️ Arquitectura del Sistema

### Componentes Principales

```
face3_corrected.py (Main Server)
├── CorrectedFaceRecognitionSystem (Core)
│   ├── InsightFace (ArcFace) - Primary model
│   ├── FaceNet (MTCNN + InceptionResnetV1) - Base model
│   ├── DeepFace - Backup model
│   ├── Anti-Spoofing Detector
│   ├── OCR System (EasyOCR)
│   └── ID Card Detector
├── Flask App (Web Server)
│   ├── /verify
│   ├── /verify-profile
│   ├── /verify-with-profile
│   ├── /test-antispoofing
│   ├── /test-ocr
│   └── /model-info
└── Resources
    ├── anti_spoof_models/
    └── detection_model/
```

### Módulos Auxiliares

```
src/
├── anti_spoof_predict.py        # Predicción anti-spoofing
├── generate_patches.py          # Generación de patches
├── utility.py                   # Utilidades
├── id_card_detector.py          # Detector de documentos ID
├── data_io/                     # I/O de datos
└── model_lib/                   # Arquitecturas de modelos
    └── MiniFASNet.py            # Red anti-spoofing
```

### Flujo de Procesamiento

#### Verificación en Vivo (/verify)
```
1. Recibir imagen en vivo
   ↓
2. Anti-Spoofing Detection
   ├─ Generar patches 80x80
   ├─ Procesar con MiniFASNet V1SE y V2
   ├─ Calcular score promedio
   └─ Clasificar: REAL (>0.5) o FAKE (≤0.5)
   ↓
3. Si es REAL → Extracción de embedding
   ├─ Detectar rostros (InsightFace/MTCNN)
   ├─ VALIDAR: Debe ser exactamente 1 rostro
   ├─ Si >1 rostro → RECHAZAR (seguridad)
   ├─ Extraer embedding normalizado
   └─ Dimensión: 512 (InsightFace) / 512 (FaceNet)
   ↓
4. Descargar foto de perfil desde Moodle
   ↓
5. Extraer embedding de perfil
   ├─ VALIDAR: Debe ser exactamente 1 rostro
   └─ Normalizar embedding
   ↓
6. Calcular similitud coseno
   ├─ similarity = 1 - cosine_distance(emb1, emb2)
   └─ Threshold: 0.65 para live vs profile
   ↓
7. Retornar resultado
   ├─ verified = (similarity >= threshold)
   ├─ score = similarity
   └─ message + metadata
```

#### Verificación de Perfil (/verify-profile)
```
1. Recibir documento ID
   ↓
2. Validar documento con IDCardDetector
   ├─ Detectar bordes y contornos
   ├─ Validar aspect ratio (1.5-1.7)
   ├─ Verificar presencia de texto
   └─ Si no es documento → RECHAZAR
   ↓
3. Extraer rostro del documento
   ├─ Detectar todos los rostros
   ├─ PERMITIR múltiples rostros (típico en cédulas)
   ├─ Seleccionar el más grande (foto principal)
   └─ Log: "X rostros detectados, seleccionando mayor"
   ↓
4. Descargar foto de perfil
   ↓
5. Extraer rostro de perfil
   ├─ VALIDAR: Exactamente 1 rostro
   └─ Si ≠1 → RECHAZAR
   ↓
6. Comparar rostros
   ├─ Threshold: 0.7 (más estricto)
   └─ Calcular similitud
   ↓
7. OCR: Extraer número de documento
   ├─ Redimensionar imagen a 1280px max
   ├─ Preprocesar: CLAHE, binarización
   ├─ EasyOCR: Detectar texto
   ├─ Buscar patrones de cédula:
   │  ├─ xxx-xxxxxxx-x
   │  ├─ xxxxxxxxxxx (11 dígitos)
   │  └─ Otros formatos
   ├─ Filtrar: Solo ≥10 dígitos
   └─ Seleccionar mejor candidato
   ↓
8. Comparar números
   ├─ Normalizar: quitar guiones, espacios
   ├─ Comparar con idnumber de Moodle
   ├─ Calcular similitud Levenshtein
   └─ Match si: exacto || contiene || similarity>0.9
   ↓
9. Retornar resultado
   ├─ verified = (face_match && id_match)
   ├─ score + id_verification
   └─ message detallado
```

### Umbrales de Similitud

| Comparación | Threshold | Modelo | Razón |
|-------------|-----------|--------|-------|
| **Perfil vs ID** | 0.7 | InsightFace/FaceNet | Más estricto (fotos diferentes contextos) |
| **Live vs Perfil** | 0.65 | InsightFace/FaceNet | Moderado (misma persona, condiciones diferentes) |
| **InsightFace** | 0.4 | Distancia coseno | Específico para ArcFace |
| **FaceNet** | 0.7 | Similitud coseno | Basado en paper original |
| **DeepFace** | 0.68 | Similitud coseno | Configuración por defecto |

### Sistema de Validación de Rostros (v2.1)

**CRÍTICO:** El sistema diferencia entre imágenes en vivo y documentos ID

| Tipo de Imagen | Rostros Permitidos | Comportamiento |
|----------------|-------------------|----------------|
| **Live (en vivo)** | Exactamente 1 | Rechaza si detecta 0, 2+ |
| **ID Document** | 1 o más | Selecciona el más grande |
| **Profile** | Exactamente 1 | Rechaza si detecta 0, 2+ |

**Ejemplo de cédula con 2 fotos:**
```python
# Documento ID detecta 2 rostros:
# - Rostro 1: 15,360 px² (foto principal)
# - Rostro 2: 2,450 px² (foto pequeña/holograma)
# → Sistema selecciona Rostro 1 automáticamente
```

---

## ⚙️ Configuración Avanzada

### Ajustar Umbrales

Edita `face3_corrected.py` líneas 106-113:

```python
self.thresholds = {
    'profile_vs_id': 0.7,      # ↑ más estricto, ↓ más permisivo
    'live_vs_profile': 0.65,   # ↑ más estricto, ↓ más permisivo
    'insightface': 0.4,
    'facenet': 0.7,
    'deepface': 0.68
}
```

**Recomendaciones:**
- **Alta seguridad:** Aumenta a 0.75-0.8
- **Más permisivo:** Reduce a 0.6-0.65
- **Testing:** Usa 0.5 para pruebas iniciales

### Habilitar GPU

Si tienes GPU NVIDIA con CUDA:

```python
# En face3_corrected.py línea ~103
device = 'cuda' if torch.cuda.is_available() else 'cpu'
system = CorrectedFaceRecognitionSystem(device=device)
```

Verifica CUDA:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Configurar CORS

Edita `face3_corrected.py` para permitir solo tu servidor Moodle:

```python
# Línea ~96
from flask_cors import CORS

# Opción 1: Permitir solo Moodle
CORS(app, resources={
    r"/*": {
        "origins": ["http://moodle.example.com", "https://moodle.example.com"]
    }
})

# Opción 2: Permitir todos (solo desarrollo)
CORS(app)  # Actual configuración
```

### Logging Personalizado

Configurar nivel de logging:

```python
import logging

# En face3_corrected.py línea ~97
logging.basicConfig(
    level=logging.INFO,  # Cambiar a DEBUG para más detalle
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('faceid_server.log'),
        logging.StreamHandler()
    ]
)
```

Niveles disponibles:
- `DEBUG`: Todo el detalle
- `INFO`: Operaciones importantes
- `WARNING`: Advertencias
- `ERROR`: Solo errores

---

## 🔒 Seguridad

### Medidas Implementadas

1. **Validación estricta de rostros:**
   - Imágenes en vivo: 1 rostro obligatorio
   - Rechaza múltiples personas automáticamente

2. **Anti-spoofing multi-modelo:**
   - Detección de fotos impresas
   - Detección de pantallas
   - Detección de videos reproducidos

3. **Umbrales conservadores:**
   - Diseñados para minimizar falsos positivos
   - Requieren similitud alta para verificación

4. **Validación de documentos:**
   - IDCardDetector verifica que sea documento real
   - No solo cualquier imagen con rostro

### Recomendaciones de Producción

#### 1. Usar HTTPS

```bash
# Con certificado SSL
python face3_corrected.py --host 0.0.0.0 --port 5001

# Configurar reverse proxy (nginx)
server {
    listen 443 ssl;
    server_name faceid.example.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://127.0.0.1:5001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

#### 2. Firewall

```bash
# Linux (ufw)
sudo ufw allow from 192.168.1.0/24 to any port 5001 proto tcp
sudo ufw enable

# iptables
sudo iptables -A INPUT -p tcp --dport 5001 -s 192.168.1.50 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 5001 -j DROP
```

#### 3. Rate Limiting

Instalar Flask-Limiter:

```bash
pip install Flask-Limiter
```

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

@app.route('/verify', methods=['POST'])
@limiter.limit("10 per minute")
def verify():
    # ...
```

#### 4. Autenticación

Añadir API key:

```python
from functools import wraps

API_KEY = "tu_clave_secreta_aqui"

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        key = request.headers.get('X-API-Key')
        if key != API_KEY:
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/verify', methods=['POST'])
@require_api_key
def verify():
    # ...
```

#### 5. Monitoreo

Usar systemd para auto-restart:

```ini
# /etc/systemd/system/faceid-server.service
[Unit]
Description=Face ID Flask Server
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/RFSERVER
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/python face3_corrected.py --host 0.0.0.0 --port 5001
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Activar:
```bash
sudo systemctl daemon-reload
sudo systemctl enable faceid-server
sudo systemctl start faceid-server
sudo systemctl status faceid-server
```

---

## 🐛 Solución de Problemas

### Error: ModuleNotFoundError

```
ModuleNotFoundError: No module named 'insightface'
```

**Solución:**
```bash
pip install insightface onnxruntime
```

### Error: OMP duplicate library

```
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
```

**Solución:**
Ya está implementado en el código (línea 46-47):
```python
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
```

### Error: CUDA out of memory

**Solución:**
Usar CPU en lugar de GPU:
```python
device = 'cpu'  # En lugar de 'cuda'
```

### Servidor lento

**Causas y soluciones:**

| Problema | Solución |
|----------|----------|
| CPU sin GPU | Considerar usar GPU con CUDA |
| Modelos no optimizados | Usar solo InsightFace + FaceNet |
| Imágenes muy grandes | El servidor ya redimensiona automáticamente |
| Red lenta | Optimizar ancho de banda Moodle ↔ Flask |

### Anti-Spoofing da falsos positivos

**Síntomas:** Rechaza rostros reales

**Soluciones:**
1. Mejorar iluminación del entorno
2. Usar cámara de mejor calidad
3. Ajustar threshold (actualmente 0.5 en línea ~875):
```python
is_real = label == 1 and confidence > 0.4  # Más permisivo
```

### OCR no detecta número

**Síntomas:** `"No se pudo extraer el número de documento"`

**Soluciones:**
1. Mejorar calidad de imagen del documento
2. Asegurar que el número sea claramente visible
3. Verificar patrón de búsqueda en líneas 618-624
4. Revisar logs del servidor para ver texto detectado

### Logs del Servidor

```bash
# Ver logs en tiempo real
tail -f faceid_server.log

# Buscar errores
grep ERROR faceid_server.log

# Buscar verificaciones fallidas
grep "not verified" faceid_server.log
```

---

## 📊 Performance

### Tiempos de Respuesta Típicos

| Endpoint | CPU (i5) | CPU (i7) | GPU (GTX 1060) |
|----------|----------|----------|----------------|
| `/verify` | 2-4s | 1.5-3s | 0.5-1s |
| `/verify-profile` | 3-6s | 2-4s | 0.8-1.5s |
| `/test-antispoofing` | 0.5-1s | 0.3-0.8s | 0.1-0.3s |
| `/test-ocr` | 1.5-3s | 1-2s | 1-2s |

**Nota:** OCR no se acelera significativamente con GPU

### Optimización

**Para máxima velocidad:**
```python
# Usar solo InsightFace (más rápido)
# En CorrectedFaceRecognitionSystem.compare_faces()
# Comentar líneas de FaceNet y DeepFace
```

**Para máxima precisión:**
```python
# Usar consenso de 3 modelos
# Mantener InsightFace + FaceNet + DeepFace
```

---

## 🧪 Testing

### Test Manual de Endpoints

```bash
# 1. Health check
curl http://127.0.0.1:5001/health

# 2. Model info
curl http://127.0.0.1:5001/model-info

# 3. Test anti-spoofing
curl -X POST http://127.0.0.1:5001/test-antispoofing \
  -F "image=@test_images/real_face.jpg"

# 4. Test OCR
curl -X POST http://127.0.0.1:5001/test-ocr \
  -F "image=@test_images/cedula.jpg"

# 5. Verify (requiere Moodle funcionando)
curl -X POST http://127.0.0.1:5001/verify \
  -F "image=@test_images/live.jpg" \
  -F "userid=1" \
  -F "quizid=1" \
  -F "wwwroot=http://localhost/moodle"
```

### Test con Python

```python
import requests

# Test health
response = requests.get('http://127.0.0.1:5001/health')
print(response.json())

# Test anti-spoofing
files = {'image': open('test_image.jpg', 'rb')}
response = requests.post('http://127.0.0.1:5001/test-antispoofing', files=files)
print(response.json())
```

---

## 📄 Licencia

Este proyecto está licenciado bajo **GNU General Public License v3.0**

---

## 🙏 Agradecimientos

- **Silent-Face-Anti-Spoofing** - Modelos MiniFASNet
- **InsightFace** - Modelos ArcFace de última generación
- **FaceNet PyTorch** - Implementación de FaceNet
- **DeepFace** - Framework de reconocimiento facial
- **EasyOCR** - Biblioteca de OCR
- **Flask** - Framework web Python

---

## 📚 Referencias

- [InsightFace Paper](https://arxiv.org/abs/1801.07698)
- [FaceNet Paper](https://arxiv.org/abs/1503.03832)
- [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)
- [EasyOCR Documentation](https://www.jaided.ai/easyocr/)

---

## 📞 Soporte

Si tienes problemas:

1. Revisa [Solución de Problemas](#-solución-de-problemas)
2. Busca en [Issues](https://github.com/Galo45/faceid-flask-server-/issues)
3. Abre un [nuevo Issue](https://github.com/Galo45/faceid-flask-server-/issues/new)

---

**⭐ Si este proyecto te resulta útil, considera darle una estrella en GitHub!**
