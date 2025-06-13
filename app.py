import os
import uuid
import io
from flask import Flask, render_template, request, redirect, jsonify, send_file
from werkzeug.utils import secure_filename
from processing.tumor_detector import TumorDetector
from processing.image_processor import generate_visualizations
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader

app = Flask(__name__)

# Configuración de carpetas
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Crear carpetas si no existen
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Interfaz web
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            detector = TumorDetector()
            result = detector.detect_tumor(filepath)

            base_name = os.path.splitext(filename)[0]
            result_images = generate_visualizations(filepath, app.config['RESULT_FOLDER'], base_name)
            
            web_path = os.path.join('uploads', filename).replace('\\', '/')
            result_web_paths = [
                os.path.join('results', os.path.basename(img)).replace('\\', '/')
                for img in result_images
            ]
            
            return render_template('index.html',
                                   original_image=web_path,
                                   result_images=result_web_paths,
                                   has_tumor=result['has_tumor'])
    
    return render_template('index.html')

# API de análisis básica (JSON)
@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No se encontró archivo"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Nombre de archivo vacío"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        detector = TumorDetector()
        result = detector.detect_tumor(file_path)

        base_name = os.path.splitext(filename)[0]
        result_images = generate_visualizations(file_path, app.config['RESULT_FOLDER'], base_name)

        if result_images:
            marked_image_url = f"{request.host_url}static/results/{os.path.basename(result_images[0])}"
        else:
            marked_image_url = None

        return jsonify({
            "resultado": "procesado",
            "hay_tumor": result['has_tumor'],
            "imagen_marcada": marked_image_url
        })

    return jsonify({"error": "Archivo no válido"}), 400

# API que genera PDF con imagen
@app.route('/api/pdf-report', methods=['POST'])
def api_pdf_report():
    if 'file' not in request.files:
        return jsonify({"error": "No se encontró archivo"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Nombre de archivo vacío"}), 400

    if file and allowed_file(file.filename):
        filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        detector = TumorDetector()
        result = detector.detect_tumor(file_path)

        base_name = os.path.splitext(filename)[0]
        result_images = generate_visualizations(file_path, app.config['RESULT_FOLDER'], base_name)

        pdf_bytes = io.BytesIO()
        c = canvas.Canvas(pdf_bytes, pagesize=letter)
        c.setFont("Helvetica", 12)
        c.drawString(100, 750, "📄 Reporte de Detección de Tumores Cerebrales")
        c.drawString(100, 730, f"¿Hay tumor?: {'Sí' if result['has_tumor'] else 'No'}")

        y = 680  # Posición inicial para imágenes
        for rel_path in result_images:
            relative_path = rel_path.replace('results/', '')
            img_path = os.path.abspath(os.path.join(app.config['RESULT_FOLDER'], relative_path))

            if os.path.exists(img_path):
                try:
                    image = ImageReader(img_path)
                    c.drawImage(image, 100, y, width=300, height=180)
                    y -= 190  # Espacio entre imágenes
                    if y < 100:
                        c.showPage()  # Página nueva si se acaba el espacio
                        y = 750
                except Exception as e:
                    c.drawString(100, y, f"⚠ Error al cargar imagen: {str(e)}")
                    y -= 20
            else:
                c.drawString(100, y, f"⚠ Imagen no encontrada: {img_path}")
                y -= 20

        c.save()
        pdf_bytes.seek(0)

        return send_file(pdf_bytes, mimetype='application/pdf', download_name='reporte_tumor.pdf')

    return jsonify({"error": "Archivo no válido"}), 400

# Ejecutar localmente o en Render
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
