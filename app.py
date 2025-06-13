import os
from flask import Flask, render_template, request, redirect, jsonify
from werkzeug.utils import secure_filename
from processing.tumor_detector import TumorDetector
from processing.image_processor import generate_visualizations
from flask import send_file


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
            
            # Procesamiento de imagen
            detector = TumorDetector()
            result = detector.detect_tumor(filepath)
            
            # Visualizaciones
            base_name = os.path.splitext(filename)[0]
            result_images = generate_visualizations(
                filepath, 
                app.config['RESULT_FOLDER'], 
                base_name
            )
            
            # Rutas para el frontend
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


# 🔁 Nuevo endpoint para automatización desde n8n
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

        # Visualización
        base_name = os.path.splitext(filename)[0]
        result_images = generate_visualizations(file_path, app.config['RESULT_FOLDER'], base_name)

        # Devolver solo la primera imagen de resultado
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
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
import io

@app.route('/api/pdf-report', methods=['POST'])
def api_pdf_report():
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

        # Crear PDF en memoria
        pdf_bytes = io.BytesIO()
        c = canvas.Canvas(pdf_bytes, pagesize=letter)
        c.drawString(100, 750, "Resultado del análisis de tumor cerebral")
        c.drawString(100, 730, f"¿Hay tumor?: {'Sí' if result['has_tumor'] else 'No'}")

        if result_images:
            img_path = result_images[0]
            image = ImageReader(img_path)
            c.drawImage(image, 100, 400, width=300, height=300)

        c.save()
        pdf_bytes.seek(0)

        return send_file(pdf_bytes, mimetype='application/pdf', download_name='reporte_tumor.pdf')

    return jsonify({"error": "Archivo no válido"}), 400


# Este bloque es clave para que funcione en Render
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
