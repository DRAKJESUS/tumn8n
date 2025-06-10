import cv2
import numpy as np
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
from processing.tumor_detector import TumorDetector

def generate_visualizations(input_path, output_folder, base_filename):
    """Genera la imagen original, la máscara binaria, la imagen con overlay y un panel comparativo"""

    # 1. Detectar tumor y obtener la máscara
    detector = TumorDetector()
    image_sitk = detector._load_image(input_path)
    processed_image = detector._preprocess(image_sitk)
    mask_sitk = detector._segment(processed_image)

    # Convertir a array para visualización
    image_array = sitk.GetArrayFromImage(processed_image)
    mask_array = sitk.GetArrayFromImage(mask_sitk)

    # 2. Preparar imágenes (usar solo la primera capa si es 3D)
    if len(image_array.shape) == 3:
        image_array = image_array[0]
        mask_array = mask_array[0]

    # Convertir a BGR para overlay
    image_color = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
    overlay = image_color.copy()
    overlay[mask_array == 1] = [0, 0, 255]  # rojo

    # 3. Guardar imágenes individuales
    original_filename = f"{base_filename}_original.png"
    mask_filename = f"{base_filename}_mask.png"
    overlay_filename = f"{base_filename}_overlay.png"
    panel_filename = f"{base_filename}_panel.png"

    cv2.imwrite(os.path.join(output_folder, original_filename), image_color)
    cv2.imwrite(os.path.join(output_folder, mask_filename), (mask_array * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(output_folder, overlay_filename), overlay)

    # 4. Crear figura comparativa con matplotlib
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))
    axs[0].set_title("MRI del Cerebro")
    axs[0].axis('off')

    axs[1].imshow(mask_array, cmap='gray')
    axs[1].set_title("Máscara")
    axs[1].axis('off')

    axs[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axs[2].set_title("MRI con máscara")
    axs[2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, panel_filename))
    plt.close()

    # 5. Devolver rutas relativas para mostrarlas en HTML
    return [
        f"results/{original_filename}",
        f"results/{mask_filename}",
        f"results/{overlay_filename}",
        f"results/{panel_filename}"
    ]
