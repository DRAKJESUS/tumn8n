import cv2
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

class TumorDetector:
    def detect_tumor(self, image_path):
        """Detecta tumores cerebrales con autoajuste por imagen"""
        try:
            image = self._load_image(image_path)
            processed = self._preprocess(image)
            segmented = self._segment(processed)
            has_tumor = self._analyze(processed, segmented)

            return {
                'has_tumor': has_tumor,
                'image_shape': processed.GetSize()
            }
        except Exception as e:
            print(f"Error en detección: {str(e)}")
            return {'has_tumor': False}

    def _load_image(self, path):
        """Carga la imagen en formato adecuado"""
        if path.lower().endswith('.dcm'):
            return sitk.ReadImage(path)
        else:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            return sitk.GetImageFromArray(img)

    def _preprocess(self, image):
        """Mejora la calidad de la imagen"""
        array = sitk.GetArrayFromImage(image)

        # Normalización a 8 bits
        array = cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Reducción de ruido
        if len(array.shape) == 3:
            for i in range(array.shape[0]):
                array[i] = cv2.fastNlMeansDenoising(array[i], None, h=10)
        else:
            array = cv2.fastNlMeansDenoising(array, None, h=10)

        # Contraste local con CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        if len(array.shape) == 3:
            for i in range(array.shape[0]):
                array[i] = clahe.apply(array[i])
        else:
            array = clahe.apply(array)

        return sitk.GetImageFromArray(array)

    def _segment(self, image):
        """Segmenta automáticamente áreas tumorales según intensidad y posición"""
        array = sitk.GetArrayFromImage(image)

        # Si es 3D, usar el slice con más intensidad
        if len(array.shape) == 3:
            slice_idx = np.argmax(np.sum(array, axis=(1, 2)))
            array = array[slice_idx]

        # Normalización y contraste local
        norm = cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        norm = clahe.apply(norm)

        # Umbral dinámico por percentil (ajuste automático)
        threshold_val = np.percentile(norm, 85)
        _, thresh = cv2.threshold(norm, threshold_val, 255, cv2.THRESH_BINARY)

        # Limpieza morfológica
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # Componentes conectados
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened)
        h, w = opened.shape
        total_pixels = h * w
        min_area = int(total_pixels * 0.001)   # 0.1% de la imagen
        max_area = int(total_pixels * 0.05)    # hasta 5% de la imagen

        final_mask = np.zeros_like(opened)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            cx, cy = centroids[i]

            # Filtro inteligente por posición y tamaño
            if (min_area <= area <= max_area and
                0.25 * w < cx < 0.80 * w and
                0.25 * h < cy < 0.80 * h):
                final_mask[labels == i] = 255

        return sitk.GetImageFromArray((final_mask > 0).astype(np.uint8))

    def _analyze(self, image, mask):
        """Evalúa si hay tumor basándose en cantidad y contraste"""
        image_array = sitk.GetArrayFromImage(image)
        mask_array = sitk.GetArrayFromImage(mask)

        if len(image_array.shape) == 3:
            image_array = image_array[0]
            mask_array = mask_array[0]

        tumor_pixels = np.sum(mask_array)
        total_pixels = mask_array.size
        tumor_ratio = tumor_pixels / total_pixels

        masked_image = image_array * mask_array
        std_intensity = np.std(masked_image[mask_array > 0]) if tumor_pixels > 0 else 0

        has_tumor = tumor_ratio > 0.01 and std_intensity > 20

        return has_tumor
