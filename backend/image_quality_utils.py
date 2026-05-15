import cv2
import numpy as np

class ImageDiagnostics:
    """
    Advanced Image Diagnostics for HarvestShield.
    Analyzes multiple dimensions of image quality to support probabilistic fusion.
    """

    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not read image at {image_path}")
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

    def analyze_blur(self):
        """Variance of Laplacian for sharpness."""
        variance = cv2.Laplacian(self.gray, cv2.CV_64F).var()
        # Normalize to 0-1 (300+ is usually very sharp)
        return min(variance / 500.0, 1.0)

    def analyze_brightness(self):
        """Mean intensity of the image."""
        mean = np.mean(self.gray)
        # Ideal brightness is around 127. 
        # Score penalizes very dark or very bright images.
        score = 1.0 - abs(mean - 127) / 127
        return max(0.0, score)

    def analyze_contrast(self):
        """Standard deviation of pixel intensities."""
        std = np.std(self.gray)
        # Normalize: std of 60+ is usually good contrast
        return min(std / 80.0, 1.0)

    def estimate_leaf_visibility(self):
        """Green color masking to estimate leaf presence."""
        # Range for green color in HSV
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(self.hsv, lower_green, upper_green)
        visibility = np.sum(mask > 0) / mask.size
        return min(visibility * 2.0, 1.0)  # Scale so 50% green is 1.0 visibility

    def analyze_background_complexity(self):
        """Edge density as a proxy for background clutter."""
        edges = cv2.Canny(self.gray, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size
        # High density often means complex/cluttered background
        # We return a 'simplicity' score where high is better
        simplicity = 1.0 - min(edge_density * 5.0, 1.0)
        return simplicity

    def analyze_texture_clarity(self):
        """Sobel gradients to measure fine texture detail."""
        sobelx = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = cv2.magnitude(sobelx, sobely)
        texture_score = np.mean(mag)
        # Normalize: mean mag of 20+ is good texture
        return min(texture_score / 40.0, 1.0)

    def get_quality_report(self):
        """
        Returns a comprehensive quality report.
        """
        blur = self.analyze_blur()
        brightness = self.analyze_brightness()
        contrast = self.analyze_contrast()
        leaf_vis = self.estimate_leaf_visibility()
        simplicity = self.analyze_background_complexity()
        texture = self.analyze_texture_clarity()

        # Weighted aggregate score for backward compatibility
        aggregate = (
            blur * 0.35 +
            brightness * 0.15 +
            contrast * 0.15 +
            leaf_vis * 0.20 +
            texture * 0.15
        ) * simplicity

        return {
            "blur_score": round(blur, 2),
            "brightness_score": round(brightness, 2),
            "contrast_score": round(contrast, 2),
            "leaf_visibility": round(leaf_vis, 2),
            "background_simplicity": round(simplicity, 2),
            "texture_clarity": round(texture, 2),
            "aggregate_quality": round(aggregate, 2)
        }


def analyze_image_quality(image_path):
    """Convenience wrapper for the ImageDiagnostics class."""
    diag = ImageDiagnostics(image_path)
    return diag.get_quality_report()
