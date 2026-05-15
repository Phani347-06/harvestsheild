"""
HarvestShield — Shared Image Context
=====================================
Single-load image container that caches all computed representations.
Passed between pipeline stages to eliminate redundant cv2.imread / 
cvtColor / contour operations.

Performance constraint: <= 2s total backend inference on CPU.
This module avoids recomputation by lazily computing and caching
HSV, grayscale, edges, contours, and vegetation masks.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class ImageContext:
    """
    Shared intermediate image representations for all pipeline stages.
    
    Usage:
        ctx = ImageContext.from_path("leaf.jpg")
        hsv = ctx.hsv          # Computed once, cached
        edges = ctx.edges      # Computed on first access, cached
        contours = ctx.contours  # Computed on first access, cached
    """
    
    # Core images (always computed on init)
    original_bgr: np.ndarray = field(repr=False)
    hsv: np.ndarray = field(repr=False)
    gray: np.ndarray = field(repr=False)
    height: int = 0
    width: int = 0
    total_pixels: int = 0
    image_path: str = ""
    
    # Lazily computed caches (None = not yet computed)
    _edges: Optional[np.ndarray] = field(default=None, repr=False)
    _contours: Optional[List[np.ndarray]] = field(default=None, repr=False)
    _vegetation_mask: Optional[np.ndarray] = field(default=None, repr=False)
    _leaf_mask: Optional[np.ndarray] = field(default=None, repr=False)
    _lesion_mask: Optional[np.ndarray] = field(default=None, repr=False)
    _resized_224: Optional[np.ndarray] = field(default=None, repr=False)
    _sobel_mag: Optional[np.ndarray] = field(default=None, repr=False)
    
    # HSV ranges (shared constants)
    HEALTHY_GREEN_LOWER = np.array([35, 40, 40])
    HEALTHY_GREEN_UPPER = np.array([85, 255, 255])
    LESION_BROWN_LOWER = np.array([8, 40, 20])
    LESION_BROWN_UPPER = np.array([25, 255, 180])
    LESION_YELLOW_LOWER = np.array([20, 60, 100])
    LESION_YELLOW_UPPER = np.array([35, 255, 255])
    NECROTIC_LOWER = np.array([0, 0, 10])
    NECROTIC_UPPER = np.array([180, 80, 60])
    
    @classmethod
    def from_path(cls, image_path: str) -> "ImageContext":
        """Create ImageContext from an image file path."""
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        
        return cls(
            original_bgr=bgr,
            hsv=hsv,
            gray=gray,
            height=h,
            width=w,
            total_pixels=h * w,
            image_path=image_path
        )
    
    @classmethod
    def from_array(cls, bgr_array: np.ndarray, path: str = "") -> "ImageContext":
        """Create ImageContext from an already-loaded BGR array."""
        hsv = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        
        return cls(
            original_bgr=bgr_array,
            hsv=hsv,
            gray=gray,
            height=h,
            width=w,
            total_pixels=h * w,
            image_path=path
        )
    
    # ── Lazy Properties ──
    
    @property
    def edges(self) -> np.ndarray:
        """Canny edges — computed once on first access."""
        if self._edges is None:
            self._edges = cv2.Canny(self.gray, 50, 150)
        return self._edges
    
    @property
    def sobel_magnitude(self) -> np.ndarray:
        """Sobel gradient magnitude — computed once on first access."""
        if self._sobel_mag is None:
            sobelx = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=3)
            self._sobel_mag = cv2.magnitude(sobelx, sobely)
        return self._sobel_mag
    
    @property
    def vegetation_mask(self) -> np.ndarray:
        """Green vegetation binary mask — computed once on first access."""
        if self._vegetation_mask is None:
            self._vegetation_mask = cv2.inRange(
                self.hsv, self.HEALTHY_GREEN_LOWER, self.HEALTHY_GREEN_UPPER
            )
        return self._vegetation_mask
    
    @property
    def leaf_mask(self) -> np.ndarray:
        """Full leaf region mask (green + brown + yellow, morphologically closed)."""
        if self._leaf_mask is None:
            green = self.vegetation_mask
            brown = cv2.inRange(self.hsv, self.LESION_BROWN_LOWER, self.LESION_BROWN_UPPER)
            yellow = cv2.inRange(self.hsv, self.LESION_YELLOW_LOWER, self.LESION_YELLOW_UPPER)
            
            combined = cv2.bitwise_or(green, brown)
            combined = cv2.bitwise_or(combined, yellow)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            self._leaf_mask = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        return self._leaf_mask
    
    @property
    def lesion_mask(self) -> np.ndarray:
        """Lesion region mask (brown + yellow + necrotic, within leaf)."""
        if self._lesion_mask is None:
            brown = cv2.inRange(self.hsv, self.LESION_BROWN_LOWER, self.LESION_BROWN_UPPER)
            yellow = cv2.inRange(self.hsv, self.LESION_YELLOW_LOWER, self.LESION_YELLOW_UPPER)
            necrotic = cv2.inRange(self.hsv, self.NECROTIC_LOWER, self.NECROTIC_UPPER)
            
            lesion = cv2.bitwise_or(brown, yellow)
            lesion = cv2.bitwise_or(lesion, necrotic)
            lesion = cv2.bitwise_and(lesion, self.leaf_mask)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            self._lesion_mask = cv2.morphologyEx(lesion, cv2.MORPH_OPEN, kernel)
        return self._lesion_mask
    
    @property
    def contours(self) -> List[np.ndarray]:
        """Lesion contours — computed once on first access."""
        if self._contours is None:
            self._contours, _ = cv2.findContours(
                self.lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
        return self._contours
    
    @property
    def resized_224(self) -> np.ndarray:
        """224x224 resized RGB normalized for CNN input — computed once on first access."""
        if self._resized_224 is None:
            # CNN expects RGB [0,1] with batch dim
            rgb = cv2.cvtColor(self.original_bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (224, 224))
            normalized = resized.astype(np.float32) / 255.0
            self._resized_224 = np.expand_dims(normalized, axis=0)
        return self._resized_224
    
    def get_vegetation_ratio(self) -> float:
        """Fraction of image that is green vegetation (0-1)."""
        return float(np.sum(self.vegetation_mask > 0) / max(self.total_pixels, 1))
    
    def get_leaf_pixel_count(self) -> int:
        """Number of pixels belonging to the leaf region."""
        return int(np.sum(self.leaf_mask > 0))
    
    def get_lesion_pixel_count(self) -> int:
        """Number of pixels identified as lesion."""
        return int(np.sum(self.lesion_mask > 0))
