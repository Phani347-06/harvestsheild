"""
HarvestShield — Scene Context Analyzer
========================================
Extracts environmental evidence from the FULL original image
(before crop/resize) to support agricultural reasoning.

Detects:
  - Rain (vertical streaks, specular blobs)
  - Fog/Humidity (low contrast + low saturation + soft edges)
  - Blur type classification (camera shake vs environmental moisture)
  - Outdoor field environment (vegetation ratio + lighting)
  - Lighting quality

Design principles:
  - Lightweight: HSV stats, edge density, Hough heuristics only
  - No deep models or expensive segmentation
  - CPU-friendly: < 100ms target
  - Uses shared ImageContext — no redundant image loads
  - Fails gracefully: returns neutral SceneContext on any error
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class SceneContext:
    """Environmental scene analysis output."""
    scene_wetness_score: float = 0.0      # 0-1 overall wetness indicator
    rain_detected: bool = False           # Rain streaks / droplets found
    fog_detected: bool = False            # Atmospheric haze detected
    outdoor_field: bool = True            # Natural vegetation environment
    lighting_quality: str = "good"        # "good" | "overcast" | "low_light"
    blur_type: str = "none"              # "none" | "camera_blur" | "environmental_moisture"
    scene_confidence: float = 0.5         # 0-1 how confident in scene analysis
    blur_penalty_reduction: float = 0.0   # 0-1 how much to reduce quality penalty


def get_neutral_scene() -> SceneContext:
    """Neutral fallback when scene analysis fails or is unavailable."""
    return SceneContext()


class SceneContextAnalyzer:
    """
    Extracts environmental cues from the full-resolution image.
    
    This runs BEFORE crop/resize — it analyzes the original scene
    to detect rain, fog, field conditions, and distinguish environmental
    blur from camera shake.
    
    Uses shared ImageContext to avoid redundant OpenCV operations.
    """
    
    # Thresholds
    FOG_CONTRAST_THRESHOLD = 35.0    # Low std dev in grayscale = hazy
    FOG_SATURATION_THRESHOLD = 50.0  # Low saturation = desaturated/foggy
    LOW_LIGHT_THRESHOLD = 60.0       # Mean brightness below this = low light
    OVERCAST_THRESHOLD = 90.0        # Mean brightness in this range = overcast
    RAIN_LINE_THRESHOLD = 15         # Min Hough lines for rain detection
    VEGETATION_OUTDOOR_THRESHOLD = 0.10  # Min green ratio for outdoor field
    
    def analyze(self, ctx) -> SceneContext:
        """
        Run scene analysis on the shared ImageContext.
        
        Args:
            ctx: ImageContext with pre-computed BGR/HSV/Gray
            
        Returns:
            SceneContext with environmental evidence
        """
        try:
            rain = self._detect_rain(ctx)
            fog = self._detect_fog(ctx)
            lighting = self._assess_lighting(ctx)
            outdoor = self._detect_outdoor_field(ctx)
            blur_type = self._classify_blur_type(ctx, rain, fog)
            wetness = self._compute_wetness_score(ctx, rain, fog)
            
            # Compute blur penalty reduction
            # Environmental blur should NOT reduce confidence as aggressively
            penalty_reduction = 0.0
            if blur_type == "environmental_moisture":
                penalty_reduction = 0.6  # Reduce quality penalty by 60%
            elif fog:
                penalty_reduction = 0.4  # Fog causes moderate penalty reduction
            elif rain:
                penalty_reduction = 0.5  # Rain presence reduces penalty
            
            # Scene confidence based on how much we could analyze
            confidence = self._compute_scene_confidence(ctx, outdoor)
            
            return SceneContext(
                scene_wetness_score=round(wetness, 3),
                rain_detected=rain,
                fog_detected=fog,
                outdoor_field=outdoor,
                lighting_quality=lighting,
                blur_type=blur_type,
                scene_confidence=round(confidence, 3),
                blur_penalty_reduction=round(penalty_reduction, 3)
            )
        except Exception as e:
            print(f"SceneContextAnalyzer warning: {e}")
            return get_neutral_scene()
    
    def _detect_rain(self, ctx) -> bool:
        """
        Detect rain using Hough line transform.
        Rain creates near-vertical bright streaks in the image.
        """
        try:
            # Look for bright vertical-ish lines
            # Rain streaks are typically high-intensity, near-vertical
            edges = ctx.edges
            
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180,
                threshold=50,
                minLineLength=max(ctx.height // 10, 20),
                maxLineGap=10
            )
            
            if lines is None:
                return False
            
            # Count near-vertical lines (angle > 70 degrees from horizontal)
            vertical_count = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                if dy > 0 and dx / max(dy, 1) < 0.4:  # Near vertical
                    vertical_count += 1
            
            return vertical_count >= self.RAIN_LINE_THRESHOLD
        except Exception:
            return False
    
    def _detect_fog(self, ctx) -> bool:
        """
        Detect fog/haze via low contrast + low saturation + softened edges.
        Foggy images have globally reduced contrast and desaturated colors.
        """
        try:
            # Check grayscale contrast (standard deviation)
            contrast = float(np.std(ctx.gray))
            
            # Check average saturation
            saturation = float(np.mean(ctx.hsv[:, :, 1]))
            
            # Check edge density (fog reduces visible edges)
            edge_density = float(np.sum(ctx.edges > 0) / max(ctx.total_pixels, 1))
            
            is_low_contrast = contrast < self.FOG_CONTRAST_THRESHOLD
            is_low_saturation = saturation < self.FOG_SATURATION_THRESHOLD
            is_low_edges = edge_density < 0.05  # Very few edges
            
            # Fog requires at least 2 of 3 indicators
            fog_signals = sum([is_low_contrast, is_low_saturation, is_low_edges])
            return fog_signals >= 2
        except Exception:
            return False
    
    def _assess_lighting(self, ctx) -> str:
        """Classify lighting quality from grayscale mean brightness."""
        try:
            mean_brightness = float(np.mean(ctx.gray))
            
            if mean_brightness < self.LOW_LIGHT_THRESHOLD:
                return "low_light"
            elif mean_brightness < self.OVERCAST_THRESHOLD:
                return "overcast"
            else:
                return "good"
        except Exception:
            return "good"
    
    def _detect_outdoor_field(self, ctx) -> bool:
        """
        Detect if the image was taken in an outdoor field environment.
        Uses green vegetation ratio as primary indicator.
        """
        try:
            veg_ratio = ctx.get_vegetation_ratio()
            return veg_ratio >= self.VEGETATION_OUTDOOR_THRESHOLD
        except Exception:
            return True  # Default to outdoor assumption
    
    def _classify_blur_type(self, ctx, rain_detected: bool, fog_detected: bool) -> str:
        """
        Distinguish camera shake blur from environmental moisture blur.
        
        Camera blur: directional, affects entire image uniformly
        Environmental: localized soft patches, often with moisture indicators
        
        This is CRITICAL — environmental blur should NOT reduce confidence
        as aggressively as camera shake.
        """
        try:
            # Compute Laplacian variance (sharpness)
            laplacian_var = float(cv2.Laplacian(ctx.gray, cv2.CV_64F).var())
            
            # If image is sharp enough, no blur classification needed
            if laplacian_var > 200:
                return "none"
            
            # If rain or fog is detected, classify as environmental
            if rain_detected or fog_detected:
                return "environmental_moisture"
            
            # Check for directional blur (camera shake indicator)
            # Camera shake creates directional smearing in Fourier domain
            # Simple heuristic: check if blur is uniform across image quadrants
            h, w = ctx.gray.shape
            quad_vars = []
            for qy in [0, h // 2]:
                for qx in [0, w // 2]:
                    quad = ctx.gray[qy:qy + h // 2, qx:qx + w // 2]
                    qvar = cv2.Laplacian(quad, cv2.CV_64F).var()
                    quad_vars.append(qvar)
            
            # Uniform low sharpness across quadrants = camera blur
            # Uneven sharpness = could be environmental / depth-of-field
            variance_of_variances = float(np.var(quad_vars))
            mean_var = float(np.mean(quad_vars))
            
            if mean_var < 100:
                # Image is blurry
                if variance_of_variances < mean_var * 0.5:
                    # Uniform blur = likely camera shake
                    return "camera_blur"
                else:
                    # Non-uniform = could be depth/environmental
                    # Check moisture indicators
                    saturation = float(np.mean(ctx.hsv[:, :, 1]))
                    if saturation < 70:
                        return "environmental_moisture"
                    return "camera_blur"
            
            return "none"
        except Exception:
            return "none"
    
    def _compute_wetness_score(self, ctx, rain: bool, fog: bool) -> float:
        """
        Composite scene wetness indicator.
        Combines rain detection, fog, saturation, and specular highlights.
        """
        try:
            score = 0.0
            
            if rain:
                score += 0.40
            if fog:
                score += 0.25
            
            # Check for specular highlights (water droplets / wet surfaces)
            # Very bright spots with low saturation = reflective wet surfaces
            value_channel = ctx.hsv[:, :, 2]
            sat_channel = ctx.hsv[:, :, 1]
            
            bright_low_sat = np.sum(
                (value_channel > 220) & (sat_channel < 40)
            ) / max(ctx.total_pixels, 1)
            
            # Specular highlights contribute to wetness
            score += min(bright_low_sat * 50, 0.35)
            
            return float(min(score, 1.0))
        except Exception:
            return 0.0
    
    def _compute_scene_confidence(self, ctx, outdoor: bool) -> float:
        """
        How confident are we in the scene analysis?
        Higher for outdoor field images with clear features.
        """
        try:
            confidence = 0.5  # Base confidence
            
            # Outdoor field → higher confidence in environmental analysis
            if outdoor:
                confidence += 0.20
            
            # Good contrast → features are more analyzable
            contrast = float(np.std(ctx.gray))
            if contrast > 40:
                confidence += 0.15
            
            # Reasonable image size → better analysis
            if ctx.total_pixels > 100000:  # > ~316x316
                confidence += 0.15
            
            return float(min(confidence, 1.0))
        except Exception:
            return 0.5
