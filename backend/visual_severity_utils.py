"""
HarvestShield — Visual Disease Severity Analysis
=================================================
OpenCV-based module that estimates actual disease severity
from leaf images to prevent overconfident classification
on mildly affected or healthy-looking leaves.

Provides:
  1. Lesion area percentage
  2. Discoloration severity
  3. Edge damage detection
  4. Symptom density scoring
  5. Composite severity classification
  6. Visual realism score (influences CNN trust)
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class SeverityReport:
    """Complete visual severity assessment of a leaf image."""
    lesion_area_pct: float        # % of leaf area with lesions (0-100)
    discoloration_score: float    # Severity of color deviation (0-1)
    edge_damage_score: float      # Border/edge necrosis intensity (0-1)
    symptom_density: float        # How clustered symptoms are (0-1)
    texture_abnormality: float    # Abnormal texture patterns (0-1)

    severity_level: str           # healthy_appearing | mild | moderate | severe
    severity_score: float         # Composite severity (0-1)
    realism_score: float          # How "real" the disease appears visually (0-1)


class VisualSeverityAnalyzer:
    """
    Analyzes leaf images to estimate actual disease severity,
    independent of CNN classification confidence.
    
    This module acts as a reality check — if the CNN says "90% late blight"
    but the leaf looks mostly green with no visible lesions, the realism
    score will be low, triggering skepticism in the fusion engine.
    """

    # HSV ranges for detecting common leaf disease symptoms
    # Brown/dark lesions (late blight, early blight)
    LESION_BROWN_LOWER = np.array([8, 40, 20])
    LESION_BROWN_UPPER = np.array([25, 255, 180])
    
    # Yellowing (chlorosis)
    LESION_YELLOW_LOWER = np.array([20, 60, 100])
    LESION_YELLOW_UPPER = np.array([35, 255, 255])
    
    # Dark necrotic spots
    NECROTIC_LOWER = np.array([0, 0, 10])
    NECROTIC_UPPER = np.array([180, 80, 60])

    # Healthy green range (for baseline comparison)
    HEALTHY_GREEN_LOWER = np.array([35, 40, 40])
    HEALTHY_GREEN_UPPER = np.array([85, 255, 255])

    # Severity classification thresholds (lesion area %)
    THRESHOLD_MILD = 5.0
    THRESHOLD_MODERATE = 15.0
    THRESHOLD_SEVERE = 35.0

    def __init__(self, image_path: Optional[str] = None, image_array: Optional[np.ndarray] = None):
        if image_array is not None:
            self.image = image_array
        elif image_path is not None:
            self.image = cv2.imread(image_path)
            if self.image is None:
                raise ValueError(f"Could not read image at {image_path}")
        else:
            # Placeholder for orchestrator late-binding
            self.image = None
            
        if self.image is not None:
            self._init_representations()

    def _init_representations(self):
        self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.h, self.w = self.gray.shape[:2]
        self.total_pixels = self.h * self.w

    def analyze_severity(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Convenience method for orchestrator to analyze a raw BGR array."""
        self.image = image_array
        self._init_representations()
        report = self.analyze()
        
        # Convert dataclass to dict for easier JSON serialization in orchestrator
        return {
            "lesion_area_pct": report.lesion_area_pct,
            "discoloration": report.discoloration_score,
            "edge_damage": report.edge_damage_score,
            "symptom_density": report.symptom_density,
            "severity_level": report.severity_level,
            "severity_score": report.severity_score,
            "realism_score": report.realism_score
        }

    def analyze(self) -> SeverityReport:
        """Run the full severity analysis pipeline."""
        # Step 1: Isolate the leaf region (green + diseased areas)
        leaf_mask = self._get_leaf_mask()
        leaf_pixels = max(np.sum(leaf_mask > 0), 1)

        # Step 2: Detect lesion areas
        lesion_mask = self._detect_lesions(leaf_mask)
        lesion_pixels = np.sum(lesion_mask > 0)
        lesion_area_pct = (lesion_pixels / leaf_pixels) * 100.0

        # Step 3: Measure discoloration
        discoloration = self._measure_discoloration(leaf_mask)

        # Step 4: Detect edge damage
        edge_damage = self._detect_edge_damage(lesion_mask)

        # Step 5: Compute symptom density
        density = self._compute_symptom_density(lesion_mask, leaf_mask)

        # Step 6: Compute texture abnormality
        texture_abnormality = self._compute_texture_abnormality(leaf_mask, lesion_mask)

        # Step 7: Classify severity
        severity_level = self._classify_severity(lesion_area_pct, discoloration, edge_damage)
        severity_score = self._compute_severity_score(lesion_area_pct, discoloration, edge_damage, density)

        # Step 8: Compute realism score
        realism_score = self._compute_realism_score(
            lesion_area_pct, discoloration, texture_abnormality, density, severity_score
        )

        return SeverityReport(
            lesion_area_pct=round(lesion_area_pct, 2),
            discoloration_score=round(discoloration, 3),
            edge_damage_score=round(edge_damage, 3),
            symptom_density=round(density, 3),
            texture_abnormality=round(texture_abnormality, 3),
            severity_level=severity_level,
            severity_score=round(severity_score, 3),
            realism_score=round(realism_score, 3)
        )

    # ── Internal Methods ──

    def _get_leaf_mask(self) -> np.ndarray:
        """
        Create a mask of the entire leaf region (healthy + diseased).
        Combines green detection with brown/yellow detection,
        then fills holes to capture the whole leaf.
        """
        green_mask = cv2.inRange(self.hsv, self.HEALTHY_GREEN_LOWER, self.HEALTHY_GREEN_UPPER)
        brown_mask = cv2.inRange(self.hsv, self.LESION_BROWN_LOWER, self.LESION_BROWN_UPPER)
        yellow_mask = cv2.inRange(self.hsv, self.LESION_YELLOW_LOWER, self.LESION_YELLOW_UPPER)
        
        combined = cv2.bitwise_or(green_mask, brown_mask)
        combined = cv2.bitwise_or(combined, yellow_mask)
        
        # Morphological closing to fill small gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        
        return combined

    def _detect_lesions(self, leaf_mask: np.ndarray) -> np.ndarray:
        """Detect brown, yellow, and necrotic regions within the leaf."""
        brown = cv2.inRange(self.hsv, self.LESION_BROWN_LOWER, self.LESION_BROWN_UPPER)
        yellow = cv2.inRange(self.hsv, self.LESION_YELLOW_LOWER, self.LESION_YELLOW_UPPER)
        necrotic = cv2.inRange(self.hsv, self.NECROTIC_LOWER, self.NECROTIC_UPPER)
        
        lesion_mask = cv2.bitwise_or(brown, yellow)
        lesion_mask = cv2.bitwise_or(lesion_mask, necrotic)
        
        # Only count lesions that are within the leaf region
        lesion_mask = cv2.bitwise_and(lesion_mask, leaf_mask)
        
        # Remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, kernel)
        
        return lesion_mask

    def _measure_discoloration(self, leaf_mask: np.ndarray) -> float:
        """
        Measure how far the leaf's color deviates from healthy green.
        Returns 0 (perfectly green) to 1 (severely discolored).
        """
        # Extract hue channel within the leaf region
        hue = self.hsv[:, :, 0]
        sat = self.hsv[:, :, 1]
        
        leaf_pixels = leaf_mask > 0
        if not np.any(leaf_pixels):
            return 0.0
        
        leaf_hues = hue[leaf_pixels]
        leaf_sats = sat[leaf_pixels]
        
        # Healthy green hue is roughly 35-85 in OpenCV HSV
        healthy_center = 60.0
        hue_deviations = np.abs(leaf_hues.astype(float) - healthy_center) / 90.0
        
        # Low saturation also indicates damage (pale/necrotic tissue)
        sat_penalty = np.where(leaf_sats < 50, 0.3, 0.0)
        
        discoloration = np.mean(np.clip(hue_deviations + sat_penalty, 0, 1))
        return float(discoloration)

    def _detect_edge_damage(self, lesion_mask: np.ndarray) -> float:
        """
        Detect necrotic damage near leaf edges.
        High edge damage suggests more advanced disease progression.
        """
        # Create an edge region (border of the leaf)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        dilated = cv2.dilate(lesion_mask, kernel)
        
        # Find edges in the image
        edges = cv2.Canny(self.gray, 50, 150)
        
        # Overlap of edges and dilated lesion regions
        edge_near_lesion = cv2.bitwise_and(edges, dilated)
        
        total_edge = max(np.sum(edges > 0), 1)
        damaged_edge = np.sum(edge_near_lesion > 0)
        
        return float(min(damaged_edge / total_edge, 1.0))

    def _compute_symptom_density(self, lesion_mask: np.ndarray, leaf_mask: np.ndarray) -> float:
        """
        Measure how concentrated (dense) vs. scattered the symptoms are.
        Dense clusters suggest active infection; scattered spots may be noise.
        """
        contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return 0.0
        
        # Compute areas of each lesion cluster
        areas = [cv2.contourArea(c) for c in contours]
        total_lesion_area = sum(areas)
        
        if total_lesion_area == 0:
            return 0.0
        
        # Density = how much area is in the largest cluster vs. total
        # High density (few big clusters) → more concerning
        # Low density (many tiny spots) → less concerning
        largest_cluster = max(areas)
        concentration = largest_cluster / total_lesion_area
        
        # Also factor in number of clusters relative to total area
        num_clusters = len([a for a in areas if a > 50])  # Filter noise
        leaf_area = max(np.sum(leaf_mask > 0), 1)
        
        # Density score combines concentration and meaningful cluster count
        if num_clusters == 0:
            return 0.0
        
        cluster_score = min(num_clusters / 10.0, 1.0)  # More clusters = more spread
        density = 0.6 * concentration + 0.4 * cluster_score
        
        return float(min(density, 1.0))

    def _compute_texture_abnormality(self, leaf_mask: np.ndarray, lesion_mask: np.ndarray) -> float:
        """
        Compare texture in healthy vs. lesioned regions.
        Diseased tissue typically has different texture patterns.
        """
        # Compute Sobel gradient magnitude
        sobelx = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = cv2.magnitude(sobelx, sobely)
        
        # Healthy region = leaf minus lesion
        healthy_mask = cv2.bitwise_and(leaf_mask, cv2.bitwise_not(lesion_mask))
        
        healthy_pixels = healthy_mask > 0
        lesion_pixels = lesion_mask > 0
        
        if not np.any(healthy_pixels) or not np.any(lesion_pixels):
            return 0.0
        
        healthy_texture = np.mean(mag[healthy_pixels])
        lesion_texture = np.mean(mag[lesion_pixels])
        
        # Abnormality = how different the textures are
        max_texture = max(healthy_texture, lesion_texture, 1.0)
        abnormality = abs(lesion_texture - healthy_texture) / max_texture
        
        return float(min(abnormality, 1.0))

    def _classify_severity(self, lesion_pct: float, discoloration: float, edge_damage: float) -> str:
        """Classify into discrete severity levels."""
        # Use multiple signals, not just lesion area
        composite = lesion_pct + (discoloration * 15) + (edge_damage * 10)
        
        if composite < self.THRESHOLD_MILD:
            return "healthy_appearing"
        elif composite < self.THRESHOLD_MODERATE:
            return "mild"
        elif composite < self.THRESHOLD_SEVERE:
            return "moderate"
        else:
            return "severe"

    def _compute_severity_score(self, lesion_pct: float, discoloration: float, 
                                 edge_damage: float, density: float) -> float:
        """Composite severity score (0-1)."""
        # Normalize lesion area to 0-1 (50%+ is max)
        normalized_lesion = min(lesion_pct / 50.0, 1.0)
        
        score = (
            normalized_lesion * 0.40 +
            discoloration * 0.25 +
            edge_damage * 0.15 +
            density * 0.20
        )
        return float(min(score, 1.0))

    def _compute_realism_score(self, lesion_pct: float, discoloration: float,
                                texture_abnormality: float, density: float,
                                severity_score: float) -> float:
        """
        Visual Realism Score — how "real" does the disease look?
        
        A high realism score means the image shows clear, consistent
        visual evidence of disease. A low score means the CNN's disease
        prediction is not well-supported by visual symptoms.
        
        This directly influences CNN trust in the fusion engine.
        """
        # If severity is very low, realism should also be low
        # (no visible symptoms → CNN prediction may be wrong)
        if severity_score < 0.05:
            return 0.1  # Minimal realism — leaf looks healthy
        
        # Realism factors:
        # 1. Visible lesions must be present
        lesion_factor = min(lesion_pct / 20.0, 1.0)  # 20%+ lesion = full credit
        
        # 2. Color must be noticeably abnormal
        color_factor = min(discoloration / 0.4, 1.0)
        
        # 3. Texture should differ between healthy and lesioned regions
        texture_factor = min(texture_abnormality / 0.5, 1.0)
        
        # 4. Symptoms should show some clustering pattern (not random noise)
        pattern_factor = min(density / 0.5, 1.0)
        
        realism = (
            lesion_factor * 0.35 +
            color_factor * 0.30 +
            texture_factor * 0.15 +
            pattern_factor * 0.20
        )
        
        return float(min(max(realism, 0.05), 1.0))
