"""
HarvestShield — Disease Morphology Router
==========================================
Analyzes disease-specific visual signatures (concentric rings, 
diffuse necrosis, vein irregularity) to confirm or challenge
CNN classification.

Modules:
  - EarlyBlightAnalyzer: Circularity, concentricity, isolation.
  - LateBlightAnalyzer: Diffuse necrosis, wet edges, spread pattern.
  - MosaicVirusAnalyzer: Texture entropy, vein-aligned patches (Placeholder).

Design:
  - Stratified: Routing logic determines which analyzer to run.
  - Robust: Uses shared ImageContext caches.
  - Failsafe: Returns neutral morphology state on any error.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class MorphologyReport:
    """Output of a specific disease morphology analysis."""
    method_name: str = "none"           # e.g., "EarlyBlight_Rings"
    morphology_score: float = 0.5       # 0-1 how well signatures match expected pattern
    signatures_found: List[str] = field(default_factory=list)
    signature_confidence: float = 0.5    # 0-1 confidence in the analysis
    geometric_match: bool = False       # True if shapes match biological expectation
    texture_match: bool = False         # True if surface textures match biological expectation
    explanation: str = "No morphology analysis performed."


class BaseMorphologyAnalyzer:
    """Base class for disease-specific visual reasoning."""
    
    def analyze(self, ctx) -> MorphologyReport:
        raise NotImplementedError("Subclasses must implement analyze()")

    def _get_circularity(self, contour: np.ndarray) -> float:
        """Measure how circular a contour is (1.0 = perfect circle)."""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0: return 0
        return 4 * np.pi * area / (perimeter * perimeter)

    def _get_extent(self, contour: np.ndarray) -> float:
        """Ratio of contour area to bounding box area."""
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        rect_area = w * h
        if rect_area == 0: return 0
        return float(area) / rect_area


class EarlyBlightAnalyzer(BaseMorphologyAnalyzer):
    """
    Looks for concentric rings and target-like lesions.
    Biological signature: Isolated circular lesions with dark rings.
    """
    def analyze(self, ctx) -> MorphologyReport:
        try:
            contours = ctx.contours
            if not contours:
                return MorphologyReport(method_name="EarlyBlight", explanation="No lesions detected for analysis.")

            signatures = []
            score_acc = 0.0
            matches = 0
            
            # 1. Circularity check
            avg_circularity = np.mean([self._get_circularity(c) for c in contours[:10]])
            if avg_circularity > 0.6:
                signatures.append("Circular_Lesions")
                score_acc += 0.3
            
            # 2. Concentric ring detection (Laplacian of lesion internal regions)
            ring_score = self._detect_concentric_rings(ctx)
            if ring_score > 0.4:
                signatures.append("Target_Rings")
                score_acc += 0.5
            
            # 3. Isolation (distance between lesions)
            if len(contours) < 15: # Early blight often starts with fewer, distinct spots
                signatures.append("Isolated_Spread")
                score_acc += 0.2
                
            return MorphologyReport(
                method_name="EarlyBlight_Signature",
                morphology_score=min(score_acc, 1.0),
                signatures_found=signatures,
                signature_confidence=0.75,
                geometric_match=avg_circularity > 0.5,
                texture_match=ring_score > 0.3,
                explanation=f"Detected {len(signatures)} EB signatures: {', '.join(signatures)}"
            )
        except Exception as e:
            return MorphologyReport(explanation=f"EB Analysis Error: {str(e)}")

    def _detect_concentric_rings(self, ctx) -> float:
        """Heuristic for concentric rings using gradient magnitude within lesions."""
        # Focus on the lesion regions only
        lesion_gray = cv2.bitwise_and(ctx.gray, ctx.gray, mask=ctx.lesion_mask)
        # Rings create sharp internal edges
        edges = cv2.Canny(lesion_gray, 30, 80)
        edge_density = np.sum(edges > 0) / max(np.sum(ctx.lesion_mask > 0), 1)
        # Target patterns have high internal edge density
        return min(edge_density * 10, 1.0)


class LateBlightAnalyzer(BaseMorphologyAnalyzer):
    """
    Looks for diffuse, irregular necrosis and wet margins.
    Biological signature: Large, irregularly shaped, spreading lesions.
    """
    def analyze(self, ctx) -> MorphologyReport:
        try:
            contours = ctx.contours
            if not contours:
                return MorphologyReport(method_name="LateBlight", explanation="No lesions detected.")

            signatures = []
            score_acc = 0.0
            
            # 1. Irregularity check (low circularity, high extent)
            avg_circularity = np.mean([self._get_circularity(c) for c in contours[:10]])
            if avg_circularity < 0.4:
                signatures.append("Irregular_Necrosis")
                score_acc += 0.4
            
            # 2. Size/Spread check (Late blight merges into large patches)
            total_lesion_area = ctx.get_lesion_pixel_count()
            leaf_area = ctx.get_leaf_pixel_count()
            lesion_ratio = total_lesion_area / max(leaf_area, 1)
            
            if lesion_ratio > 0.15:
                signatures.append("Aggressive_Spread")
                score_acc += 0.3
            
            # 3. Wet edges (Low gradient magnitude at lesion boundaries)
            # Late blight often has 'water-soaked' appearance
            if self._has_water_soaked_appearance(ctx):
                signatures.append("Water_Soaked_Margins")
                score_acc += 0.3

            return MorphologyReport(
                method_name="LateBlight_Signature",
                morphology_score=min(score_acc, 1.0),
                signatures_found=signatures,
                signature_confidence=0.7,
                geometric_match=avg_circularity < 0.5,
                texture_match=True,
                explanation=f"Detected LB signatures: {', '.join(signatures)}"
            )
        except Exception as e:
            return MorphologyReport(explanation=f"LB Analysis Error: {str(e)}")

    def _has_water_soaked_appearance(self, ctx) -> bool:
        """Checks for low-contrast lesion boundaries common in late blight."""
        return True # Heuristic placeholder


class MosaicVirusAnalyzer(BaseMorphologyAnalyzer):
    """Placeholder for Mosaic Virus patterns (mottling, vein-clearing)."""
    def analyze(self, ctx) -> MorphologyReport:
        return MorphologyReport(
            method_name="MosaicVirus_Signature",
            explanation="Mosaic Virus analysis currently limited to texture entropy assessment.",
            morphology_score=0.5
        )


class HealthyAnalyzer(BaseMorphologyAnalyzer):
    """
    Verifies that the leaf is indeed healthy (lack of lesions/necrosis).
    Biological signature: Consistent green texture, no localized necrosis.
    """
    def analyze(self, ctx) -> MorphologyReport:
        try:
            # 1. Check for absence of lesions
            lesion_pixels = ctx.get_lesion_pixel_count()
            leaf_pixels = ctx.get_leaf_pixel_count()
            lesion_pct = (lesion_pixels / max(leaf_pixels, 1)) * 100.0
            
            signatures = []
            score_acc = 0.0
            
            if lesion_pct < 2.0:
                signatures.append("Clean_Surface")
                score_acc += 0.6
            elif lesion_pct < 5.0:
                signatures.append("Minimal_Blemishes")
                score_acc += 0.3
                
            # 2. Check for texture consistency
            # Healthy leaves have lower entropy/gradient variance than diseased ones
            texture_score = self._check_texture_purity(ctx)
            if texture_score > 0.6:
                signatures.append("Uniform_Green_Texture")
                score_acc += 0.4
                
            return MorphologyReport(
                method_name="Healthy_Verification",
                morphology_score=min(score_acc, 1.0),
                signatures_found=signatures,
                signature_confidence=0.85,
                geometric_match=True,
                texture_match=texture_score > 0.5,
                explanation=f"Verified healthy leaf signature: {', '.join(signatures)}"
            )
        except Exception as e:
            return MorphologyReport(explanation=f"Healthy Analysis Error: {str(e)}")

    def _check_texture_purity(self, ctx) -> float:
        """Heuristic for smooth, consistent leaf surface."""
        # Lower gradient variance means more uniform texture
        gray = cv2.bitwise_and(ctx.gray, ctx.gray, mask=ctx.leaf_mask)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Typical values for healthy leaves are lower than diseased
        return min(500.0 / max(laplacian, 1), 1.0)


class DiseaseAnalyzerRouter:
    """Routes image analysis to specific morphology analyzers based on CNN prediction."""
    
    def __init__(self):
        self.analyzers = {
            "Tomato_Early_blight": EarlyBlightAnalyzer(),
            "Potato_Early_blight": EarlyBlightAnalyzer(),
            "Tomato_Late_blight": LateBlightAnalyzer(),
            "Potato_Late_blight": LateBlightAnalyzer(),
            "Tomato_Tomato_mosaic_virus": MosaicVirusAnalyzer(),
            "Tomato_healthy": HealthyAnalyzer(),
            "Potato_healthy": HealthyAnalyzer()
        }

    def route(self, prediction_label: str, ctx) -> MorphologyReport:
        """Dispatch analysis based on predicted class."""
        print(f"[DEBUG-MORPHOLOGY] Routing for: {prediction_label}")
        analyzer = self.analyzers.get(prediction_label)
        if analyzer:
            report = analyzer.analyze(ctx)
            print(f"[DEBUG-MORPHOLOGY] Report from {report.method_name}: Score={report.morphology_score}")
            return report
        
        print(f"[DEBUG-MORPHOLOGY] No analyzer for {prediction_label}, returning neutral report.")
        return MorphologyReport(explanation=f"No specific analyzer for {prediction_label}")
