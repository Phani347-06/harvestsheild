"""
HarvestShield — CNN Confidence Calibration Module
===================================================
Transforms raw overconfident softmax probabilities into calibrated,
uncertainty-aware predictions using multi-stage post-hoc calibration.

Techniques:
  1. Temperature Scaling — softens peaked softmax distributions
  2. Entropy Analysis — quantifies prediction uncertainty
  3. Margin-Based Reduction — penalizes narrow top-1 vs top-2 gaps
  4. Quality-Aware Dampening — reduces trust when image quality is poor
     (now context-aware: dampening is reduced when disease is visually obvious)
  5. Healthy/Diseased Gate — preliminary disease-presence check
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class CalibratedPrediction:
    """Output of the confidence calibration pipeline."""
    label: str
    raw_confidence: float          # Original softmax max (0-100)
    calibrated_confidence: float   # Post-calibration confidence (0-100)
    disease_presence_confidence: float  # 0-100, how likely the leaf is diseased at all
    disease_gate_passed: bool      # True if disease evidence is strong enough to classify
    entropy: float                 # Prediction entropy (0 = certain, higher = uncertain)
    margin: float                  # Gap between top-1 and top-2 probabilities
    disease_visibility_strength: float = 0.0  # Composite visual disease strength (0-1)
    calibration_factors: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Calibration Engine
# ---------------------------------------------------------------------------

CLASS_LABELS = [
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_healthy"
]

HEALTHY_LABEL = "Tomato_healthy"
# We will dynamically find HEALTHY_INDEX or default to the last index if not matched.
# HEALTHY_INDEX = CLASS_LABELS.index(HEALTHY_LABEL)


class ConfidenceCalibrator:
    """
    Multi-stage post-hoc confidence calibration for CNN disease predictions.
    
    Designed to convert overconfident softmax outputs into trustworthy
    probability estimates that align with actual predictive accuracy.
    
    Now includes context-aware quality dampening: when disease symptoms
    are visually obvious (high realism + severity), image quality penalties
    are reduced because the disease evidence transcends blur/noise.
    """

    # Temperature scaling parameter — higher = softer distribution
    # 1.0 = no change, 1.5 = moderate softening, 2.0+ = heavy softening
    # Now dynamic: reduces as visual disease strength increases
    BASE_TEMPERATURE = 1.5

    # Entropy ceiling for normalization (log2(num_classes) for 3 classes ≈ 1.585)
    MAX_ENTROPY = math.log2(len(CLASS_LABELS))

    # Margin thresholds
    NARROW_MARGIN_THRESHOLD = 0.25   # Below this → model is confused between classes
    STRONG_MARGIN_THRESHOLD = 0.50   # Above this → model is fairly decisive

    # Healthy gate threshold — if healthy prob > this, disease evidence is weak
    HEALTHY_GATE_THRESHOLD = 0.30

    # Disease presence confidence floor — below this, cap disease confidence
    DISEASE_PRESENCE_FLOOR = 0.55

    # Quality dampening range
    QUALITY_FLOOR = 0.40  # Below this quality, apply maximum dampening

    def calibrate(
        self,
        predicted_label: str,
        raw_confidence: float,
        full_probabilities: List[float],
        image_quality: float = 1.0,
        severity_report=None
    ) -> CalibratedPrediction:
        """
        Run multi-stage calibration pipeline.

        Args:
            predicted_label: Top-1 class label from CNN
            raw_confidence: Raw softmax max as percentage (0-100)
            full_probabilities: Full softmax vector [p_early, p_late, p_healthy]
            image_quality: Aggregate image quality score (0-1)
            severity_report: Optional SeverityReport for context-aware calibration

        Returns:
            CalibratedPrediction with all calibration metadata
        """
        probs = list(full_probabilities)  # Make a copy to avoid mutation
        factors = {}

        # ── Stage 0: Compute Disease Visibility Strength (Part 4) ──
        dvs = self._compute_disease_visibility_strength(severity_report)
        factors["disease_visibility_strength"] = round(dvs, 3)

        # ── Stage 1: Dynamic Temperature Scaling ──
        # If disease is visually undeniable, we can trust the peaked distribution more
        current_temp = self.BASE_TEMPERATURE
        if dvs > 0.60:
            # Scale down temperature toward 1.0 (no scaling) as dvs goes from 0.6 to 1.0
            reduction_ratio = min((dvs - 0.60) / 0.40, 1.0)
            current_temp = self.BASE_TEMPERATURE - (self.BASE_TEMPERATURE - 1.0) * reduction_ratio
            
        temp_scaled_probs = self._temperature_scale(probs, current_temp)
        top_idx = probs.index(max(probs))
        temp_scaled_conf = temp_scaled_probs[top_idx] * 100.0
        factors["temperature_scaling"] = round(temp_scaled_conf / max(raw_confidence, 0.01), 3)
        factors["applied_temperature"] = round(current_temp, 2)

        # ── Stage 2: Entropy Analysis ──
        entropy = self._compute_entropy(temp_scaled_probs)
        normalized_entropy = entropy / self.MAX_ENTROPY  # 0 = certain, 1 = max uncertainty
        # Entropy penalty: high entropy → reduce confidence
        entropy_factor = 1.0 - (normalized_entropy * 0.4)  # At max entropy, lose 40%
        
        # LIFT entropy penalty if it's high but disease visibility is strong
        if dvs > 0.50 and entropy_factor < 0.90:
            entropy_factor = min(entropy_factor + (dvs - 0.50) * 0.4, 0.95)
            
        entropy_adjusted_conf = temp_scaled_conf * entropy_factor
        factors["entropy_factor"] = round(entropy_factor, 3)

        # ── Stage 3: Disease-Aware Margin Reduction ──
        sorted_probs = sorted(temp_scaled_probs, reverse=True)
        margin = sorted_probs[0] - (sorted_probs[1] if len(sorted_probs) > 1 else 0)
        
        # Check if the model is just unsure WHICH disease it is
        # Dynamically determine the healthy index if possible, otherwise assume it's the last class
        healthy_idx = -1
        if len(temp_scaled_probs) == len(CLASS_LABELS):
            try:
                healthy_idx = CLASS_LABELS.index(HEALTHY_LABEL)
            except ValueError:
                pass
        
        if healthy_idx == -1:
            # Fallback: assume the highest probability class that has 'healthy' in the name
            # Or default to the last element
            healthy_idx = len(temp_scaled_probs) - 1
            for i, label in enumerate(CLASS_LABELS[:len(temp_scaled_probs)]):
                if "healthy" in label.lower():
                    healthy_idx = i
                    break

        p_idx0 = temp_scaled_probs.index(sorted_probs[0])
        p_idx1 = temp_scaled_probs.index(sorted_probs[1]) if len(temp_scaled_probs) > 1 else -1
        is_disease_confusion = (p_idx0 != healthy_idx and p_idx1 != healthy_idx)
        
        try:
            healthy_prob = temp_scaled_probs[healthy_idx]
        except IndexError:
            healthy_prob = 0.0 if "healthy" not in predicted_label.lower() else temp_scaled_probs[p_idx0]
            
        disease_presence_conf = (1.0 - healthy_prob) * 100.0

        if margin < self.NARROW_MARGIN_THRESHOLD:
            # Model is confused — penalize proportionally
            margin_factor = 0.6 + (margin / self.NARROW_MARGIN_THRESHOLD) * 0.4
            
            # LIFT penalty if we are sure it's A disease (just not which one)
            if is_disease_confusion and disease_presence_conf > 85:
                # Soften the penalty because disease presence is undeniable
                margin_factor = 0.85 + (margin_factor - 0.6) * 0.3
        elif margin < self.STRONG_MARGIN_THRESHOLD:
            margin_factor = 0.85 + (margin - self.NARROW_MARGIN_THRESHOLD) / (self.STRONG_MARGIN_THRESHOLD - self.NARROW_MARGIN_THRESHOLD) * 0.15
        else:
            margin_factor = 1.0

        margin_adjusted_conf = entropy_adjusted_conf * margin_factor
        factors["margin_factor"] = round(margin_factor, 3)

        # ── Stage 4: Context-Aware Quality Dampening ──
        quality_factor = self._compute_quality_factor(image_quality, dvs)
        calibrated_conf = margin_adjusted_conf * quality_factor
        factors["quality_factor"] = round(quality_factor, 3)

        # ── Stage 5: Healthy/Diseased Gate ──
        disease_gate_passed = True
        is_healthy = "healthy" in predicted_label.lower()
        
        if not is_healthy and disease_presence_conf < self.DISEASE_PRESENCE_FLOOR * 100:
            # Weak disease evidence — cap confidence
            gate_penalty = disease_presence_conf / (self.DISEASE_PRESENCE_FLOOR * 100)
            calibrated_conf *= gate_penalty
            disease_gate_passed = False
            factors["disease_gate_penalty"] = round(gate_penalty, 3)

        # ── Stage 6: Enhanced Visual Evidence Recovery ──
        # If disease is visually undeniable, recover more aggressively from calibration loss
        if severity_report and dvs > 0.50 and disease_presence_conf > 80:
            # The disease is REAL — recover lost confidence
            # OBVIOUS disease (dvs > 0.75) should never be below 75% of raw if it's undeniably diseased
            recovery_floor = raw_confidence * (0.65 + (dvs - 0.5) * 0.4)
            recovery_floor = min(recovery_floor, 90.0) 
            
            if calibrated_conf < recovery_floor:
                # Scale recovery by dvs — higher visibility = faster recovery
                recovery_strength = (dvs - 0.50) / 0.50
                recovery = (recovery_floor - calibrated_conf) * (0.4 + 0.6 * recovery_strength)
                calibrated_conf += recovery
                factors["visual_evidence_recovery"] = round(recovery, 2)

        # Final clamp
        calibrated_conf = max(min(calibrated_conf, 99.0), 1.0)

        return CalibratedPrediction(
            label=predicted_label,
            raw_confidence=round(raw_confidence, 2),
            calibrated_confidence=round(calibrated_conf, 2),
            disease_presence_confidence=round(disease_presence_conf, 2),
            disease_gate_passed=disease_gate_passed,
            entropy=round(entropy, 4),
            margin=round(margin, 4),
            disease_visibility_strength=round(dvs, 3),
            calibration_factors=factors
        )

    def _compute_disease_visibility_strength(self, sr) -> float:
        """
        Quantifies how visually undeniable the disease symptoms are.
        Derived from:
          - Lesion area %
          - Discoloration intensity
          - Symptom density
          - Texture abnormality
          - Edge damage
        
        This has HIGH priority in final confidence - obvious symptoms
        should dominate minor image imperfections.
        """
        if not sr:
            return 0.0
            
        # Support both dataclass and dict
        def get_val(key, default=0.0):
            if hasattr(sr, key):
                return getattr(sr, key)
            # Map dict keys if necessary
            mapping = {
                "lesion_area_pct": "lesion_area_pct",
                "discoloration_score": "discoloration",
                "symptom_density": "symptom_density",
                "texture_abnormality": "texture_abnormality",
                "edge_damage_score": "edge_damage"
            }
            return sr.get(mapping.get(key, key), default)

        # Normalize lesion area: 20%+ gets full score
        lesion_strength = min(get_val("lesion_area_pct") / 20.0, 1.0)
        
        # Combine signals with strong weighting on direct symptom indicators
        dvs = (
            lesion_strength * 0.30 +
            get_val("discoloration_score") * 0.25 +
            get_val("symptom_density") * 0.20 +
            get_val("texture_abnormality", 0.0) * 0.15 + 
            get_val("edge_damage_score") * 0.10
        )
        
        return float(min(dvs, 1.0))

    def _compute_quality_factor(self, image_quality: float, dvs: float) -> float:
        """
        Context-aware image quality penalty. (Part 1)
        
        Old behavior: blur always penalizes confidence equally.
        New behavior: if disease visibility is high, the penalty shrinks.
        """
        # Base quality factor
        if image_quality < self.QUALITY_FLOOR:
            base_factor = 0.5 + 0.5 * (image_quality / self.QUALITY_FLOOR)
        else:
            base_factor = 0.75 + 0.25 * image_quality
        
        # Context-aware override: reduce penalty when disease is visually obvious
        if dvs > 0.50:
            override_strength = min((dvs - 0.50) / 0.50, 1.0)
            penalty_reduction = override_strength * 0.7  # Increased from 0.6
            base_factor = base_factor + (1.0 - base_factor) * penalty_reduction
        elif dvs > 0.30:
            override_strength = (dvs - 0.30) / 0.20
            penalty_reduction = override_strength * 0.35 # Increased from 0.3
            base_factor = base_factor + (1.0 - base_factor) * penalty_reduction
        
        return float(min(base_factor, 1.0))

    def _temperature_scale(self, probs: List[float], temp: float = 1.5) -> List[float]:
        """
        Apply temperature scaling to soften peaked softmax distributions.
        """
        # Convert to numpy array
        probs_arr = np.array(probs, dtype=np.float64)
        eps = 1e-10
        
        # Apply eps lower bound safely
        p_safe = np.maximum(probs_arr, eps)
        
        # Optimize math: exp(log(p_safe) / temp) is mathematically equivalent to p_safe ** (1.0 / temp)
        scaled_exp = p_safe ** (1.0 / temp)

        # Normalize and convert back to list
        return (scaled_exp / np.sum(scaled_exp)).tolist()

    def _compute_entropy(self, probs: List[float]) -> float:
        """Shannon entropy of the probability distribution."""
        eps = 1e-10
        return -sum(p * math.log2(max(p, eps)) for p in probs)
