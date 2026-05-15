"""
HarvestShield — Consistency Engine
===================================
A post-fusion validation layer that enforces semantic coherence
across visual, environmental, and sensor evidence.

The engine resolves contradictions such as:
  - "Severe disease detected" + "Low Reliability" (Inconsistent)
  - "High Confidence" + "Environmental Contradiction" (Inconsistent)
  - "Poor Image Quality" + "Environment explains the quality" (Explainable)

Design:
  - Non-destructive: Log adjustments, don't just overwrite.
  - Evidence-based: Visual symptoms override minor quality penalties.
  - Safety-first: High severity + high realism = High Reliability floor.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class ConsistencyLog:
    """Detailed log of consistency checks and adjustments."""
    check_name: str
    passed: bool
    adjustment: str
    impact: float = 0.0


@dataclass
class ConsistencyReport:
    """Final output of the consistency validation process."""
    overall_status: str = "Consistent"
    logs: List[ConsistencyLog] = field(default_factory=list)
    reliability_override: float = -1.0
    confidence_override: float = -1.0
    contradiction_found: bool = False
    warning_flag: str = ""


class ConsistencyEngine:
    """
    Validates fusion output against physical and biological reality.
    
    This runs AFTER fusion but BEFORE final explanation generation.
    It ensures that the probabilistic math hasn't produced a result
     that makes no sense to a human observer.
    """

    def validate(self, 
                 fusion_output: Dict[str, Any], 
                 scene_context: Any, 
                 morphology_report: Any,
                 severity_report: Dict[str, Any]) -> ConsistencyReport:
        
        report = ConsistencyReport()
        
        # Extract core metrics
        raw_conf = fusion_output.get("calibrated_confidence", 0.0)
        raw_rel = fusion_output.get("reliability_score", 0.5)
        severity = severity_report.get("severity_score", 0.0)
        realism = severity_report.get("realism_score", 0.0)
        
        # ── Rule 1: Visual Evidence Override ──
        # If disease is visually undeniable (high severity + high realism),
        # reliability cannot be "Low".
        if severity > 0.6 and realism > 0.7:
            if raw_rel < 0.6:
                report.reliability_override = 0.85
                report.logs.append(ConsistencyLog(
                    "VisibleEvidenceOverride", False, 
                    "Symptom clarity overrides image quality penalties", 0.25
                ))

        # ── Rule 2: Environmental Explanation for Quality ──
        # If image is blurry but scene analysis confirms rain/fog, 
        # the 'uncertainty' should be attributed to nature, not system failure.
        if scene_context.blur_penalty_reduction > 0 and raw_rel < 0.7:
            report.reliability_override = max(report.reliability_override, raw_rel + scene_context.blur_penalty_reduction)
            report.logs.append(ConsistencyLog(
                "EnvQualityExplain", True, 
                f"Quality penalty reduced due to {scene_context.blur_type}", 0.15
            ))

        # ── Rule 3: Morphology Validation ──
        # If morphology confirms CNN class, boost confidence slightly.
        if morphology_report.morphology_score > 0.7 and morphology_report.signature_confidence > 0.6:
            report.confidence_override = min(raw_conf + 0.1, 0.99)
            report.logs.append(ConsistencyLog(
                "MorphologyConfirmation", True, 
                f"Visual signatures confirm {morphology_report.method_name}", 0.1
            ))

        # ── Rule 4: Severity-Certainty Coherence ──
        # Severe symptoms with weak confidence is a contradiction.
        if severity > 0.7 and raw_conf < 0.5:
            report.contradiction_found = True
            report.warning_flag = "Low confidence despite severe visual symptoms"
            report.logs.append(ConsistencyLog(
                "SeverityDissonance", False, 
                "Extreme symptoms detected but confidence is weak", 0.0
            ))

        # ── Rule 5: Sensor-Environment Consistency ──
        # High humidity sensors + Scene rain detection = Extremely consistent.
        # (Assuming sensors are passed in fusion_output)
        sensor_hum = fusion_output.get("sensor_data", {}).get("humidity", 50)
        if scene_context.rain_detected and sensor_hum > 80:
            report.logs.append(ConsistencyLog(
                "SensorSceneLock", True, 
                "Visual and sensor environment are perfectly aligned", 0.0
            ))

        # Determine overall status
        failed_checks = [l for l in report.logs if not l.passed]
        if report.contradiction_found:
            report.overall_status = "Contradictory"
        elif failed_checks:
            report.overall_status = "Adjusted"
        
        return report
