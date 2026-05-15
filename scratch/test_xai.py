"""
HarvestShield — Calibration Refinement Test Suite
Tests the balance between skepticism (weak evidence) and confidence (strong evidence).
"""
import sys
sys.path.insert(0, 'backend')

from fusion_engine import advanced_fusion_engine

def run_test(name, **kwargs):
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"{'='*70}")
    result = advanced_fusion_engine(**kwargs)
    
    print(f"  Prediction:         {result['prediction']}")
    print(f"  Final Confidence:   {result['overall_confidence']}%")
    print(f"  Banner:             {result['banner_color']}")
    print(f"  Reliability:        {result['reliability']}")
    print(f"  Visual Severity:    {result['visual_severity']} (score: {result['severity_score']})")
    print(f"  Realism Score:      {result['realism_score']}")
    print(f"  Disease Presence:   {result.get('disease_presence_confidence', 'N/A')}")
    print(f"  Disease Gate:       {'PASSED' if result.get('disease_gate_passed', True) else 'FAILED'}")
    print(f"  Calibrated CNN:     {result.get('calibrated_confidence', 'N/A')}% (raw: {result['raw_metrics']['cnn_confidence']}%)")
    print(f"  DVS:                {result.get('disease_visibility_strength', 'N/A')}")
    print(f"  Evidence Override:  {result.get('strong_evidence_override', False)}")
    print(f"  Contradiction:      {result['contradiction_level']}")
    print(f"  Visual Match:       {result.get('visual_detection_confidence', 'N/A')}")
    print(f"  Farmer Summary:     {result['farmer_summary']}")
    print(f"  Recommendation:     {result['recommendation']}")
    
    if result.get('warnings'):
        print(f"  Warnings:")
        for w in result['warnings']:
            print(f"    [!] {w}")
    
    print(f"  Tech Chain:")
    for line in result['technical_chain']:
        print(f"    -> {line}")
    
    return result

if __name__ == "__main__":
    IMAGE = "backend/test_image.jpg"
    
    # ================================================================
    # GROUP A: OBVIOUS DISEASE (should be CONFIDENT)
    # ================================================================
    
    # Test 1: High CNN + High Env + Strong Disease (ideal case)
    run_test("OBVIOUS DISEASE + STRONG ENV (expect HIGH confidence)",
        image_path=IMAGE,
        predicted_class="Tomato_Late_blight",
        cnn_confidence=92.0,
        temperature=19.0,
        humidity=90.0,
        soil=65.0,
        climate_context={"climate_zone": "tropical", "is_online": True, "weather": {"rain": 10.0}},
        full_probabilities=[0.04, 0.92, 0.04]
    )

    # Test 2: High CNN + Moderate Env (should still be reasonably confident)
    run_test("OBVIOUS DISEASE + MODERATE ENV (expect MODERATE confidence)",
        image_path=IMAGE,
        predicted_class="Tomato_Early_blight",
        cnn_confidence=88.0,
        temperature=26.0,
        humidity=65.0,
        soil=45.0,
        climate_context={"climate_zone": "tropical", "is_online": False},
        full_probabilities=[0.88, 0.07, 0.05]
    )

    # ================================================================
    # GROUP B: WEAK/AMBIGUOUS EVIDENCE (should be SKEPTICAL)
    # ================================================================

    # Test 3: High CNN + Weak Env (contradiction - be skeptical)
    run_test("HIGH CNN + WEAK ENV (expect SKEPTICISM)",
        image_path=IMAGE,
        predicted_class="Tomato_Late_blight",
        cnn_confidence=90.0,
        temperature=22.0,
        humidity=15.0,
        soil=10.0,
        climate_context={"climate_zone": "temperate", "is_online": False},
        full_probabilities=[0.05, 0.90, 0.05]
    )

    # Test 4: High CNN but high healthy probability (disease gate test)
    run_test("HIGH HEALTHY PROB (expect disease gate FAIL)",
        image_path=IMAGE,
        predicted_class="Tomato_Late_blight",
        cnn_confidence=55.0,
        temperature=25.0,
        humidity=50.0,
        soil=40.0,
        climate_context={"climate_zone": "temperate", "is_online": False},
        full_probabilities=[0.10, 0.55, 0.35]
    )

    # ================================================================
    # GROUP C: LOW CONFIDENCE / HEALTHY
    # ================================================================

    # Test 5: Low CNN confidence
    run_test("LOW CNN (expect UNCERTAINTY)",
        image_path=IMAGE,
        predicted_class="Tomato_Early_blight",
        cnn_confidence=45.0,
        temperature=26.0,
        humidity=70.0,
        soil=50.0,
        climate_context={"climate_zone": "tropical", "is_online": False},
        full_probabilities=[0.45, 0.30, 0.25]
    )

    # Test 6: Healthy prediction
    run_test("HEALTHY PREDICTION (expect calm, no urgency)",
        image_path=IMAGE,
        predicted_class="Tomato_healthy",
        cnn_confidence=88.0,
        temperature=26.0,
        humidity=55.0,
        soil=45.0,
        climate_context={"climate_zone": "tropical", "is_online": False},
        full_probabilities=[0.06, 0.06, 0.88]
    )
    
    print(f"\n{'='*70}")
    print("ALL TESTS COMPLETE")
    print(f"{'='*70}")
