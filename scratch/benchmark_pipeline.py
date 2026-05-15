import time
import sys
import os
import json
import cv2
import numpy as np
from PIL import Image

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from model_utils import load_model
from reasoning_orchestrator import HierarchicalOrchestrator

def benchmark():
    print("Loading model...")
    model = load_model()
    
    orchestrator = HierarchicalOrchestrator(model)
    
    # Create a dummy image
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.imwrite("bench_test.jpg", img)
    
    sensor_data = {"temperature": 28, "humidity": 85, "soil": 40}
    location = {"latitude": 17.385, "longitude": 78.486}
    
    print("\nStarting Benchmark...")
    start = time.time()
    
    try:
        # We'll mock the internal steps to see where time is spent
        from image_context import ImageContext
        
        step_start = time.time()
        ctx = ImageContext.from_path("bench_test.jpg")
        print(f"Step 1: ImageContext took {time.time() - step_start:.4f}s")
        
        step_start = time.time()
        scene = orchestrator.scene_analyzer.analyze(ctx)
        print(f"Step 2: Scene Analysis took {time.time() - step_start:.4f}s")
        
        step_start = time.time()
        from model_utils import predict_image
        label, conf, all_probs = predict_image(model, "bench_test.jpg")
        print(f"Step 3: CNN Inference took {time.time() - step_start:.4f}s")
        
        step_start = time.time()
        severity_report = orchestrator.severity_analyzer.analyze_severity(ctx)
        print(f"Step 4: Visual Severity took {time.time() - step_start:.4f}s")
        
        step_start = time.time()
        morphology = orchestrator.morphology_router.analyze_morphology(ctx, label)
        print(f"Step 5: Morphology Router took {time.time() - step_start:.4f}s")
        
        step_start = time.time()
        from fusion_engine import SensorReading
        sensors = SensorReading(**sensor_data)
        fusion_result = orchestrator.fusion_engine.fuse(
            label, conf, sensors, location, 
            severity_report, morphology, scene
        )
        print(f"Step 6: Fusion Engine took {time.time() - step_start:.4f}s")
        
        step_start = time.time()
        consistency = orchestrator.consistency_engine.check(fusion_result, location)
        print(f"Step 7: Consistency Engine took {time.time() - step_start:.4f}s")
        
        step_start = time.time()
        explanation = orchestrator.explanation_engine.generate_explanation(
            fusion_result, consistency
        )
        print(f"Step 8: Explanation Engine took {time.time() - step_start:.4f}s")
        
    except Exception as e:
        print(f"Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
        
    total = time.time() - start
    print(f"\nTotal Pipeline Time: {total:.4f}s")
    
    if os.path.exists("bench_test.jpg"):
        os.remove("bench_test.jpg")

if __name__ == "__main__":
    benchmark()
