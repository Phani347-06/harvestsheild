import sys
import os
import cv2
import numpy as np

# Add backend to path to import modules
sys.path.append(os.path.join(os.getcwd(), 'backend'))

try:
    from reasoning_orchestrator import HierarchicalOrchestrator
    from model_utils import load_model
    
    print("Loading model...")
    model = load_model()
    orchestrator = HierarchicalOrchestrator(model)
    
    # Create a dummy image
    dummy_img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.putText(dummy_img, "Test", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite("dummy_test.jpg", dummy_img)
    
    print("Running orchestrator...")
    result = orchestrator.process_request(
        image_path="dummy_test.jpg",
        sensor_data={"temperature": 25, "humidity": 60, "soil": 50},
        location={"latitude": 17.3850, "longitude": 78.4867}
    )
    
    import json
    print(json.dumps(result, indent=2))
    
    # Cleanup
    if os.path.exists("dummy_test.jpg"):
        os.remove("dummy_test.jpg")
    
except Exception as e:
    print(f"Test Failed: {e}")
    import traceback
    traceback.print_exc()
