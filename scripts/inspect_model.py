"""
Inspect the actual input/output tensor shapes of both ONNX models.
Run this before anything else when debugging model issues.
"""
import numpy as np
import onnxruntime as ort
from pathlib import Path

def inspect_model(model_path: str):
    print(f"\n{'='*60}")
    print(f"Model: {model_path}")
    print('='*60)

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    print("\nINPUTS:")
    for inp in session.get_inputs():
        print(f"  name={inp.name!r}  shape={inp.shape}  dtype={inp.type}")

    print("\nOUTPUTS:")
    for out in session.get_outputs():
        print(f"  name={out.name!r}  shape={out.shape}  dtype={out.type}")

    # Run a dummy forward pass to see actual output shapes
    inp = session.get_inputs()[0]
    # Replace dynamic dims with concrete values
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    dummy = np.random.randn(*shape).astype(np.float32)

    outputs = session.run(None, {inp.name: dummy})
    print("\nACTUAL OUTPUT SHAPES (dummy forward pass):")
    for i, out in enumerate(outputs):
        print(f"  [{i}] shape={out.shape}  dtype={out.dtype}  min={out.min():.3f}  max={out.max():.3f}")

if __name__ == "__main__":
    inspect_model("models/retinaface_10g.onnx")
    inspect_model("models/arcface_w600k_r50.onnx")