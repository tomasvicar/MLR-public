"""Export YOLO11n (Ultralytics, AGPL-3.0) to ONNX at 320 px for the detector page.

Writes model/yolo11n_320.onnx and test/bus.jpg next to this file. Ultralytics is
NOT a project dependency; run it in a throw-away environment from a scratch
directory (the weights yolo11n.pt are downloaded into the current directory):

    cd <scratch dir>
    uv run --project /path/to/MLR --with ultralytics==8.4.216 --with onnx --with onnxslim --with onnxruntime \
        python /path/to/MLR/course_pages/apps/detector/make_model.py

Exported with ultralytics 8.4.216, onnx 1.23.2, torch 2.15.0 (22 November 2026).
The output tensor is 1 x 84 x 2100: per candidate cx, cy, w, h (pixels of the
320 x 320 input) and 80 COCO class scores (already sigmoid); no NMS inside.
"""
import shutil
from pathlib import Path

import onnx
import ultralytics
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
SIZE = 320

out = Path(YOLO("yolo11n.pt").export(format="onnx", imgsz=SIZE, opset=17, simplify=True,
                                     dynamic=False, nms=False))
(HERE / "model").mkdir(exist_ok=True)
target = HERE / "model" / f"yolo11n_{SIZE}.onnx"
shutil.copy2(out, target)

m = onnx.load(target)
print("input", [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim],
      "output", [d.dim_value for d in m.graph.output[0].type.tensor_type.shape.dim])
(HERE / "test").mkdir(exist_ok=True)
shutil.copy2(Path(ultralytics.__file__).parent / "assets" / "bus.jpg", HERE / "test" / "bus.jpg")
print("written", target, HERE / "test" / "bus.jpg")
