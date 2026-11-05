Group Project Task -  Edge AI resources offered by   laitatheiYOLOv8-ONNX-RKNN-HORIZON-TensorRT-Detection 

FAQs:

Q. What should we do for this Group Project Task?

A. Your group needs to study the Edge AI resources offered by YOLOv8 model + rknn toolkit2 + python + rk3588 + orangpi 5 plus + lots of others.

Q. Wow! We are not sure we can achieve all those in a very short given time. What should we do?

A. Try to set up a playground (you need to set up rknn toolkit2) and play (if you have a playground, you can learn things naturally). Maybe we can give it a try by doing the following steps:

Step 1: Visit   https://github.com/laitathei/YOLOv8-ONNX-RKNN-HORIZON-TensorRT-Detection

qhttps://github.com/airockchip/rknn-toolkit2

https://github.com/airockchip/rknn_model_zoo

Step 2: Set up your Linux (Ubuntu) OS (under Windows OS is possible, but you are on your own)

Step 3: Duplicate and run this open-source project (by laitathei, the first link provided above) on your own computer (laptop/desktop)

(Please review the lecture; you can install rknn toolkit2 and do the model conversion from onnx to rknn; your mp4 as an evidence should show that you have done the conversion; you can also use the onnx model to do object detection on your laptop (x86 computer) - please see the lecture, try it anyway; if failed, no worries.)

Step 4: Run the inference task on RK3588 using the rknn model (with the instructor’s support in class – no need to do it yourself)

 

Step 5: Each group needs to submit a presentation that includes three files (1. Report in PDF, 2. PPT file, 3. Mp4 file)

Q. What should we write in PDF?

A. PDF - must have 2000 words; some general review of the yolov8, rknn tollkit2, rk3588, My Closing Thoughts (Actually, My Group's Closing Thoughts) and at least two references. 

Q. What should we write in PPT?

A. A condensed version of you PDF.

Q. What should we present in mp4?

A. About 20 minutes of video that goes through you PPT and some screen video demo (just like the class lecture).

Q. What should we include in our screen video demo?

A. You need to at least show that you can do model conversion (from onnx to rknn). This proves that your group went through the rknn toolchain installation and usage. I encourage you to do object detection of a given photo on your laptop directly in the toolchain environment (my lecture mp4 has demonstrated it; you may need to download rnkk model zoo). Please try it (if it failed, that's fine - no deduction of grades). 


GUIDE: 
xx Review the GitHub repositories
Understand YOLOv8, RKNN Toolkit2, RK3588, and the example project.


[ x] Set up a Linux environment

Use Ubuntu, either installed directly or in a compatible virtual environment.


[ x] Download the open-source project

Clone the YOLOv8 detection repository and inspect its instructions and example files.


[ x] Install RKNN Toolkit2

Set up Python, dependencies, and the RKNN model-conversion tools.


[ x] Convert an ONNX model to RKNN

Run the conversion and record your screen showing the commands, results, and generated .rknn file.


[ x] Try object detection (optional)

Use an ONNX model to detect objects in an image on your laptop. Your professor says failure is acceptable.


[ ] Write the 2,000-word PDF report

Explain the technologies, setup, experiments, results, group reflections, and references.


[ ] Create the PowerPoint

Summarize the important information from your report.


[ ] Record a roughly 20-minute MP4

Present the slides and include a screen demonstration of the ONNX-to-RKNN conversion.


SETUP:

python3 -m venv .venv

python3 -m pip install -r requirements.txt

setup_yolov8_rknn_ubuntu.sh

cd ~/cpsc_476_project_yolo_v2
source .venv/bin/activate

python - <<'PY'
import sys

print("=== YOLOv8 / RKNN Environment Test ===")
print("Python:", sys.version.split()[0])

packages = [
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("onnx", "onnx"),
    ("onnxruntime", "onnxruntime"),
    ("ultralytics", "ultralytics"),
    ("RKNN Toolkit2", "rknn.api"),
]

for name, module in packages:
    try:
        imported = __import__(module, fromlist=["RKNN"])
        version = getattr(imported, "__version__", "Installed")
        print(f"[PASS] {name}: {version}")
    except Exception as e:
        print(f"[FAIL] {name}: {e}")

print("=== Test Complete ===")
PY

EXPECTED: 
=== YOLOv8 / RKNN Environment Test ===
Python: 3.8.10
[PASS] numpy: 1.23.5
[PASS] torch: 1.10.1+cu102
[PASS] onnx: 1.10.0
[PASS] onnxruntime: 1.10.0
[PASS] ultralytics: 8.0.147
[PASS] RKNN Toolkit2: Installed
=== Test Complete ===