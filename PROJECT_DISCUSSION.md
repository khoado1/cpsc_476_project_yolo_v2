# YOLOv8 Object Detection with ONNX and RKNN: Project Discussion

## Project Overview

Our group explored the practical process of preparing an object detection model for edge AI deployment. The project centered on YOLOv8, the Open Neural Network Exchange format, and Rockchip’s RKNN Toolkit2. Our main objectives were to establish a usable software environment, work with an ONNX model, convert that model into RKNN format, and generate images showing object detection results. This gave us experience with the steps between obtaining a pretrained neural network and preparing it for a specialized inference platform.

The project required more than installing a single Python package. We had to set up both the ONNX software components and the RKNN conversion tools, understand their different responsibilities, and follow a conversion sequence. The two conversion scripts were `onnx2rknn_step1.py` and `onnx2rknn_step2.py`. The second script also included inference and image visualization, connecting the model conversion process to a visible output.

## Technologies and Their Roles

YOLOv8 supplied the object detection model used in this project. Object detection combines recognition with localization: the system predicts an object category and a bounding box describing where that object appears. This differs from classifying an entire image, because a single photograph can contain several objects from different categories. Ultralytics describes YOLOv8 as using an anchor-free detection head and providing model variants for different computational requirements. Our project selected the pretrained nano model, `yolov8n`, and used the scripts’ 80 object categories. Our work focused on deployment rather than training a new detector. [1](https://docs.ultralytics.com/models/yolov8/)

ONNX served as the intermediate model representation. It describes a computational graph with operators, inputs, outputs, and model parameters, allowing compatible tools to exchange a trained model. This role was essential because our workflow moved from a PyTorch-based YOLO model toward Rockchip’s deployment format. ONNX provided the representation that the RKNN conversion script loaded. The official ONNX introduction explains its role as a common representation for machine learning models. [2](https://onnx.ai/onnx/intro/)

ONNX Runtime has a separate purpose: it executes models. Installing ONNX supports working with the model format, while installing ONNX Runtime enables an inference session that processes input tensors. Our repository includes `onnxruntime_inference.py`, providing a host-side inference path separate from RKNN conversion. This distinction helped us understand why the project needed both packages. [3](https://onnxruntime.ai/docs/)

RKNN Toolkit2 supplied the conversion interface for the Rockchip target. Rockchip describes Toolkit2 as supporting model conversion, inference, and performance evaluation, while Toolkit-Lite2 provides Python interfaces for deployment on supported Rockchip platforms. Our conversion code imports `RKNN` from `rknn.api`; the separate deployment script imports `RKNNLite` from `rknnlite.api`. These imports represent different stages of the software stack. [4](https://github.com/airockchip/rknn-toolkit2)

The conversion target in our scripts is `rk3588`. The assignment connects this target with the RK3588 and Orange Pi 5 Plus deployment context. The relevant principle is that a model must be prepared for the hardware and runtime that will execute it. Rockchip lists the RK3588 series among supported platforms. However, selecting that target in a conversion script does not establish that our saved image was produced on a physical board. [4](https://github.com/airockchip/rknn-toolkit2)

## Environment Setup

Establishing the environment was a major part of the project. The local setup script targets Ubuntu on an x86-64 computer and recommends Ubuntu 20.04 with Python 3.8 for the legacy RKNN Toolkit2 1.5.0 workflow. It checks the operating system, processor architecture, and Python version before proceeding. These checks express the compatibility assumptions of this project’s setup rather than universal requirements for every RKNN release.

We used the project’s virtual-environment approach to organize the Python tools. The documented sequence creates `.venv`, activates it, and installs dependencies within that environment. This makes the working interpreter and its packages easier to identify. It also reduces interference from unrelated Python software, which matters when reproducing a project that depends on specific package versions.

The project pins several dependencies: Ultralytics 8.0.147, NumPy 1.23.5, ONNX 1.10.0, and ONNX Runtime 1.10.0. Its requirements file also specifies OpenCV 4.8.1.78 and ranges for Pillow and PyYAML. These are the documented project versions; an expected-output example in the instructions should not be treated as a captured record of every installed version. A reproducible submission should preserve the actual environment check from the machine used for the demonstration.

RKNN Toolkit2 required an additional installation step. The setup script looks for a Toolkit2 1.5.0 wheel compatible with Python 3.8 and Linux x86-64. If that wheel is absent, it tells the user to obtain the compatible package and install it separately. Therefore, reaching the end of the setup script does not by itself prove that RKNN was installed. Successfully importing `RKNN` from `rknn.api` is an essential verification before conversion.

The instructions include an environment test that imports NumPy, PyTorch, ONNX, ONNX Runtime, Ultralytics, and RKNN. This is useful because successful package installation and successful package import are different milestones. Import failures can reveal missing dependencies or an incompatible interpreter before a long conversion begins. Our setup experience emphasized that environment verification is part of the engineering work, not merely preparation outside the project.

## Model Preparation and Conversion Experiments

The repository’s export script, `pytorch2onnx.py`, loads `yolov8n.pt` through the Ultralytics interface and requests an ONNX export using operator set 12. It specifies an input height of 480 pixels and width of 640 pixels, then saves the model as `model/yolov8n-480-640.onnx`. The repository supports a batch size of one. These settings establish the model interface used by the remaining scripts.

The first conversion experiment used `onnx2rknn_step1.py`. This script creates a list of JPEG images from `dataset/` and writes their paths into `dataset.txt`. The saved manifest contains 51 entries. In this workflow, these images support quantization calibration. Calibration allows the conversion process to characterize model behavior using sample inputs; it does not retrain the detector or demonstrate accuracy on a labeled test set.

The script configures the RKNN object with zero mean values, standard deviation values of 255 for each channel, and the `rk3588` target. It then loads the ONNX model and specifies `output0` as the output node. These settings connect the exported model to the conversion tool’s expected input and output behavior. If either the model path or output node were wrong, the remaining conversion steps could not reliably proceed.

Next, the script calls `hybrid_quantization_step1` using the image manifest. It produces intermediate `.data`, `.model`, and `.quantization.cfg` files and moves them into `config/`. All three corresponding artifacts are present in our project. The saved quantization configuration includes custom layer entries marked `float16`, reflecting a hybrid precision workflow. They are the handoff between the two conversion stages, showing why running the first script twice is not equivalent to completing both stages.

The second conversion experiment used `onnx2rknn_step2.py`. It reads the intermediate model, calibration data, and quantization configuration, then calls `hybrid_quantization_step2`. After that stage succeeds, it exports `yolov8n-480-640.rknn` and moves the result into `model/`. The project therefore follows an explicit sequence: prepare the ONNX representation, generate hybrid-quantization intermediates, build the RKNN model, and export the deployment artifact.

Our model files provide a concrete conversion result. The ONNX artifact contains 13,773,386 bytes, while the RKNN artifact contains 5,388,098 bytes. The latter occupies approximately 39 percent of the former’s storage, or about 61 percent less. This is a comparison of saved artifact sizes. It does not measure runtime memory, prove an equivalent speed improvement, or isolate quantization from other differences between the two formats.

## Inference and Image Generation

After exporting the RKNN model, `onnx2rknn_step2.py` initializes the Toolkit runtime and performs inference. Its default image input is `dataset/bus.jpg`, with video processing disabled. The script passes the prepared image to `rknn.inference`, processes the returned predictions, and invokes the visualization function when detections are available. Thus, this script connects the conversion experiment to the final object detection images.

The distinction between input and output folders is important. The default workflow reads an image from `dataset/` and writes the resulting images to `result/`. The files in `result/` are saved products of inference, rather than the default inputs to the two conversion scripts. Describing this accurately makes the procedure easier for another group to reproduce.

Preprocessing begins with OpenCV image loading. The utility function converts BGR color ordering to RGB and uses letterboxing to fit the image into the 480-by-640 input space while preserving its aspect ratio. Letterboxing introduces padding where needed. The saved output visibly contains gray side padding, consistent with this procedure for the portrait-oriented source photograph.

The scripts use a confidence threshold of 0.25 and an intersection-over-union threshold of 0.45. Confidence filtering removes low-scoring predictions. Non-maximum suppression compares overlapping boxes and retains selected detections to reduce duplication. These settings affect the final image: changing them can alter the number of boxes even when the model and source photograph remain unchanged.

The visualization function draws red rectangles and green category labels with confidence scores. It saves `result/origin_image.jpg`, containing the prepared image without detection annotations, and `result/visual_image.jpg`, containing a side-by-side comparison. The left half is unannotated and the right half displays the predictions. This format lets a reader directly compare the scene with the detector’s interpretation.

## Results and Interpretation

The saved visualization shows a bus and three person detections. The bus label displays a confidence score of approximately 0.86. Two readable person labels show approximately 0.88 and 0.78, while overlapping text makes another label harder to read. These observations come directly from the saved image. Confidence values describe individual predictions and should not be reported as the overall accuracy of the system.

The strongest supported result is the presence of the conversion intermediates, the exported RKNN model, and the final annotated image. Together with the group’s completed workflow, these provide evidence of practical progress through the toolchain. The saved image alone does not identify its generating runtime, because the ONNX and RKNN paths share the visualization function and output names. A terminal recording would connect a particular run to its output more conclusively.

We also need to distinguish host-side execution from board execution. The conversion script initializes its runtime without an explicit hardware target argument, while `rknn_lite_inference.py` provides a separate board deployment path. The materials inspected here do not include board execution logs. Consequently, our results establish model conversion and a saved detection demonstration without establishing measured RK3588 acceleration or successful execution on a particular Orange Pi board.

No recorded precision, recall, mean average precision, power measurements, or repeated latency measurements accompany the output. Our experiment is therefore best understood as a functional demonstration. That remains a meaningful result because the assignment emphasizes installing the toolchain and performing conversion, but the evidence does not support claims of benchmark superiority or production readiness.

## Group Reflections and Future Experiments

From a group perspective, the project showed how much of applied AI involves integrating components. The neural network was only one element. Python compatibility, toolkit installation, directory paths, model dimensions, intermediate files, preprocessing, and visualization all affected whether we could obtain a usable result. Following the flow of data through these components made the project more understandable than treating each command as an isolated instruction.

For future work, Rockchip’s Model Zoo offers additional deployment examples. [5](https://github.com/airockchip/rknn_model_zoo) We would use a separate labeled evaluation set containing varied lighting, object sizes, occlusion, and scene types. We would run both ONNX and RKNN inference with consistent thresholds, preserve outputs under unique names, and compare detections with the annotations. This would support a more informative evaluation of whether conversion changes detection quality across a range of inputs.

We would also measure repeated inference runs on the intended RK3588 board. A useful experiment would separate model execution time from preprocessing and visualization, include warm-up runs, and record the hardware and software configuration. Such measurements would help us evaluate the practical value of hardware acceleration while avoiding conclusions based on a single image or a single timing sample.

## References

1. Ultralytics. [YOLOv8 documentation](https://docs.ultralytics.com/models/yolov8/).
2. ONNX. [Introduction to ONNX](https://onnx.ai/onnx/intro/).
3. ONNX Runtime. [Documentation](https://onnxruntime.ai/docs/).
4. Rockchip. [RKNN Toolkit2](https://github.com/airockchip/rknn-toolkit2).
5. Rockchip. [RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo).
6. laitathei. [Upstream detection project](https://github.com/laitathei/YOLOv8-ONNX-RKNN-HORIZON-TensorRT-Detection).

Project evidence: local setup instructions, conversion and inference scripts, saved models, and detection images. References accessed November 5, 2026.

## Closing Thoughts

Our group’s main achievement was working through the practical connection between a pretrained object detector and an edge deployment toolchain. We had to set up ONNX and RKNN, understand their different roles, run the two conversion stages, and connect inference predictions to saved images. The exported RKNN model and the bus-and-person visualization gave us concrete artifacts that demonstrate the purpose of the workflow.

We leave the project with a clearer understanding of model portability, calibration, runtime interfaces, and the responsibilities of preprocessing and postprocessing. More broadly, we learned that deploying AI requires careful coordination between software and hardware assumptions. This experience gives us a practical foundation for expanding the experiment, improving reproducibility, and evaluating object detection on the intended edge platform.
