<python으로 작성된 캠으로 컵객체인식하는 프로그램 환경설정>

1. 가상환경

-파이썬 버전
C:\>python --version
Python 3.12.3

-생성
C:\>python -m venv pythonEnv

-활성화
C:\>pythonEnv\Scripts\activate
(pythonEnv) C:\>

-비활성화
(pythonEnv) C:\>deactivate

-가상환경 삭제
C:\>rmdir /s /q pythonEnv

2. 모듈설치
-실행시키면서 설치할모듈 설치
  
Exception has occurred: ModuleNotFoundError
No module named 'cv2'
  File "C:\ObjectDetect\camDetect.py", line 1, in <module>
    import cv2
ModuleNotFoundError: No module named 'cv2'
(pythonEnv) C:\>pip install opencv-python 
Successfully installed numpy-2.0.1 opencv-python-4.10.0.84

[notice] A new release of pip is available: 24.0 -> 24.2
[notice] To update, run: python.exe -m pip install --upgrade pip
(pythonEnv) C:\>python.exe -m pip install --upgrade pip
Successfully installed pip-24.2

(pythonEnv) C:\>pip list
Package       Version
------------- ---------
numpy         2.0.1
opencv-python 4.10.0.84
pip           24.2
slim          0.1

Exception has occurred: ModuleNotFoundError
No module named 'torch'
  File "C:\ObjectDetect\camDetect.py", line 2, in <module>
    import torch
ModuleNotFoundError: No module named 'torch'

(pythonEnv) C:\>python -c "import torch; print(torch.__version__)"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'torch'

(gpu용)
(pythonEnv) C:\>pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu120
Successfully installed MarkupSafe-2.1.5 filelock-3.15.4 fsspec-2024.6.1 jinja2-3.1.4 mpmath-1.3.0 networkx-3.3 numpy-1.26.4 pillow-10.4.0 setuptools-72.1.0 sympy-1.13.1 torch-2.4.0 torchaudio-2.4.0 torchvision-0.19.0 typing-extensions-4.12.2

(pythonEnv) C:\>python -c "import torch; print(torch.__version__)"
2.4.0+cpu

(pythonEnv) C:\>pip uninstall torch torchvision torchaudio
Successfully uninstalled torch-2.4.0
Successfully uninstalled torchvision-0.19.0
Successfully uninstalled torchaudio-2.4.0

(pythonEnv) C:\>pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
Looking in indexes: https://download.pytorch.org/whl/cu121
Successfully installed torch-2.4.0+cu121 torchaudio-2.4.0+cu121 torchvision-0.19.0+cu121
(pythonEnv) C:\>python -c "import torch; print(torch.__version__)"
(pythonEnv) C:\>python cudaInfo.py
C:\Users\nayou\AppData\Local\Programs\Python\Python312\python.exe: can't open file 'C:\\cudaInfo.py': [Errno 2] No such file or directory

(pythonEnv) C:\>python C:\pythonEnv\cudaInfo.py
import torch

# PyTorch 버전 출력
print("PyTorch version:", torch.__version__)

# CUDA가 사용 가능한지 확인
print("CUDA available:", torch.cuda.is_available())

# GPU 장치 목록 출력
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        
(output)      
PyTorch version: 2.4.0+cu121
CUDA available: True
CUDA device count: 1
Device 0: NVIDIA GeForce RTX 4060 Laptop GPU

Exception has occurred: ModuleNotFoundError
No module named 'requests'
  File "C:\ObjectDetect\yolov5\utils\downloads.py", line 9, in <module>
    import requests
  File "C:\ObjectDetect\yolov5\models\experimental.py", line 10, in <module>
    from yolov5.utils.downloads import attempt_download
  File "C:\ObjectDetect\camDetect.py", line 3, in <module>
    from yolov5.models.experimental import attempt_load
ModuleNotFoundError: No module named 'requests'
(pythonEnv) C:\>pip install requests
Successfully installed certifi-2024.7.4 charset-normalizer-3.3.2 idna-3.7 requests-2.32.3 urllib3-2.2.2
(pythonEnv) C:\>pip show requests
Name: requests
Version: 2.32.3
Summary: Python HTTP for Humans.
Home-page: https://requests.readthedocs.io
Author: Kenneth Reitz
Author-email: me@kennethreitz.org
License: Apache-2.0
Location: C:\pythonEnv\Lib\site-packages
Requires: certifi, charset-normalizer, idna, urllib3
Required-by:

Exception has occurred: ModuleNotFoundError
No module named 'pandas'
  File "C:\ObjectDetect\yolov5\utils\general.py", line 31, in <module>
    import pandas as pd
  File "C:\ObjectDetect\camDetect.py", line 4, in <module>
    from yolov5.utils.general import non_max_suppression
ModuleNotFoundError: No module named 'pandas'
(pythonEnv) C:\>pip install pandas
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
slim 0.1 requires tf-slim>=1.1, which is not installed.
Successfully installed pandas-2.2.2 python-dateutil-2.9.0.post0 pytz-2024.1 six-1.16.0 tzdata-2024.1
>>> import pandas as pd
>>> print(pd.__version__)
2.2.2
tf-slim 충돌 -> tf-slim 패키지를 설치
(pythonEnv) C:\>pip install tf-slim
Successfully installed absl-py-2.1.0 tf-slim-1.1.0

Exception has occurred: ModuleNotFoundError
No module named 'yaml'
  File "C:\ObjectDetect\yolov5\utils\general.py", line 35, in <module>
    import yaml
  File "C:\ObjectDetect\camDetect.py", line 4, in <module>
    from yolov5.utils.general import non_max_suppression
ModuleNotFoundError: No module named 'yaml'
(pythonEnv) C:\>pip install pyyaml
Successfully installed pyyaml-6.0.2

C:\ObjectDetect\yolov5\utils\general.py
os.system("pip install -U ultralytics")
Successfully installed opencv-python-4.10.0.84 ultralytics-8.2.75

Exception has occurred: ModuleNotFoundError
No module named 'ultralytics.utils'
  File "C:\ObjectDetect\yolov5\utils\general.py", line 48, in <module>
    from ultralytics.utils.checks import check_requirements
  File "C:\ObjectDetect\camDetect.py", line 4, in <module>
    from yolov5.utils.general import non_max_suppression
ModuleNotFoundError: No module named 'ultralytics.utils'
(pythonEnv) C:\>pip install ultralytics
Successfully installed colorama-0.4.6 contourpy-1.2.1 cycler-0.12.1 fonttools-4.53.1 kiwisolver-1.4.5 matplotlib-3.9.1.post1 packaging-24.1 psutil-6.0.0 py-cpuinfo-9.0.0 pyparsing-3.1.2 scipy-1.14.0 seaborn-0.13.2 tqdm-4.66.5 ultralytics-8.2.75 ultralytics-thop-2.0.0

Exception has occurred: ModuleNotFoundError
No module named 'dill'
  File "C:\ObjectDetect\yolov5\models\experimental.py", line 98, in attempt_load
    ckpt = torch.load(attempt_download(w), map_location="cpu")  # load
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\ObjectDetect\camDetect.py", line 11, in <module>
    model = attempt_load(weights).float().to(device)  # Load model
            ^^^^^^^^^^^^^^^^^^^^^
ModuleNotFoundError: No module named 'dill'
(pythonEnv) C:\>pip install dill
Successfully installed dill-0.3.8

(output)
Fusing layers... 
Model summary: 212 layers, 20852934 parameters, 0 gradients, 47.9 GFLOPs
성공!

3. C:\ObjectDetect\camDetect.py
import cv2
import torch
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
from pathlib import Path
from datetime import datetime

# Initialize YOLOv5 model
weights = 'C:/yolov5/runs/train/exp35/weights/best.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Check if GPU is available
model = attempt_load(weights).float().to(device)  # Load model

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print('Error reading frame from webcam')
        break

    # Resize frame to match model's expected input size
    img = cv2.resize(frame, (640, 480))
    
    # Convert image to torch tensor
    img = torch.from_numpy(img).to(device)
    img = img.float()  # Convert to float
    img /= 255.0  # Normalize to 0-1
    
    # Ensure correct shape and number of channels
    img = img.permute(2, 0, 1).unsqueeze(0)  # Change shape to [1, 3, H, W]

    # Inference
    pred = model(img)[0]

    # Display results
    for det in pred:
        det[:, :4] *= torch.tensor([frame.shape[1]/640, frame.shape[0]/480, frame.shape[1]/640, frame.shape[0]/480], device=device)  # Scale coordinates to original frame size
        det[:, :4] = det[:, :4].clamp(min=0, max=frame.shape[1])  # Clamp coordinates within frame
        
        # Ensure det is a tensor before passing it to non_max_suppression
        if isinstance(det, torch.Tensor):
            det = non_max_suppression(det.unsqueeze(0), 0.4, 0.5)[0]  # Perform non-maximum suppression
            if det is not None and len(det):
                for *xyxy, conf, cls in det:
                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    
                    # Draw bounding box if confidence is over 90%
                    if conf > 0.9:
                        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                        cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                        # Capture the frame
                        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
                        capture_name = f'{current_time}.jpg'
                        cv2.imwrite(capture_name, frame)
                        print(f'Image captured with confidence: {conf:.2f}')

    cv2.imshow('YOLOv5 Object Detection', frame)
    
    # Check for quit key
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()


                    
