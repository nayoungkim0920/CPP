<python onnx 이미지 객체인식>
1. 가상환경
(pythonEnv) C:\>

2. yolov5 best.pt -> best.onnx export 프로그램
(yolov5 옵션 참고)
python c:/yolov5/train.py --img 640 --batch 16 --epochs 200 --data c:/data.yaml --weights c:/yolov5/models/yolov5m.pt --device 0
# C:\data.yaml
train: c:/cup_test/train/images
val: c:/cup_test/valid/images
test: c:/cup_test/test/images

nc: 1
names: ['cup']

mosaic: True  # Mosaic 데이터 증강 활성화, 모델의 일반화성능향상
mixup: 0.5  # Mixup 데이터 증강 비율 (0 ~ 1), 과적합줄일수있음
cache_images: True # 이미지캐싱, 훈련속도향상

roboflow:
  workspace: nayoungkim-xrg93
  project: cup-krqrq
  version: 27
  license: CC BY 4.0
  url: https://universe.roboflow.com/nayoungkim-xrg93/cup-krqrq/dataset/27

(위의 정보를 가지고 프로그램 작성)
# C:\myLab\Project1\Project1\python\myTorchScript.py
=> "C:\myLab\Project1\Project1\python\best.onnx" 생성
import sys
import torch
import onnx
import onnxruntime as ort
import os

# 디버깅 정보 저장 리스트
debug_info = []

def log_debug_info(message):
    debug_info.append(message)

# YOLOv5 경로를 추가합니다
sys.path.append('C:/yolov5')
log_debug_info("YOLOv5 path added to system path.")

# 모델 경로 설정
model_path = 'C:/yolov5/runs/train/exp30/weights/best.pt'
log_debug_info(f"Model path set to: {model_path}")

# CUDA가 사용 가능한 경우, CUDA로 로드하고, 그렇지 않으면 CPU로 로드합니다
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log_debug_info(f"Using device: {device}")

# 모델 로드
try:
    log_debug_info("Attempting to load the model...")
    # YOLOv5는 모델 구조를 직접 정의한 후 가중치를 로드해야 합니다.
    from models.yolo import Model  # 모델 구조를 정의한 모듈을 가져와야 함
    model = Model(cfg='yolov5m.yaml')  # 모델 구조 로드
    model.load_state_dict(torch.load(model_path)['model'].state_dict())
    model.to(device).eval()
    log_debug_info("Model loaded and set to evaluation mode.")
except Exception as e:
    log_debug_info(f"Error loading model: {str(e)}")
    sys.exit()

# 더미 입력 텐서 생성 (CUDA 또는 CPU에 맞게 설정)
dummy_input = torch.randn(1, 3, 640, 640).to(device)
log_debug_info(f"Dummy input tensor created with shape: {dummy_input.shape}")

# ONNX로 모델 내보내기
onnx_path = 'C:/myLab/Project1/Project1/python/best.onnx'
log_debug_info(f"Export path set to: {onnx_path}")

try:
    log_debug_info("Attempting to export model to ONNX format...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        verbose=True,
        opset_version=12,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    if os.path.exists(onnx_path):
        log_debug_info(f"ONNX file successfully created at: {onnx_path}")
    else:
        log_debug_info(f"Failed to create ONNX file at: {onnx_path}")
except Exception as e:
    log_debug_info(f"Error exporting model to ONNX: {str(e)}")
    sys.exit()

# ONNX Runtime 버전과 CUDA 버전 출력
try:
    log_debug_info(f"ONNX Runtime version: {ort.__version__}")
except Exception as e:
    log_debug_info(f"Error getting ONNX Runtime version: {str(e)}")

try:
    log_debug_info(f"CUDA version: {torch.version.cuda}")
except Exception as e:
    log_debug_info(f"Error getting CUDA version: {str(e)}")

# ONNX 모델 검증
try:
    log_debug_info("Attempting to validate the ONNX model...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    log_debug_info("ONNX model is valid.")
except Exception as e:
    log_debug_info(f"Error validating ONNX model: {str(e)}")
    sys.exit()

# ONNX Runtime 세션 생성 (GPU 사용)
try:
    log_debug_info("Creating ONNX Runtime session...")
    providers = ['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    ort_session = ort.InferenceSession(onnx_path, providers=providers)
    log_debug_info(f"ONNX Runtime session created with {'GPU' if torch.cuda.is_available() else 'CPU'} support.")
except Exception as e:
    log_debug_info(f"Error creating ONNX Runtime session: {str(e)}")
    sys.exit()

# 더미 입력 준비
dummy_input_np = dummy_input.cpu().numpy()
log_debug_info(f"Dummy input numpy array created with shape: {dummy_input_np.shape}")

# 추론 실행
try:
    log_debug_info("Running inference...")
    outputs = ort_session.run(None, {'input': dummy_input_np})
    log_debug_info("Inference completed. Outputs:")
    log_debug_info(str(outputs))
except Exception as e:
    log_debug_info(f"Error running inference: {str(e)}")

# 모든 디버깅 정보를 출력
print("\n--- Debug Information ---")
for info in debug_info:
    print(info)

3.모듈설치
명령프롬프트(관리자권한실행)
Exception has occurred: ModuleNotFoundError
No module named 'onnxruntime'
  File "C:\ObjectDetect\onnxDetectImage.py", line 3, in <module>
    import onnxruntime as ort
ModuleNotFoundError: No module named 'onnxruntime'
(pythonEnv) C:\>pip install onnxruntime
Successfully installed coloredlogs-15.0.1 flatbuffers-24.3.25 humanfriendly-10.0 onnxruntime-1.18.1 protobuf-5.27.3 pyreadline3-3.4.1
참고)명령프롬프트(관리자권한실행하지않고하는방법)
C:\Users\nayou>pip install --user onnxruntime
Successfully installed coloredlogs-15.0.1 humanfriendly-10.0 onnxruntime-1.18.1 pyreadline3-3.4.1

4. 파이썬테스트
#가상환경(c:/pythonEnv)
#C:\ObjectDetect\onnxDetectImage.py
=>바운딩박스도 잘그려지고 확률도 잘출력됨을 확인
import sys
import torch
import onnx
import onnxruntime as ort
import os

# 디버깅 정보 저장 리스트
debug_info = []

def log_debug_info(message):
    debug_info.append(message)

# YOLOv5 경로를 추가합니다
sys.path.append('C:/yolov5')
log_debug_info("YOLOv5 path added to system path.")

# 모델 경로 설정
model_path = 'C:/yolov5/runs/train/exp30/weights/best.pt'
log_debug_info(f"Model path set to: {model_path}")

# CUDA가 사용 가능한 경우, CUDA로 로드하고, 그렇지 않으면 CPU로 로드합니다
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log_debug_info(f"Using device: {device}")

# 모델 로드
try:
    log_debug_info("Attempting to load the model...")
    # YOLOv5는 모델 구조를 직접 정의한 후 가중치를 로드해야 합니다.
    from models.yolo import Model  # 모델 구조를 정의한 모듈을 가져와야 함
    model = Model(cfg='yolov5m.yaml')  # 모델 구조 로드
    model.load_state_dict(torch.load(model_path)['model'].state_dict())
    model.to(device).eval()
    log_debug_info("Model loaded and set to evaluation mode.")
except Exception as e:
    log_debug_info(f"Error loading model: {str(e)}")
    sys.exit()

# 더미 입력 텐서 생성 (CUDA 또는 CPU에 맞게 설정)
dummy_input = torch.randn(1, 3, 640, 640).to(device)
log_debug_info(f"Dummy input tensor created with shape: {dummy_input.shape}")

# ONNX로 모델 내보내기
onnx_path = 'C:/myLab/Project1/Project1/python/best.onnx'
log_debug_info(f"Export path set to: {onnx_path}")

try:
    log_debug_info("Attempting to export model to ONNX format...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        verbose=True,
        opset_version=12,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    if os.path.exists(onnx_path):
        log_debug_info(f"ONNX file successfully created at: {onnx_path}")
    else:
        log_debug_info(f"Failed to create ONNX file at: {onnx_path}")
except Exception as e:
    log_debug_info(f"Error exporting model to ONNX: {str(e)}")
    sys.exit()

# ONNX Runtime 버전과 CUDA 버전 출력
try:
    log_debug_info(f"ONNX Runtime version: {ort.__version__}")
except Exception as e:
    log_debug_info(f"Error getting ONNX Runtime version: {str(e)}")

try:
    log_debug_info(f"CUDA version: {torch.version.cuda}")
except Exception as e:
    log_debug_info(f"Error getting CUDA version: {str(e)}")

# ONNX 모델 검증
try:
    log_debug_info("Attempting to validate the ONNX model...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    log_debug_info("ONNX model is valid.")
except Exception as e:
    log_debug_info(f"Error validating ONNX model: {str(e)}")
    sys.exit()

# ONNX Runtime 세션 생성 (GPU 사용)
try:
    log_debug_info("Creating ONNX Runtime session...")
    providers = ['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    ort_session = ort.InferenceSession(onnx_path, providers=providers)
    log_debug_info(f"ONNX Runtime session created with {'GPU' if torch.cuda.is_available() else 'CPU'} support.")
except Exception as e:
    log_debug_info(f"Error creating ONNX Runtime session: {str(e)}")
    sys.exit()

# 더미 입력 준비
dummy_input_np = dummy_input.cpu().numpy()
log_debug_info(f"Dummy input numpy array created with shape: {dummy_input_np.shape}")

# 추론 실행
try:
    log_debug_info("Running inference...")
    outputs = ort_session.run(None, {'input': dummy_input_np})
    log_debug_info("Inference completed. Outputs:")
    log_debug_info(str(outputs))
except Exception as e:
    log_debug_info(f"Error running inference: {str(e)}")

# 모든 디버깅 정보를 출력
print("\n--- Debug Information ---")
for info in debug_info:
    print(info)

5. C++프로그램구현

























