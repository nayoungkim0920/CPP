<python onnx 이미지 객체인식>
1. 가상환경
(pythonEnv) C:\>

2.모듈설치
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

