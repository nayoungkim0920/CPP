# 이미지프로세싱처리별 속도비교 및 객체인식 속도 비교 

(OpenCV/IPP/NPP/CUDA/CUDAKernel/GStreamer)

Date. 2024-08-06

1) 기능 : Open, Save, Exit,

         Grayscale, Gaussian Blur, Canny Edges, Median Filter, Sobel Filter, Laplacian Filter,

         Bilateral Filter,

         Rotate, Zoom in, Zoom Out, Undo, Redo, First

         처리속도/입출력정보

3) 개발도구 : Visual Studio 2019, Visual C++, CMAKE 3.14, QT 6.7.1

2-1) LIB/SDK/API : OpenCV, CUDA 12.1, CUDNN v9.1, GSteramer 1.0, Torch, flatbuffers, Protobuf, GLS
                   VulkanSDK 1.3.290.0, ipp 2021.11, IPLIB, Abseil, CAFFE2, ONNX Runtime, cuSPARSELt

2-2) 객체인식 : ONNX 

3) 화면

3-1) 이미지프로세싱처리별 속도비교

![Image](https://github.com/user-attachments/assets/7c7c30ab-c68e-41ba-83ff-2687f1ab8b4a)

3-2) 이미지프로세싱처리별 객체인식 속도비교 (ONNX 진행중)
