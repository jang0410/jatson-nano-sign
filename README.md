Jetson 기반 도로 교통 표지판 감지 시스템
Jetson-Based Road Traffic Sign Detection System
프로젝트 설명 | Project Description
이 프로젝트는 YOLOv5를 활용하여 도로 교통 표지판을 실시간으로 감지하며, 안전한 도로 환경을 조성합니다. NVIDIA Jetson 장치를 사용하여 효율적이고 저전력으로 실시간 처리가 가능합니다.
This project uses YOLOv5 to detect road traffic signs in real-time, contributing to safer road environments. It leverages NVIDIA Jetson devices for efficient, low-power real-time processing.

기능 | Features
YOLOv5를 활용한 교통 표지판 감지 | Traffic sign detection using YOLOv5
NVIDIA Jetson 장치에 최적화 | Optimized for NVIDIA Jetson devices
실시간 추론 가능 | Real-time inference capabilities

사전 준비 사항 | Prerequisites
Python 3.8 이상 | 3.8 or higher
NVIDIA GPU (CUDA 지원 | CUDA support)
NVIDIA Jetson 장치 (Nano, Xavier NX 등 | Nano, Xavier NX, etc.) (선택 사항 | optional)
Docker (선택 사항 | optional)

설치 방법 | Setup Instructions
1. 가상 환경 생성 (선택 사항) | Create a Virtual Environment (Optional)
python -m venv venv

3. 가상 환경 활성화 | Activate the Virtual Environment
Windows:
venv\Scripts\activate
Linux/MacOS:
source venv/bin/activate

4. 종속성 설치 | Install Dependencies
pip install -r requirements.txt

4. PyTorch 설치 (CUDA 12.6 지원) | Install PyTorch (CUDA 12.6 Support)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

6. OpenCV 설치 | Install OpenCV
pip install opencv-python

6. YOLOv5 레포지토리 클론 | Clone the YOLOv5 Repository
git clone https://github.com/ultralytics/yolov5.git
cd yolov5

8. YOLOv5 종속성 설치 | Install YOLOv5 Dependencies
pip install -U -r requirements.txt

모델 학습 | Model Training
YOLOv5 모델을 학습시키려면 아래 명령어를 실행하세요:
Train the YOLOv5 model using the following command:
python train.py --img 640 --batch 16 --epochs 10 --data ./data.yaml --weights yolov5s.pt

--img: 이미지 크기 (예: 640x640) | Image size (e.g., 640x640)
--batch: 배치 크기 (예: 16) | Batch size (e.g., 16)
--epochs: 학습 에폭 수 | Number of training epochs

추론 실행 | Run Inference
학습된 모델로 추론을 실행하려면 아래 명령어를 사용하세요:
Run inference on images using the trained model:
python detect.py --weights runs/train/exp/weights/best.pt --img 640 --source path/to/images

데이터셋 | Dataset
공개된 교통 표지판 데이터셋을 사용하거나 직접 데이터를 수집할 수 있습니다.
A sample dataset is available at:
Roboflow Traffic Sign Dataset

테스트 및 평가 | Testing and Evaluation
학습 후 모델을 평가하려면 아래 명령어를 사용하세요:
After training, evaluate the model using:
python test.py --weights runs/train/exp/weights/best.pt --data data.yaml
