# CE_MOT

## Installation
### 1. Installing on the host machine
```shell
git clone https://github.com/G-uit-student/KMC_MOT.git
cd KMC_MOT
pip install -r requirements.txt
python setup.py develop
pip install cython
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install cython_bbox
```

## Run on valid dataset
Step1. Download bounding boxes for fast inference
```shell
gdown --fuzzy https://drive.google.com/file/d/1OKTlVSgNx91s-TDrkva1TpzTiQPLrmSU/view?usp=share_link
```
Step2. Evaluation
```shell
python tools/track_modify.py
```

You will get IDF1 82.5% and MOTA 78.4%. 
