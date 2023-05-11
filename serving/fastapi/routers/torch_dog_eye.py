from fastapi import APIRouter, File, UploadFile

from setup.config import DataInput, PredictOutput
from setup.config import ProjectConfig
from routers.modules.resnet import *

from io import BytesIO
import PIL
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

# Project config 설정
project_config = ProjectConfig('tde')
# 라벨가져오기
labels = project_config.load_label()

# 모델 가져오기
# 클래스의 갯수도 미리 확인하자
model = ResNet(num_classes=len(labels))
model.load_state_dict(project_config.load_model())
# label도 읽어서 처리해야함
model.eval()

to_tensor=transforms.Compose([
    # 학습시 사이즈를 확인해서 가져 올수 있게 하자
    # 파라미터를 정규화를 해서 사용해보자
    transforms.Resize([64,64]), 
    transforms.ToTensor()
])

tde = APIRouter(prefix='/torch-dog-eye')

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file)).convert('RGB')
    #im = Image.open("image.png").convert('RGBA')
    return image

# router 마다 경로 설정
@tde.get('/', tags=['torch-dog-eye'])
async def start_ncf():
    return {'msg' : 'Here is TDE'}

#@tde.post('/predict', tags=['tde'], response_model=PredictOutput)
@tde.post('/predict', tags=['tde'])
async def predict_api(file: UploadFile = File(...)):
    # json형식으로 전달할경우 post 처리가 커짐
    # 이미지 업로드후 경로를 전달해서 처리하는걸로 합시다
    # 업로드된 이미지만 읽어서 처리하는 로직을 추가할필요있음
    #print(data_request.img_path);

    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    
    img = read_imagefile(await file.read())
    tensor_img = to_tensor(img)

    with torch.no_grad():
        output = model(torch.unsqueeze(tensor_img, 0))
        percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100
        _, indices = torch.sort(output, descending=True)

    return {'result' : [(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]}