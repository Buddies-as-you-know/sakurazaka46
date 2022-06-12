from io import BytesIO

import cv2
import numpy as np
import torch
from PIL import Image
from retinaface import RetinaFace

members = ['上村 莉菜', '尾関 梨香', '小池 美波', '小林 由依', '齋藤 冬優花', '菅井 友香', '土生 瑞穂', '原田 葵',
       '井上 梨名', '遠藤 光莉', '大園 玲', '大沼 晶保', '幸阪 茉里乃', '関 有美子', '武元 唯衣', '田村 保乃', '藤吉 夏鈴', 
       '増本 綺良', '森田 ひかる', '松田 里奈', '守屋 麗奈', '山﨑 天']
def load_model():
    net = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
    model_path = 'nvidia_efficientnet_b0.pth'
    net.load_state_dict(torch.load(model_path))
    return net

def read_image(image_encoded):
    pil_image = cv2.imread(str(image_encoded))
    return pil_image

def preprocess(images):
    resp = RetinaFace.detect_faces(images, threshold = 0.5)
    face_cut = []
    faces = []
    for key in resp:
        p = resp[key]
        facial_area = p["facial_area"]
        face = images[facial_area[1]: facial_area[3], facial_area[0]: facial_area[2]]
        faces.append(face)
        face = np.array(cv2.resize(face, (244, 244))/255.).transpose(2, 0, 1).astype(np.float32)
        #face= cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_cut.append(face)
    face_cut = np.array(face_cut) 
    return face_cut

def predict(image: np.ndarray):
    net = load_model()
    if net is None:
        net = load_model()
    images = torch.tensor(image)
    #print(images.shape)
    response = []
    with torch.no_grad():
        for im in images:
            #print(im.shape)
            im = im.view(1,3,244,244)
            outputs = net(im)
            #print(outputs.data)
            _, predicted = torch.max(outputs.data, 1)
            #print(predicted)
            response += [members[predicted]]
    return response
