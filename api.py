from torchvision import models, transforms
import torch
from sklearn import preprocessing
from PIL import Image
from deepface import DeepFace
import numpy as np
import cv2

torch.manual_seed(0)

transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
     ])
model = models.efficientnet_b0(pretrained=True)
newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))
newmodel = newmodel.to('cpu')
newmodel.eval()
model_yolov5 = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model_yolov5 = model_yolov5.to('cpu')

from pymilvus import Collection
from pymilvus import connections
import redis
connections.connect(host="127.0.0.1", port=19530)
red = redis.Redis(host = '127.0.0.1', port=6379, db=0)
collection_total = Collection("VNE_image_search")
collection_face = Collection("VNE_image_search_face")
red_face = redis.Redis(host = '127.0.0.1', port=6379, db=1)

def extract_embedding(model, image):
    with torch.no_grad():
        x = transform(image)
        x = x.unsqueeze(0)
        embed = model(x)
        embed = embed.squeeze()
        return embed
def extract_embedding_face(model,img):
    results = model(img)
    xyxy = []
    s_max=0
    for pred in results.pred[0].tolist():
        x1,y1,x2,y2,conf,cls = pred
        x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
        if int(cls)==0:
            s = abs(x2-x1)*abs(y2-y1)
            if s>s_max:
                xyxy = [x1,y1,x2,y2]
                s_max=s
    if not xyxy:
        return None
    crop_image  = img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
    
    embedding = DeepFace.represent(img_path = crop_image, 
                               model_name = 'Facenet512', model = None, enforce_detection = True, 
                               detector_backend = 'retinaface', align = True, normalization = 'Facenet2018')
    
    return embedding

def search_total(image):
    vec_query = extract_embedding(newmodel,image)
    vec_query = vec_query.unsqueeze(0)
    norm_vec_query = preprocessing.normalize(vec_query.detach().numpy(), norm='l2')
    norm_vec_query = norm_vec_query.squeeze()
    norm_vec_query = norm_vec_query.tolist()
    print("????", norm_vec_query[:20])
    search_params = {"metric_type": "IP", "params": {"nprobe": 32}}
    results = collection_total.search([norm_vec_query], "norm_vector", param=search_params, limit=10, expr=None)
    result_files = [red.get(y.id).decode('utf-8') for y in results[0]]
    distances = [y.distance for y in results[0]]
    distances = [str(round(i,3)) for i in distances]
    print(">>>>>", distances)
    print(">>>>>", result_files)
    return result_files, distances

def search_face(image):
    img = np.asarray(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    vec_query = extract_embedding_face(model=model_yolov5,img=img)
    if vec_query is None:
        return [], []
    norm_vec_query = preprocessing.normalize([vec_query], norm='l2')
    search_params = {"metric_type": "IP", "params": {"nprobe": 32}}
    print("?????", norm_vec_query)
    results = collection_face.search(norm_vec_query, "norm_vector_face", param=search_params, limit=10, expr=None)
    result_files = [red_face.get(y.id).decode('utf-8') for y in results[0]]
    distances = [y.distance for y in results[0]]
    distances = [str(round(i,3)) for i in distances]
    print(">>>>>", distances)
    print(">>>>>", result_files)
    return result_files, distances

def process(img_path):
    image = Image.open(img_path)
    result_files_total, distances_total = search_total(image=image)
    result_files_face, distances_face = search_face(image=image)
    
    return list(zip(result_files_total, distances_total)), list(zip(result_files_face, distances_face))


def main():
    return 0

if __name__ == '__main__':
    print('hello')
    # print(">>>>", collection.num_entities)


    
    
