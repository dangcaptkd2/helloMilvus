import pandas as pd
df_url = pd.read_csv('DB_Newpaper_object_reference_20221107.csv')

from PIL import Image
import requests
from io import BytesIO
import numpy as np
import cv2
from sklearn import preprocessing
import argparse

from torchvision import models, transforms
import torch
from tqdm import tqdm

from pymilvus import connections
import redis
from pymilvus import CollectionSchema, FieldSchema, DataType, Collection

from deepface import DeepFace


transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

model = models.efficientnet_b0(pretrained=True)
newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))
newmodel.eval()
model_yolov5 = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def setup(arg):
    connections.connect(host="127.0.0.1", port=19530)
    red = redis.Redis(host = '127.0.0.1', port=6379, db=arg.id_db_redis)
    red.flushdb()

    #Creat a collection

    collection_name = arg.collection_name
    dim = arg.dim_vector
    default_fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    #     FieldSchema(name="name", dtype=DataType.STRING, description="image label"),
        FieldSchema(name=arg.name_vec, dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    default_schema = CollectionSchema(fields=default_fields, description="Collection for image in VNE")

    collection = Collection(name=collection_name, schema=default_schema, consistency_level="Strong")

    # Create IVF_SQ8 index to the  collection
    default_index = {"index_type": "IVF_FLAT", "params": {"nlist": 1280}, "metric_type": "IP"}
    collection.create_index(field_name=arg.name_vec, index_params=default_index)
    collection.load()

    return collection, red

def extract_embedding(model, image):
    x = transform(image)
    x = x.unsqueeze(0)
    embed = model(x)
    embed = embed.squeeze()
    return embed

def get_embedding(collection, red, row, img):
    v = extract_embedding(newmodel, img)
    v = v.unsqueeze(0)
    norm_v = preprocessing.normalize(v.detach().numpy(), norm='l2')
    norm_v = norm_v.tolist()
    hello_milvus = collection.insert([norm_v])
    id = hello_milvus.primary_keys[0]
    red.set(str(id), row['full_url'])

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

def get_embedding_face(collection, red, row, img):
    img = np.asarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    v = extract_embedding_face(model_yolov5, img)
    if v is not None:
        # v = v.unsqueeze(0)
        norm_v = preprocessing.normalize([v], norm='l2')
        norm_v = norm_v.tolist()
        hello_milvus = collection.insert([norm_v])
        id = hello_milvus.primary_keys[0]
        red.set(str(id), row['full_url'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection_name', type=str, default='VNE_image_search_face', help='name of mivus collection')
    parser.add_argument('--id_db_redis', default=1, type=int, help='id of redis server')
    parser.add_argument('--dim_vector', type=int, default=512, help='dimention of vector')
    parser.add_argument('--name_vec', type=str, default='norm_vector_face', help='name of vector')
    parser.add_argument('--description', type=str, default='Collection for image in VNE, include face embedding', help='descrip for schema')
    parser.add_argument('--target', type=str, default='total', help='face | total')

    args = parser.parse_args()
    collection, red = setup(arg=args)
    c=0
    for index, row in tqdm(list(df_url.iterrows())):
        try:
            response = requests.get(row['full_url'])
            img = Image.open(BytesIO(response.content)).convert("RGB") 
            if args.target == 'face':
                get_embedding_face(collection, red, row, img)
            elif args.target == 'total':
                get_embedding(collection, red, row, img)
            else:
                print("Error, target nmust be: 'face' or 'total'")
        except:
            c+=1
        
    print("error url:", c)
    print("num entities:", collection.num_entities)

if __name__ == '__main__':
    main()


        