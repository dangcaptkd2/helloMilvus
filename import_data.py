import pandas as pd
df_url = pd.read_csv('DB_Newpaper_object_reference_20221107.csv')

from PIL import Image
import requests
from io import BytesIO

from torchvision import models, transforms
import torch
from tqdm import tqdm

from pymilvus import connections
import redis
from pymilvus import CollectionSchema, FieldSchema, DataType, Collection

from sklearn import preprocessing

transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
     ])
model = models.efficientnet_b0(pretrained=True)
newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))
newmodel.eval()

def extract_embedding(model, image):
    x = transform(image)
    x = x.unsqueeze(0)
    embed = model(x)
    embed = embed.squeeze()
    return embed

connections.connect(host="127.0.0.1", port=19530)
red = redis.Redis(host = '127.0.0.1', port=6379, db=0)
red.flushdb()

#Creat a collection

collection_name = "VNE_image_search"
dim = 1280
default_fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
#     FieldSchema(name="name", dtype=DataType.STRING, description="image label"),
    FieldSchema(name="norm_vector", dtype=DataType.FLOAT_VECTOR, dim=dim)
]
default_schema = CollectionSchema(fields=default_fields, description="Collection for image in VNE")

collection = Collection(name=collection_name, schema=default_schema, consistency_level="Strong")

# Create IVF_SQ8 index to the  collection
default_index = {"index_type": "IVF_FLAT", "params": {"nlist": 1280}, "metric_type": "IP"}
collection.create_index(field_name="norm_vector", index_params=default_index)
collection.load()

from tqdm import tqdm
import matplotlib.pyplot as plt
embeds = []
urls = []
c=0
for index, row in tqdm(list(df_url.iterrows())):
    try:
        response = requests.get(row['full_url'])
        img = Image.open(BytesIO(response.content)).convert("RGB") 
        v = extract_embedding(newmodel, img)
        v = v.unsqueeze(0)
        norm_v = preprocessing.normalize(v.detach().numpy(), norm='l2')
        norm_v = norm_v.tolist()
        hello_milvus = collection.insert([norm_v])
        id = hello_milvus.primary_keys[0]
        red.set(str(id), row['full_url'])
    except:
        c+=1
    
print("error url:", c)
print("num entities:", collection.num_entities)


        