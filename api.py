from torchvision import models, transforms
import torch
from sklearn import preprocessing
from PIL import Image

torch.manual_seed(0)

transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
     ])
model = models.efficientnet_b0(pretrained=True)
newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))
newmodel.eval()

from pymilvus import Collection
from pymilvus import connections
import redis
connections.connect(host="127.0.0.1", port=19530)
red = redis.Redis(host = '127.0.0.1', port=6379, db=0)
collection = Collection("VNE_image_search")

def extract_embedding(model, image):
    with torch.no_grad():
        x = transform(image)
        x = x.unsqueeze(0)
        embed = model(x)
        embed = embed.squeeze()
        return embed

def process(img_path):
    image = Image.open(img_path)
    vec_query = extract_embedding(newmodel,image)
    vec_query = vec_query.unsqueeze(0)
    norm_vec_query = preprocessing.normalize(vec_query.detach().numpy(), norm='l2')
    norm_vec_query = norm_vec_query.squeeze()
    norm_vec_query = norm_vec_query.tolist()
    print("????", norm_vec_query[:20])
    search_params = {"metric_type": "IP", "params": {"nprobe": 32}}
    results = collection.search([norm_vec_query], "norm_vector", param=search_params, limit=10, expr=None)
    result_files = [red.get(y.id).decode('utf-8') for y in results[0]]
    distances = [y.distance for y in results[0]]
    distances = [str(round(i,3)) for i in distances]
    print(">>>>>", distances)
    print(">>>>>", result_files)
    
    return list(zip(result_files, distances))


def main():
    return 0

if __name__ == '__main__':
    print(">>>>", collection.num_entities)


    
    
