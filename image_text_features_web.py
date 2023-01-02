import uvicorn
if __name__ == '__main__':
    uvicorn.run('image_text_features_web:app', host='127.0.0.1', port=33338, log_level="info")
    exit()

from os.path import exists
from typing import Optional, Union
import torch
from pydantic import BaseModel
from fastapi import FastAPI, File,Form, HTTPException, Response, status
import numpy as np
import asyncio
from PIL import Image
import torch
import clip
from torchvision import transforms
# from pathlib import Path
import io
import faiss
import lmdb
# import pickle

dim = 512
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {device}")
model, _ = clip.load("ViT-B/16", device=device)
# model = torch.jit.optimize_for_inference(torch.jit.script(model.eval()))
model.eval()
model.to(device)
index = None
DATA_CHANGED_SINCE_LAST_SAVE = False
# pca_w_file = Path("./pca_w.pkl")
pca = None
# if pca_w_file.is_file():
#     with open(pca_w_file, 'rb') as pickle_file:
#         pca = pickle.load(pickle_file)
app = FastAPI()
_transform=transforms.Compose([
                       transforms.Resize((224,224)),
                       transforms.ToTensor(),
                       transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

def main():
    global DB_features
    DB_features = lmdb.open('./features.lmdb',map_size=500*1_000_000) #5000mb
    init_index()
    loop = asyncio.get_event_loop()
    loop.call_later(10, periodically_save_index,loop)

def init_index():
    global index
    if exists("./populated.index"):
        index = faiss.read_index("./populated.index")
    else:
        print("Index is not found! Exiting...")
        exit()

def transform(im):
    # desired_size = 224
    # old_size = im.size  # old_size[0] is in (width, height) format
    # ratio = float(desired_size)/max(old_size)
    # new_size = tuple([int(x*ratio) for x in old_size])
    # im = im.resize(new_size, Image.Resampling.LANCZOS)
    # new_im = Image.new("RGB", (desired_size, desired_size))
    # new_im.paste(im, ((desired_size-new_size[0])//2, (desired_size-new_size[1])//2))
    return _transform(im)

def int_to_bytes(x: int) -> bytes:
    return x.to_bytes(4, 'big')

def delete_descriptor_by_id(id):
    with DB_features.begin(write=True,buffers=True) as txn:
        txn.delete(int_to_bytes(id))   #True = deleted False = not found

def add_descriptor(id, features):
    with DB_features.begin(write=True, buffers=True) as txn:
        txn.put(int_to_bytes(id), np.frombuffer(features,dtype=np.float32))

def read_img_buffer(image_data):
    img = Image.open(io.BytesIO(image_data))
    # img=img.convert('L').convert('RGB') #GREYSCALE
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img

def get_features(image_buffer):
    image=read_img_buffer(image_buffer)
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feature_vector = model.encode_image(image)
        feature_vector/=torch.linalg.norm(feature_vector)
    return feature_vector.cpu().numpy().astype(np.float32)

def get_text_features(text):
    text_tokenized = clip.tokenize([text]).to(device)
    with torch.no_grad():
        feature_vector = model.encode_text(text_tokenized)
        feature_vector/=torch.linalg.norm(feature_vector)
    return feature_vector.cpu().numpy().astype(np.float32)

def get_aqe_vector(feature_vector, n, alpha):
    _, I = index.search(feature_vector, n)
    top_features=[]
    for i in range(n):
        top_features.append(index.reconstruct(int(list(I[0])[i])).flatten())
    new_feature=[]
    for i in range(dim):
        _sum=0
        for j in range(n):
            _sum+=top_features[j][i] * np.dot(feature_vector, top_features[j].T)**alpha
        new_feature.append(_sum)

    new_feature=np.array(new_feature)
    new_feature/=np.linalg.norm(new_feature)
    new_feature=new_feature.astype(np.float32).reshape(1,-1)
    return new_feature

def nn_find_similar(feature_vector, k, distance_threshold, aqe_n, aqe_alpha):
    if aqe_n is not None and aqe_alpha is not None:
        feature_vector=get_aqe_vector(feature_vector,aqe_n, aqe_alpha)
    if k is not None:
        D, I = index.search(feature_vector, k)
        D = D.flatten()
        I = I.flatten()
    elif distance_threshold is not None:
        _, D, I = index.range_search(feature_vector, distance_threshold)

    res=[{"image_id":int(I[i]),"distance":float(D[i])} for i in range(len(D))]
    res = sorted(res, key=lambda x: x["distance"]) 
    return res

@app.get("/")
async def read_root():
    return {"Hello": "World"}

class Item_image_text_features_get_similar_images_by_id(BaseModel):
    image_id: int
    k: Union[str,int,None] = None
    distance_threshold: Union[str,float,None] = None
    aqe_n: Union[str,int,None] = None
    aqe_alpha: Union[str,float,None] = None

class Item_image_text_features_get_similar_images_by_text(BaseModel):
    text: str
    k: Union[str,int,None] = None
    distance_threshold: Union[str,float,None] = None
    aqe_n: Union[str,int,None] = None
    aqe_alpha: Union[str,float,None] = None

@app.post("/image_text_features_get_similar_images_by_id")
async def image_text_features_get_similar_images_by_id_handler(item: Item_image_text_features_get_similar_images_by_id):
    try:
        k=item.k
        distance_threshold=item.distance_threshold
        aqe_n = item.aqe_n
        aqe_alpha = item.aqe_alpha
        if k:
            k = int(k)
        if distance_threshold:
            distance_threshold = float(distance_threshold)
        if aqe_n:
            aqe_n = int(aqe_n)
        if aqe_alpha:
            aqe_alpha = float(aqe_alpha)
        if (k is None) == (distance_threshold is None):
            raise HTTPException(status_code=500, detail="both k and distance_threshold present")

        target_features = index.reconstruct(item.image_id).reshape(1,-1)         
        results = nn_find_similar(target_features, k, distance_threshold, aqe_n, aqe_alpha)
        return results
    except RuntimeError:
        raise HTTPException(status_code=500, detail="Error in image_text_features_get_similar_images_by_id")

@app.post("/image_text_features_get_similar_images_by_image_buffer")
async def image_text_features_get_similar_images_by_image_buffer_handler(image: bytes = File(...), k: Optional[str] = Form(None),
 distance_threshold: Optional[str] = Form(None), aqe_n: Optional[str] = Form(None),aqe_alpha: Optional[str] = Form(None)):
    try:
        if k:
            k = int(k)
        if distance_threshold:
            distance_threshold = float(distance_threshold)
        if aqe_n:
            aqe_n = int(aqe_n)
        if aqe_alpha:
            aqe_alpha = float(aqe_alpha)
        if (k is None) == (distance_threshold is None):
            raise HTTPException(status_code=500, detail="both k and distance_threshold present")

        target_features=get_features(image)
        # if pca:
        #     target_features=pca.transform(target_features)
        #     target_features/=np.linalg.norm(target_features)
        results = nn_find_similar(target_features, k, distance_threshold, aqe_n, aqe_alpha)
        return results
    except RuntimeError:
        raise HTTPException(status_code=500, detail="Error in image_text_features_get_similar_images_by_image_buffer")

@app.post("/image_text_features_get_similar_images_by_text")
async def image_text_features_get_similar_images_by_text_handler(item: Item_image_text_features_get_similar_images_by_text):
    try:
        k=item.k
        distance_threshold=item.distance_threshold
        aqe_n = item.aqe_n
        aqe_alpha = item.aqe_alpha
        if k:
            k = int(k)
        if distance_threshold:
            distance_threshold = float(distance_threshold)
        if aqe_n:
            aqe_n = int(aqe_n)
        if aqe_alpha:
            aqe_alpha = float(aqe_alpha)
        if (k is None) == (distance_threshold is None):
            raise HTTPException(status_code=500, detail="both k and distance_threshold present")

        target_features = get_text_features(item.text)       
        results = nn_find_similar(target_features, k, distance_threshold, aqe_n, aqe_alpha)
        return results
    except RuntimeError:
        raise HTTPException(status_code=500, detail="Error in image_text_features_get_similar_images_by_text")

@app.post("/calculate_image_text_features")
async def calculate_image_text_features_handler(image: bytes = File(...),image_id: str = Form(...)):
    try:
        global DATA_CHANGED_SINCE_LAST_SAVE
        image_id=int(image_id)
        features=get_features(image)
        add_descriptor(image_id, features.astype(np.float32))
        # if pca:
        #     target_features=pca.transform(target_features)
        #     target_features/=np.linalg.norm(target_features)
        index.add_with_ids(features, np.int64([image_id])) # index.add_items(features,[image_id])
        DATA_CHANGED_SINCE_LAST_SAVE = True
        return Response(status_code=status.HTTP_200_OK)
    except:
        raise HTTPException(status_code=500, detail="Can't calculate global features")

class Item_image_id(BaseModel):
    image_id: int

@app.post("/delete_image_text_features")
async def delete_image_text_features_handler(item:Item_image_id):
    try:
        global DATA_CHANGED_SINCE_LAST_SAVE
        delete_descriptor_by_id(item.image_id)
        res = index.remove_ids(np.int64([item.image_id]))
        if res != 0: 
            DATA_CHANGED_SINCE_LAST_SAVE = True
        else: #nothing to delete
            print(f"err: no image with id {item.image_id}")
        return Response(status_code=status.HTTP_200_OK)
    except:
        raise HTTPException(status_code=500, detail="Can't delete global features")


def periodically_save_index(loop):
    global DATA_CHANGED_SINCE_LAST_SAVE, index
    if DATA_CHANGED_SINCE_LAST_SAVE:
        DATA_CHANGED_SINCE_LAST_SAVE=False
        faiss.write_index(index, "./populated.index")
    loop.call_later(10, periodically_save_index,loop)

main()      



