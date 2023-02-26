import uvicorn
if __name__ == '__main__':
    uvicorn.run('image_text_features_web:app', host='127.0.0.1', port=33338, log_level="info")
    exit()

from os import environ
if not "GET_FILENAMES" in environ:
    print("GET_FILENAMES not found! Defaulting to 0...")
    GET_FILENAMES = 0
else:
    if environ["GET_FILENAMES"] not in ["0","1"]:
        print("GET_FILENAMES has wrong argument! Defaulting to 0...")
        GET_FILENAMES = 0
    else:
        GET_FILENAMES = int(environ["GET_FILENAMES"])

import traceback
from os.path import exists
from typing import Optional, Union
from pydantic import BaseModel
from fastapi import FastAPI, File,Form, HTTPException, Response, status
import numpy as np
import asyncio
from PIL import Image
# from pathlib import Path
import io
import faiss
# import pickle

from modules.byte_ops import int_to_bytes
from modules.inference_ops import get_image_features, get_text_features, get_device
from modules.transform_ops import transform
from modules.lmdb_ops import get_dbs

dim = 512
device = get_device()

index = None
DATA_CHANGED_SINCE_LAST_SAVE = False
# pca_w_file = Path("./pca_w.pkl")
pca = None
# if pca_w_file.is_file():
#     with open(pca_w_file, 'rb') as pickle_file:
#         pca = pickle.load(pickle_file)
app = FastAPI()

def main():
    global DB_features, DB_filename_to_id, DB_id_to_filename
    init_index()
    DB_features, DB_filename_to_id, DB_id_to_filename = get_dbs()

    init_index()
    loop = asyncio.get_event_loop()
    loop.call_later(10, periodically_save_index,loop)

def check_if_exists_by_image_id(image_id):
    with DB_features.begin(buffers=True) as txn:
        x = txn.get(int_to_bytes(image_id), default=False)
        if x:
            return True
        return False

def get_filenames_bulk(image_ids):
    image_ids_bytes = [int_to_bytes(x) for x in image_ids]

    with DB_id_to_filename.begin(buffers=False) as txn:
        with txn.cursor() as curs:
            file_names = curs.getmulti(image_ids_bytes)
    for i in range(len(file_names)):
        file_names[i] = file_names[i][1].decode()

    return file_names

def delete_descriptor_by_id(image_id):
    image_id_bytes = int_to_bytes(image_id)
    with DB_features.begin(write=True, buffers=True) as txn:
        txn.delete(image_id_bytes)   #True = deleted False = not found

    with DB_id_to_filename.begin(write=True, buffers=True) as txn:
        file_name_bytes = txn.get(image_id_bytes, default=False)
        txn.delete(image_id_bytes)  

    with DB_filename_to_id.begin(write=True, buffers=True) as txn:
        txn.delete(file_name_bytes) 

def add_descriptor(image_id, features):
    file_name_bytes = f"{image_id}.online".encode()
    image_id_bytes = int_to_bytes(image_id)
    with DB_features.begin(write=True, buffers=True) as txn:
        txn.put(image_id_bytes, features.tobytes())

    with DB_id_to_filename.begin(write=True, buffers=True) as txn:
        txn.put(image_id_bytes, file_name_bytes)

    with DB_filename_to_id.begin(write=True, buffers=True) as txn:
        txn.put(file_name_bytes, image_id_bytes)

def read_img_buffer(image_data):
    img = Image.open(io.BytesIO(image_data))
    # img=img.convert('L').convert('RGB') #GREYSCALE
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img

def get_features(image_buffer):
    image=read_img_buffer(image_buffer)
    image = transform(image).unsqueeze(0).to(device)
    feature_vector = get_image_features(image)
    return feature_vector

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
        similar = nn_find_similar(target_features, k, distance_threshold, aqe_n, aqe_alpha)
        if GET_FILENAMES:
            file_names = get_filenames_bulk([el["image_id"] for el in similar])
            for i in range(len(similar)):
                similar[i]["file_name"] = file_names[i]
        return similar
    except:
        traceback.print_exc()
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
        similar = nn_find_similar(target_features, k, distance_threshold, aqe_n, aqe_alpha)
        if GET_FILENAMES:
            file_names = get_filenames_bulk([el["image_id"] for el in similar])
            for i in range(len(similar)):
                similar[i]["file_name"] = file_names[i]
        return similar
    except:
        traceback.print_exc()
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
        similar = nn_find_similar(target_features, k, distance_threshold, aqe_n, aqe_alpha)
        if GET_FILENAMES:
            file_names = get_filenames_bulk([el["image_id"] for el in similar])
            for i in range(len(similar)):
                similar[i]["file_name"] = file_names[i]
        return similar
    except:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Error in image_text_features_get_similar_images_by_text")

@app.post("/calculate_image_text_features")
async def calculate_image_text_features_handler(image: bytes = File(...),image_id: str = Form(...)):
    try:
        global DATA_CHANGED_SINCE_LAST_SAVE
        image_id = int(image_id)
        if check_if_exists_by_image_id(image_id):
            return Response(content="Image with the same id is already in the db", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, media_type="text/plain")
            
        features=get_features(image)
        add_descriptor(image_id, features.astype(np.float32))
        # if pca:
        #     target_features=pca.transform(target_features)
        #     target_features/=np.linalg.norm(target_features)
        index.add_with_ids(features, np.int64([image_id])) # index.add_items(features,[image_id])
        DATA_CHANGED_SINCE_LAST_SAVE = True
        return Response(status_code=status.HTTP_200_OK)
    except:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Can't calculate global features")

class Item_image_id(BaseModel):
    image_id: int

@app.post("/delete_image_text_features")
async def delete_image_text_features_handler(item:Item_image_id):
    try:
        global DATA_CHANGED_SINCE_LAST_SAVE
        res = index.remove_ids(np.int64([item.image_id]))
        if res != 0: 
            delete_descriptor_by_id(item.image_id)
            DATA_CHANGED_SINCE_LAST_SAVE = True
        else: #nothing to delete
            print(f"err: no image with id {item.image_id}")    
        return Response(status_code=status.HTTP_200_OK)
    except:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Can't delete global features")

def init_index():
    global index
    if exists("./populated.index"):
        index = faiss.read_index("./populated.index")
    else:
        print("Index is not found! Exiting...")
        exit()

def periodically_save_index(loop):
    global DATA_CHANGED_SINCE_LAST_SAVE, index
    if DATA_CHANGED_SINCE_LAST_SAVE:
        DATA_CHANGED_SINCE_LAST_SAVE=False
        faiss.write_index(index, "./populated.index")
    loop.call_later(10, periodically_save_index,loop)

main()      



