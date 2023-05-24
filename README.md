# image_text_features_web
Faiss + Pytorch + FastAPI + LMDB <br>
Uses CLIP ViT B/16, aQE <br>
Supported operations: add new image, delete image,find similar images by text query, find similar images by image file, find similar images by image id <br>

You should install torch yourself https://pytorch.org/get-started/locally/.  
```bash
pip3 install -r requirements.txt
```

```generate_image_text_features.py ./path_to_img_folder``` -> generates features  
```train.py``` -> trains index  
```add_to_index.py``` -> adds features from lmdb to index  (if ./trained.index is not found, defaults to Flat Index)  
```image_text_features_web.py``` -> web microservice  


Docker:  
in main directory:  
build image -   
```docker build -t qwertyforce/image_text_features_web:1.0.0 --network host -t qwertyforce/image_text_features_web:latest ./```   
run interactively -    
```docker run -ti --rm -p 127.0.0.1:33338:33338  --mount type=bind,source="$(pwd)"/data,target=/app/data --name image_text_features_web qwertyforce/image_text_features_web:1.0.0```   
run as daemon    
```docker run -d --rm -p 127.0.0.1:33338:33338  --mount type=bind,source="$(pwd)"/data,target=/app/data --name image_text_features_web qwertyforce/image_text_features_web:1.0.0```     
