# image_text_features_web
Faiss + Pytorch + FastAPI + LMDB <br>
Uses CLIP ViT B/16, aQE <br>
Supported operations: add new image, delete image,find similar images by text query, find similar images by image file, find similar images by image id <br>

You should install torch yourself https://pytorch.org/get-started/locally/.  
```bash
pip3 install -r requirements.txt
```

```generate_image_text_features.py ./path_to_img_folder``` -> generates features  
```--use_int_filenames_as_id=0``` - images get sequential ids  
```--use_int_filenames_as_id=1``` - image id is parsed from filename ("123.jpg" -> 123)   

```train.py``` -> trains index  
```add_to_index.py``` -> adds features from lmdb to index  (if ./trained.index is not found, defaults to Flat Index)  
```image_text_features_web.py``` -> web microservice  
```GET_FILENAMES=1 image_text_features_web.py``` -> when searching, include filename in search results  
  
Docker (onnx, web inference only):  
  cd to ./onnx_web:  
  download clip_textual.onnx and clip_visual.onnx from [here](https://github.com/qwertyforce/image_text_features_web/releases/tag/clip_models_0)  
  build image -   
  ```docker build -t qwertyforce/onnx_image_text_features_web:1.0.0 --network host -t qwertyforce/onnx_image_text_features_web:latest ./```   
  cd to main directory    
  run interactively -    
  ```docker run -ti --rm -p 127.0.0.1:33338:33338 --network=ambience_net --mount type=bind,source="$(pwd)"/data,target=/app/data --name onnx_image_text_features_web qwertyforce/onnx_image_text_features_web:1.0.0```   
  run as daemon    
  ```docker run -d --rm -p 127.0.0.1:33338:33338 --network=ambience_net --mount type=bind,source="$(pwd)"/data,target=/app/data --name onnx_image_text_features_web qwertyforce/onnx_image_text_features_web:1.0.0```     
  
Docker (w pytorch):  
  in main directory:  
  build image -   
  ```docker build -t qwertyforce/image_text_features_web:1.0.0 --network host -t qwertyforce/image_text_features_web:latest ./```   
  run interactively -    
  ```docker run -ti --rm -p 127.0.0.1:33338:33338 --network=ambience_net --mount type=bind,source="$(pwd)"/data,target=/app/data --name image_text_features_web qwertyforce/image_text_features_web:1.0.0```   
  run as daemon    
  ```docker run -d --rm -p 127.0.0.1:33338:33338 --network=ambience_net --mount type=bind,source="$(pwd)"/data,target=/app/data --name image_text_features_web qwertyforce/image_text_features_web:1.0.0```     
