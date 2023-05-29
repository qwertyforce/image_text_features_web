from PIL import Image
import numpy as np

def Normalize_np(mean,std):
    mean=np.array(mean)
    std=np.array(std)
    mean = mean.reshape(-1, 1, 1)
    std = std.reshape(-1, 1, 1)
    def normalize(images):
        images-=mean
        images/=std
        return images
    return normalize

def transform_bhwc_float(images):
    new_images = images.transpose(0, 3, 1, 2).astype(np.float32) #BHWC -> BCHW
    new_images/=255.0
    return new_images

normalize_clip = Normalize_np((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))


def transform(image):
    image=image.resize((224,224),Image.Resampling.LANCZOS)
    image=np.array(image)[np.newaxis,:]
    image = transform_bhwc_float(image)
    image = normalize_clip(image)
    
    return image
