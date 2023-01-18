import torch
import clip
import numpy as np 

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
def get_device():
    return device

print(f"using {device}")

model, _ = clip.load("ViT-B/16", device=device)
# model = torch.jit.optimize_for_inference(torch.jit.script(model.eval()))
model.eval()
model.to(device)

def get_image_features(images):
    with torch.no_grad():
        feature_vector = model.encode_image(images)
        feature_vector/=torch.linalg.norm(feature_vector,axis=1).reshape(-1,1)
    return feature_vector.cpu().numpy().astype(np.float32)

def get_text_features(text):
    with torch.no_grad():
        text_tokenized = clip.tokenize([text]).to(device)
        feature_vector = model.encode_text(text_tokenized)
        feature_vector/=torch.linalg.norm(feature_vector)
    return feature_vector.cpu().numpy().astype(np.float32)