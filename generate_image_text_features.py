from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

def int_to_bytes(x: int) -> bytes:
    return x.to_bytes((x.bit_length() + 7) // 8, 'big')

_transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

def read_img_file(f):
    img = Image.open(f)
    # img=img.convert('L').convert('RGB') #GREYSCALE
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img

def transform2(image):
    desired_size = 224
    old_size = image.size  # old_size[0] is in (width, height) format
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    image = image.resize(new_size, Image.Resampling.LANCZOS)
    new_img = Image.new("RGB", (desired_size, desired_size))
    new_img.paste(image, ((desired_size-new_size[0])//2, (desired_size-new_size[1])//2))
    return _transform(new_img)

class InferenceDataset(Dataset):
        def __init__(self, images, IMAGE_PATH):
            self.images = images
            self.IMAGE_PATH = IMAGE_PATH

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            file_name = self.images[idx]
            image_id = int(file_name[:file_name.index('.')])
            img_path = self.IMAGE_PATH+"/"+file_name
            try:
                img = read_img_file(img_path)
                img = transform2(img)
                return (int_to_bytes(image_id), img)
            except Exception as e:
                print(e)
                print(f"error reading {img_path}")

def collate_wrapper(batch):
    batch = [el for el in batch if el] #remove None
    if len(batch) == 0:
        return [],[],[]
    ids, images = zip(*batch)
    return ids, images
    

if __name__ == '__main__': #entry point 
    import torch
    import clip
    from os import listdir
    from tqdm import tqdm
    import lmdb
    import argparse
    import numpy as np
    torch.multiprocessing.set_start_method('spawn') # to avoid problems when trying to fork process where torch is imported (CUDA problems)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str,nargs='?', default="./../test_images")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=torch.multiprocessing.cpu_count())
    parser.add_argument('--prefetch_factor', type=int, default=4)

    args = parser.parse_args()
    IMAGE_PATH = args.image_path
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    PREFETCH_FACTOR = args.prefetch_factor

    DB = lmdb.open('./features.lmdb',map_size=500*1_000_000) #500mb
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using {device}")
    model, _ = clip.load("ViT-B/16", device=device)
    # model = torch.jit.optimize_for_inference(torch.jit.script(model.eval()))
    model.eval()
    model.to(device)

    def check_if_exists_by_id(id):
        with DB.begin(buffers=True) as txn:
            x = txn.get(int_to_bytes(id),default=False)
            if x:
                return True
            return False

    def get_features(images):
        with torch.no_grad():
            feature_vector = model.encode_image(images)
            feature_vector/=torch.linalg.norm(feature_vector,axis=1).reshape(-1,1)
        feature_vector = feature_vector.cpu().numpy().astype(np.float32)
        return feature_vector
        
    file_names = listdir(IMAGE_PATH)
    print(f"images in {IMAGE_PATH} = {len(file_names)}")
    new_images = []
    for file_name in tqdm(file_names):
        file_id = int(file_name[:file_name.index('.')])
        if check_if_exists_by_id(file_id):
            continue
        new_images.append(file_name)
    print(f"new images = {len(new_images)}")

    infer_images = InferenceDataset(new_images,IMAGE_PATH)
    dataloader = torch.utils.data.DataLoader(infer_images, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR, collate_fn=collate_wrapper)
    for ids, images in tqdm(dataloader):
        if len(ids) == 0:
            continue
        images = torch.stack(images).to(device)
        features = [feature.tobytes() for feature in get_features(images)]
        print("pushing data to db")
        with DB.begin(write=True, buffers=True) as txn:
            with txn.cursor() as curs:
                curs.putmulti(list(zip(ids,features)))
