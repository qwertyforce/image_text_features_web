from PIL import Image
from torch.utils.data import Dataset

from modules.transform_ops import transform

def read_img_file(f):
    img = Image.open(f)
    # img=img.convert('L').convert('RGB') #GREYSCALE
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img

class InferenceDataset(Dataset):
        def __init__(self, images, IMAGE_PATH):
            self.images = images
            self.IMAGE_PATH = IMAGE_PATH

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            file_name = self.images[idx]
            img_path = self.IMAGE_PATH+"/"+file_name
            try:
                img = read_img_file(img_path)
                img = transform(img)
                return (file_name, img)
            except Exception as e:
                print(e)
                print(f"error reading {img_path}")

def collate_wrapper(batch):
    batch = [el for el in batch if el] #remove None
    if len(batch) == 0:
        return [],[],[]
    file_names, images = zip(*batch)
    return file_names, images
    

if __name__ == '__main__': #entry point 
    import torch
    from os import listdir
    from tqdm import tqdm
    import argparse
    torch.multiprocessing.set_start_method('spawn') # to avoid problems when trying to fork process where torch is imported (CUDA problems)
    
    from modules.inference_ops import get_image_features, get_device
    from modules.lmdb_ops import get_dbs
    from modules.byte_ops import int_from_bytes, int_to_bytes 

    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str,nargs='?', default="./../test_images")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=torch.multiprocessing.cpu_count())
    parser.add_argument('--prefetch_factor', type=int, default=4)
    parser.add_argument('--use_int_filenames_as_id',choices=[0,1], type=int, default=0)

    args = parser.parse_args()
    IMAGE_PATH = args.image_path
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    PREFETCH_FACTOR = args.prefetch_factor  
    USE_INT_FILENAMES = args.use_int_filenames_as_id

    DB_features, DB_filename_to_id, DB_id_to_filename = get_dbs()
    device = get_device()

    if USE_INT_FILENAMES == 0:
        with DB_id_to_filename.begin(buffers=True) as txn:
            with txn.cursor() as curs:
                curs.last()
                x = curs.item()
                SEQUENTIAL_GLOBAL_ID = int_from_bytes(x[0]) # zeros if id_to_filename.lmdb is empty
        SEQUENTIAL_GLOBAL_ID+=1

    def check_if_exists_by_file_name(file_name):
        if USE_INT_FILENAMES:
            image_id = int(file_name[:file_name.index('.')])
            image_id = int_to_bytes(image_id)
        else:
            with DB_filename_to_id.begin(buffers=True) as txn:
                image_id = txn.get(file_name.encode(), default=False)
                if not image_id:
                    return False
        with DB_features.begin(buffers=True) as txn:
            x = txn.get(image_id, default=False)
            if x:
                return True
            return False

    file_names = listdir(IMAGE_PATH)
    print(f"images in {IMAGE_PATH} = {len(file_names)}")

    new_images = []
    for file_name in tqdm(file_names):
        if check_if_exists_by_file_name(file_name):
            continue
        new_images.append(file_name)

    print(f"new images = {len(new_images)}")

    infer_images = InferenceDataset(new_images,IMAGE_PATH)
    dataloader = torch.utils.data.DataLoader(infer_images, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR, collate_fn=collate_wrapper)

    for file_names, images in tqdm(dataloader):
        file_names = list(file_names)
        if len(file_names) == 0:
            continue
        images = torch.stack(images).to(device)
        features = [feature.tobytes() for feature in get_image_features(images)]

        file_name_to_id = []
        id_to_file_name = []   
        for i in range(len(file_names)):
            if USE_INT_FILENAMES:
                idx_of_dot = file_names[i].index('.')
                image_id = int_to_bytes(int(file_names[i][:idx_of_dot]))
            else:
                image_id = int_to_bytes(SEQUENTIAL_GLOBAL_ID)
                SEQUENTIAL_GLOBAL_ID+=1

            file_name_encoded = file_names[i].encode()
            file_name_to_id.append((file_name_encoded, image_id))
            id_to_file_name.append((image_id, file_name_encoded))
            file_names[i] = image_id # FILE_NAMES ARE NOW IMAGE IDS

        with DB_filename_to_id.begin(write=True, buffers=True) as txn:
            with txn.cursor() as curs:
                curs.putmulti(file_name_to_id)

        with DB_id_to_filename.begin(write=True, buffers=True) as txn:
            with txn.cursor() as curs:
                curs.putmulti(id_to_file_name)
                
        print("pushing data to db")
        with DB_features.begin(write=True, buffers=True) as txn:
            with txn.cursor() as curs:
                curs.putmulti(list(zip(file_names, features)))
        del file_names
        del images
        del features