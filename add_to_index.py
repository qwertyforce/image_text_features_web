from pathlib import Path
from tqdm import tqdm
import numpy as np
import lmdb
import faiss
# import pickle 

DB_features = lmdb.open("features.lmdb", readonly=True)
dim = 512
faiss_dim = dim
if Path("./trained.index").is_file():
    index = faiss.read_index("./trained.index")
else:
    quantizer = faiss.IndexFlat(faiss_dim, faiss.METRIC_L2)
    index = faiss.IndexIDMap2(quantizer)
USE_PCA = False
pca = None
# pca_w_file = Path("./pca_w.pkl")
# if pca_w_file.is_file():
#     with open(pca_w_file, 'rb') as pickle_file:
#         pca = pickle.load(pickle_file)
#         USE_PCA = True
#         print("USING PCA")
# if pca is None:
#     USE_PCA = False
#     print("pca_w.pkl not found. Proceeding without PCA")

def int_from_bytes(xbytes: bytes) -> int:
    return int.from_bytes(xbytes, 'big')
    
def get_all_data_iterator(batch_size=10000):
    with DB_features.begin(buffers=True) as txn:
        with txn.cursor() as curs:
            temp_ids = np.zeros(batch_size,np.int64)
            temp_features = np.zeros((batch_size,dim),np.float32)
            retrieved = 0
            for data in curs.iternext(keys=True, values=True):
                temp_ids[retrieved] = int_from_bytes(data[0])
                temp_features[retrieved] = np.frombuffer(data[1],dtype=np.float32)
                retrieved+=1
                if retrieved == batch_size:
                    retrieved=0
                    if USE_PCA:
                        temp_features = pca.transform(temp_features)
                        for i in range(len(temp_features)):
                            temp_features[i]/=np.linalg.norm(temp_features[i])
                    yield temp_ids, temp_features
            if retrieved != 0: #retrieved is less than batch_size in the end of final iteration
                if USE_PCA:
                    temp_features = temp_features[:retrieved]
                    temp_features = pca.transform(temp_features)
                    for i in range(len(temp_features)):
                        temp_features[i]/=np.linalg.norm(temp_features[i])
                    yield temp_ids[:retrieved], temp_features
                else:
                    yield temp_ids[:retrieved], temp_features[:retrieved]

for ids, features in tqdm(get_all_data_iterator(100_000)):
    index.add_with_ids(features,ids)
faiss.write_index(index,"populated.index")