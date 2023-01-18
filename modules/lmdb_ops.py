import lmdb
DB_features = lmdb.open('./features.lmdb',map_size=5000*1_000_000) #5000mb
DB_filename_to_id = lmdb.open('./filename_to_id.lmdb',map_size=50*1_000_000) #50mb
DB_id_to_filename = lmdb.open('./id_to_filename.lmdb',map_size=50*1_000_000) #50mb

def get_dbs():
    return DB_features, DB_filename_to_id, DB_id_to_filename
