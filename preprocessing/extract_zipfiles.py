import zipfile as zf 
from pathlib import Path
import os 

SOURCE = Path('source')
LOAD_DIR = Path('data')
CURDIR = Path(".").glob("*")


def extract_zip_to(zip_path, load_path):
    current_zipfile = zf.ZipFile(zip_path)
    for to_extrac_file in current_zipfile.infolist(): 
        current_zipfile.extract(to_extrac_file, 
                                path=load_path)
    current_zipfile.close()
        

writes_path = list(Path(SOURCE).glob("*"))[0]
datasets_path = dict(writes=writes_path)

if LOAD_DIR not in CURDIR:
     os.mkdir(LOAD_DIR)
     for dataset_name, to_unzip_file_path in datasets_path.items():

        if dataset_name == 'writes': 
            extract_zip_to(to_unzip_file_path,
                           load_path=LOAD_DIR / 'writes' 
                           )
            writes_names = list(map(lambda filepath: filepath.__fspath__().split("\\")[-1].split("_"),
                                    (LOAD_DIR / 'writes').glob('*.txt')))
            os.mkdir(LOAD_DIR / 'writes' / 'train')
            os.mkdir(LOAD_DIR / 'writes' / 'test')
            train_filenames = list(filter(lambda filename: filename[1] == 'train.txt',
                                         writes_names))
            test_filenames = list(filter(lambda filename: filename[1] == 'test.txt',
                                         writes_names))
            
            for train_fn, test_fn in zip(train_filenames, test_filenames):
                os.replace(LOAD_DIR / 'writes' / Path("_".join(train_fn)), 
                           LOAD_DIR / 'writes' / 'train' / (train_fn[0] + '.txt'))
                
                os.replace(LOAD_DIR / 'writes' / Path("_".join(test_fn)), 
                           LOAD_DIR / 'writes' / 'test' / (test_fn[0] + '.txt'))
            
