import os
import os.path
import errno

################################################################
def create_dir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
            
################################################################
def file_exists(path):
    return os.path.isfile(path)
