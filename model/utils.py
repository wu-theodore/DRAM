import gzip
import os
import urllib.request
import shutil

import tensorflow as tf


def make_dir(PATH):
    try:
        os.makedirs(PATH)
    except OSError:
        pass

def download_file(download_url, local_path, 
                expected_bytes=None, 
                unzip=False):
    # check if path already exists:
    if os.path.exists(local_path) or os.path.exists(local_path[:-3]):
        print("Path ({}) already exists".format(local_path[:-3]))
    else:
        print("Downloading file from {}".format(download_url))
        local_file, _ = urllib.request.urlretrieve(download_url, local_path)
        local_stat = os.stat(local_file)

        # Confirm download.
        if expected_bytes:
            if expected_bytes == local_stat.st_size:
                print("Successfully downloaded.")
                if unzip:
                    print("Unzipping:")
                    with gzip.open(local_path, 'rb') as fsrc, open(local_path[:-3], 'wb') as fdst:
                        shutil.copyfileobj(fsrc, fdst)
                    os.remove(local_path)
            else:
                print("Downloaded file has unexpected number of bytes.")
        print("Download process for {} complete.".format(download_url))


def convert_to_dataset(data, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset

def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier
