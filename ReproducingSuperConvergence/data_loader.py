import sys
import pickle
import os
from six.moves.urllib.request import urlretrieve
import tarfile
import scipy.io
import numpy as np

def savePickle(object, filePath):
    with open(filePath, 'wb') as f:
        pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)

def openPickle(filepath): 
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f, encoding='bytes')
    except:
        return None


def load_data():
    INPUT_ROOT = "../input/"
    CIFAR_ROOT = os.path.join(INPUT_ROOT, "cifar-10-batches-py/")

    pickle_path = os.path.join(INPUT_ROOT, "data.pickle")
    data = openPickle(pickle_path)
    if data != None:
        return data

    gzip_path = os.path.join(INPUT_ROOT, "cifar-10-python.tar.gz")

    #Make sure we've downloaded the gzip
    if not os.path.exists(gzip_path):
        download_data(INPUT_ROOT, "cifar-10-python.tar.gz")

    data_batch_1_path = os.path.join(CIFAR_ROOT, "data_batch_1")
    data_batch_2_path = os.path.join(CIFAR_ROOT, "data_batch_2")
    data_batch_3_path = os.path.join(CIFAR_ROOT, "data_batch_3")
    data_batch_4_path = os.path.join(CIFAR_ROOT, "data_batch_4")
    data_batch_5_path = os.path.join(CIFAR_ROOT, "data_batch_5")
    test_batch_path = os.path.join(CIFAR_ROOT, "test_batch")

    #Make sure we've unzipped the files
    if not os.path.exists(data_batch_1_path) or \
    not os.path.exists(data_batch_2_path) or \
    not os.path.exists(data_batch_3_path) or \
    not os.path.exists(data_batch_4_path) or \
    not os.path.exists(data_batch_5_path) or \
    not os.path.exists(test_batch_path):
        tar = tarfile.open(gzip_path)
        tar.extractall(INPUT_ROOT)
        tar.close()

    data_batch_1 = openPickle(data_batch_1_path)
    data_batch_2 = openPickle(data_batch_2_path)
    data_batch_3 = openPickle(data_batch_3_path)
    data_batch_4 = openPickle(data_batch_4_path)
    data_batch_5 = openPickle(data_batch_5_path)
    test_batch = openPickle(test_batch_path)

    data1 = data_batch_1[b'data']
    labels1 = data_batch_1[b'labels']
    data2 = data_batch_2[b'data']
    labels2 = data_batch_2[b'labels']
    data3 = data_batch_3[b'data']
    labels3 = data_batch_3[b'labels']
    data4 = data_batch_4[b'data']
    labels4 = data_batch_4[b'labels']
    data5 = data_batch_5[b'data']
    labels5 = data_batch_5[b'labels']
    test_data = test_batch[b'data']
    test_labels = np.array(test_batch[b'labels'])

    #Join all datasets into a single one
    dataset = np.append(data1, data2, axis=0)
    dataset = np.append(dataset, data3, axis=0)
    dataset = np.append(dataset, data4, axis=0)
    dataset = np.append(dataset, data5, axis=0)
    
    labels = np.append(labels1, labels2, axis=0)
    labels = np.append(labels, labels3, axis=0)
    labels = np.append(labels, labels4, axis=0)
    labels = np.append(labels, labels5, axis=0)

    #Reshape to 32x32x3 for use in our conv net
    dataset = dataset.reshape(-1, 3, 32, 32)
    dataset = dataset.transpose(0, 2, 3, 1)
    labels = labels.reshape((-1, 1))
    test_data = test_data.reshape(-1, 3, 32, 32)
    test_data = test_data.transpose(0, 2, 3, 1)
    test_labels = test_labels.reshape((-1, 1))

    #Shuffle the dataset
    dataset, labels = randomize(dataset, labels)
    test_data, test_labels = randomize(test_data, test_labels)

    #We create a validation set with an even distribution of each class
    n_labels = 10
    valid_index = np.zeros(labels.shape[0], dtype=bool)
    
    for i in np.arange(n_labels):
        valid_index[(np.where(labels[:,0] == (i))[0][:250].tolist())] = True

    valid_data = dataset[valid_index, :, :, :]
    valid_labels = labels[valid_index]
    train_labels = labels[~valid_index]
    train_data = dataset[~valid_index,:,:,:]

    savePickle((train_data, train_labels, valid_data, valid_labels, test_data, test_labels), pickle_path)
    return (train_data, train_labels, valid_data, valid_labels, test_data, test_labels)


def download_data(dest_folder, filename):
    url = "https://www.cs.toronto.edu/~kriz/" + filename

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    dest_file_path = os.path.join(dest_folder, filename)
    print('Attempting to download:', url) 
    filename, _ = urlretrieve(url, dest_file_path, reporthook=download_progress_hook)
    print('\nDownload Complete!')


last_percent_reported = None
def download_progress_hook(count, blockSize, totalSize):
  """A hook to report the progress of a download. This is mostly intended for users with
  slow internet connections. Reports every 5% change in download progress.
  """
  global last_percent_reported
  percent = int(count * blockSize * 100 / totalSize)
  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()
    last_percent_reported = percent

def randomize(dataset, labels):
    np.random.seed(42) 
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

