import numpy as np

def load_images(file):
    with open(file,"rb") as imfile:
        num_images = int.from_bytes(imfile.read(4),'big')
        height = int.from_bytes(imfile.read(4),'big')
        width = int.from_bytes(imfile.read(4), 'big')
        images = np.frombuffer(imfile.read(num_images*width*height), dtype=np.uint8)
        return images.reshape((num_images,height,width))

def load_labels(file, one_hot=True):
    with open(file,"rb") as lblfile:
        labels = np.frombuffer(lblfile.read(), dtype=np.uint8)
        if one_hot:
            return dense_to_onehot(labels)
        return labels

def dense_to_onehot(dense):
    total_labels = dense.shape[0]
    total_classes = len(set(dense))
    offsets = np.arange(total_labels) * total_classes
    one_hot_vectors = np.zeros((total_labels, total_classes), dtype=np.uint8)
    one_hot_vectors.flat[ offsets + dense ] = 1
    return one_hot_vectors



class Dataset:
    def __init__(self, images, labels):
        self._num_points = images.shape[0]
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch   = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def total_points(self):
        return self._num_points

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def _shuffle(self):
        permutation = np.arange(self._num_points)
        np.random.shuffle(permutation)
        self._images = self._images[permutation]
        self._labels = self._labels[permutation]

    def next_batch(self, batchSize=100, shuffle=True):
        start = self._index_in_epoch
        if(self._epochs_completed == 0 and start == 0 and shuffle):
            self._shuffle()

        if((start + batchSize) > self._num_points):
            self._epochs_completed += 1
            rest_points = self._num_points - start
            rest_images = self._images[start:self._num_points]
            rest_labels = self._labels[start:self._num_points]
            if shuffle:
                self._shuffle()
            start = 0
            top_points = batchSize - rest_points
            self._index_in_epoch = top_points
            top_images = self._images[start: self._index_in_epoch]
            top_labels = self._labels[start: self._index_in_epoch]
            return np.concatenate((rest_images, top_images)), np.concatenate((rest_labels, top_labels))

        else:
            self._index_in_epoch += batchSize
            return self._images[start: self._index_in_epoch] , self.labels[start: self._index_in_epoch]








class cohnKanadeDataset:
    def __init__(self, impackfile, lblpackfile, train=.80):
        images = load_images(impackfile)
        labels = load_labels(lblpackfile)
        if(images.shape[0] != labels.shape[0]):
            raise ValueError('images and labels not consistent')
        num_train = int(train * images.shape[0])
        self.train = Dataset(images[:num_train], labels[:num_train])
        self.test = Dataset(images[num_train:], labels[num_train:])
