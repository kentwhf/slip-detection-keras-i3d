import numpy as np
import keras
import glob


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, list_IDs, labels, batch_size=1, dim=(64, 224, 224), n_channels=3,
                 n_classes=2, shuffle=True):
        """Initialization"""
        self.dim = dim
        self.ndim = 3
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        list_labels_temp = [self.labels[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp, list_labels_temp)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp, list_labels_temp):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        # y = np.empty((self.batch_size), dtype=int)

        # Generate data
        # print(list_IDs_temp)
        # print(self.labels)

        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            path = \
                glob.glob(r'C:\Users\kentw\OneDrive - University of Toronto\PycharmProjects\Fall-Detection-with-CNNs-and-Optical-Flow\MAA\\'
                          + '/*/' + ID + '.npy')[0]
            X[i,] = np.load(path)

        # Store class
        y = np.array(list_labels_temp, dtype=int)

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
