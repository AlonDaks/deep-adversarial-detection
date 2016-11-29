import h5py
import numpy as np

batch_size = 100

num_train_images = 40000
num_test_images = 10000

permutation = np.random.permutation(np.arange(num_train_images))

original_file = h5py.File('/home/ubuntu/storage_volume/data.h5', 'r')
randomized_file = h5py.File('/home/ubuntu/storage_volume/data_randomized.h5', 'w')

randomized_file.create_dataset('X_train', (num_train_images, 3, 224, 224), chunks=True)
randomized_file.create_dataset('labels_train', (num_train_images, 1000), chunks=True)
randomized_file.create_dataset('adversarial_labels_train', (num_train_images, 2), chunks=True)
randomized_file.create_dataset('X_test', (num_test_images, 3, 224, 224), chunks=True)
randomized_file.create_dataset('labels_test', (num_test_images, 1000), chunks=True)
randomized_file.create_dataset('adversarial_labels_test', (num_test_images, 2), chunks=True)

randomized_file['X_test'] = original_file['X_test']
randomized_file.flush()

randomized_file['labels_test'] = original_file['labels_test']
randomized_file.flush()

randomized_file['adversarial_labels_test'] = original_file['adversarial_labels_test']
randomized_file.flush()


for i in np.arange(0, num_train_images, batch_size):
	p = np.sort(permutation[i*batch_size:(i+1)*batch_size])
	randomized_file['X_train'][i*batch_size:(i+1)*batch_size, :,:,:] = original_file['X_train'][p, :,:,:]
	randomized_file['labels_train'][i*batch_size:(i+1)*batch_size,:] = original_file['labels_train'][p,:]
	randomized_file['adversarial_labels_train'][i*batch_size:(i+1)*batch_size,:] = original_file['adversarial_labels_train'][p,:]
	randomized_file.flush()

original_file.close()
randomized_file.flush()
randomized_file.close()
