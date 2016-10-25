import json
from scipy.io import loadmat

keras_indices = json.load(open('imagenet_class_index.json'))
imagenet_indices = loadmat('ILSVRC2012_devkit_t12/data/meta.mat')['synsets']

imagenet_dict = {}
for i in range(len(imagenet_indices)):
	imagenet_dict[i+1] = imagenet_indices[i][0][1][0]

keras_dict = {}
for k in keras_indices:
	v = keras_indices[k]
	keras_dict[v[0]] = k

# with open('ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt') as f:
# 	for x in f:
# 		print(keras_dict[imagenet_dict[int(x)]])
