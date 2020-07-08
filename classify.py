import random, cv2, os, sys, shutil
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import keras


os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
# To use CPU for running (can be disabled to make use of gpu)

class image_clustering:

	def __init__(self, folder_path='data/', n_clusters=5, max_examples=None, use_imagenets=False, use_pca=False):
		paths = os.listdir(folder_path)
		self.max_examples = len(paths)
		self.n_clusters = n_clusters
		self.folder_path = folder_path
		random.shuffle(paths)
		self.image_paths = paths[:self.max_examples]
		self.use_imagenets = use_imagenets
		self.use_pca = use_pca
		self.shape = (128,128)
		del paths
		os.makedirs('output')
		for i in range(self.n_clusters):
			os.makedirs('output/cluster' + str(i))
		print('Object of class "image_clustering" has been initialized.')

	def load_images(self):  
		self.images = []
		for image in self.image_paths:
			img = cv2.cvtColor(cv2.resize(cv2.imread(self.folder_path + image), self.shape[::-1]), cv2.COLOR_BGR2RGB) / 255
			self.images.append(img)
		self.images = np.array(self.images)
		print(str(self.max_examples) + ' images from "{}" folder have been loaded in a random order.'.format(self.folder_path))

	def get_new_imagevectors(self): 
		if self.use_imagenets == False:
			self.images_new = self.images
		else:
			imagenet = self.use_imagenets.lower()
			if imagenet == "inceptionv3":
				model1 = keras.applications.inception_v3.InceptionV3(include_top=False, weights='inception_weights.h5', input_shape=(*self.shape,3))
			else:
				print('Use inception for imagenets or False')
				sys.exit()

			pred = model1.predict(self.images)
			images_temp = pred.reshape(self.images.shape[0], -1)
			if self.use_pca == False:
				self.images_new = images_temp
			else:
				model2 = PCA(n_components=None, random_state=728)
				model2.fit(images_temp)
				self.images_new = model2

	def clustering(self):
		model = KMeans(n_clusters=self.n_clusters, n_jobs=-1, random_state=728)
		model.fit(self.images_new)
		predictions = model.predict(self.images_new)

		for i in range(self.max_examples):
			shutil.copy2(self.folder_path + self.image_paths[i], "output/cluster" + str(predictions[i]))

		print('Clustering complete!')
		print('Clusters and the respective images are stored in the "output" folder.')


if __name__ == "__main__":

	# input image is used as 128x128

	number_of_clusters = 5
	# 4 types of shapes and one residue cluster 

	data_path = "Hits/" 

	use_imagenets = "inceptionv3"

	temp = image_clustering(folder_path=data_path, n_clusters=number_of_clusters,use_imagenets=use_imagenets)
	temp.load_images()
	temp.get_new_imagevectors()
	temp.clustering()
