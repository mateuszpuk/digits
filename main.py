from NN import Neural_Network
import numpy as np
import pandas as pd
FILE = 'weights.json'

images = pd.read_csv('csvTrainImages 60k x 784.csv')
images_values = pd.read_csv('csvTrainLabel 60k x 1.csv')
test_images = pd.read_csv('csvTestImages 10k x 784.csv')
test_values = pd.read_csv('csvTestLabel 10k x 1.csv')

images.head()
images = np.array(images)

images_values.head()
images_values = np.array(images_values)



network = Neural_Network()
network.get_random_weights()



network.learn(images_values,images)


