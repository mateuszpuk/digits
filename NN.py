import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import json
from math import exp
FILE = 'weights.json'

class Neural_Network():
	BatchSize = 100
	input_layer = np.array(784)
	first_hidden_layer = np.array(10)
	second_hidden_layer = np.array(10)
	outcome = np.array(10)
	weights1 = np.array((10,784))
	weights2 = np.array((10,10))
	bias1 = np.array((10,1))
	bias2 = np.array((10,1))
	learning_rate = 0.05
	def __init__(self):
		pass

	def get_input(self,image):
		self.input_layer = np.array(image).reshape((784,1))


	def get_random_weights(self):

		self.weights1 = np.random.uniform(-0.5,0.5,(10,784))
		self.weights2 = np.random.uniform(-0.5,0.5,(10,10))
		self.bias1 = np.random.uniform(-0.5,0.5,(10,1))
		self.bias2 = np.random.uniform(-0.5,0.5,(10,1))
	def load_weights(self,file):
		with open(file,'r') as f:
			data = json.load(f)

			self.bias1 = np.array(data['bias1'])
			self.bias2 = np.array(data['bias2'])
			self.weights1 = np.array(data['weights1'])
			self.weights2 = np.array(data['weights2'])

	def Relu(self,array:np.array):
		"""returns relu applied to numpy array"""

		return np.maximum(array,0)

	def derivRelu(self,array:np.array):

		return array>0

	def Softmax(self,array:np.array):
		x = np.exp(array)

		y = np.sum(np.exp(array))

		return x/y

	def forward_propagation(self):

		Z1 = self.weights1.dot(self.input_layer) + self.bias1
		self.first_hidden_layer = self.Relu(Z1)


		Z2 = self.weights2.dot(self.first_hidden_layer) + self.bias2
		self.second_hidden_layer = self.Relu(Z2)

	def backward(self, correct_images):
		dZ2 = self.second_hidden_layer - (1 / self.BatchSize) * np.sum(correct_images, axis=0, keepdims=True)
		dW2 = np.dot(dZ2, self.first_hidden_layer.T) / self.BatchSize
		dB2 = np.sum(dZ2, axis=1, keepdims=True) / self.BatchSize

		dZ1 = np.dot(self.weights2.T, dZ2) * self.derivRelu(self.first_hidden_layer)
		dW1 = np.dot(dZ1, self.input_layer.T) / self.BatchSize
		dB1 = np.sum(dZ1, axis=1, keepdims=True) / self.BatchSize

		return dW2, dB2, dW1, dB1


	def upadate_parameters(self,correct_images):

		dw2,db2,dw1,db1 = self.backward(correct_images)
		self.weights1 = self.weights1 - self.learning_rate*dw1
		self.weights2 = self.weights2 - self.learning_rate*dw2
		self.bias1 = self.bias1 - self.learning_rate*db1
		self.bias2 = self.bias2 - self.learning_rate*db2
	def prediction(self,image):
		"""makes prediction when given an image"""
		self.get_input(np.array(image))
		self.forward_propagation()
		return self.Softmax(self.second_hidden_layer)

	def save_weights(self,file):
		with open(file,'w') as f:
			data = {}

			data['bias1'] = self.bias1.tolist()
			data['bias2'] = self.bias2.tolist()
			data['weights1'] = self.weights1.tolist()
			data['weights2'] = self.weights2.tolist()
			json.dump(data,f)
	def show_image(self):
		image = self.input_layer
		image = image.reshape((28,28))*255
		plt.gray()
		plt.imshow(image,interpolation='nearest')
		plt.show()
	def loss_fuction(self,first_list,target):
		ret = []

		for i in range(len(first_list)):
			ret.append((first_list[i]-target)**2)
		return np.array(ret)
	def learn(self,values,images):

		for epoch in range(self.BatchSize):
			total_loss = 0

			for image, target in zip(images, values):
				# Forward propagation
				self.get_input(image)
				self.forward_propagation()

				# Compute loss (you can use a suitable loss function here)
				loss = self.loss_fuction(self.second_hidden_layer, target)
				total_loss += loss

				# Backpropagation and parameter updates
				self.upadate_parameters(target)


			# Calculate and print the average loss for the epoch
			avg_loss = total_loss / len(images)
			print(f"Epoch {epoch + 1}/{self.BatchSize}, Loss: {avg_loss}")
			self.save_weights(FILE)




	def pas(self,images,values):
		images2 = []
		values2 = []
		for i in range(images.size):
			images2.append(images[i])
			vals = [0 for x in range(10) ]
			vals[values[i]]=1
			values2.append(np.array(vals))










