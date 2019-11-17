import gym
# Tensorflow Library
import tensorflow as tf
import numpy as np
from collections import deque
from time import time, strftime, localtime
import os
tf.get_logger().setLevel('ERROR')
import matplotlib.pyplot as plt

# Define variables for Keras components
ModelBase = tf.keras.Model
layers = tf.keras.layers

class Dqn(ModelBase):
	def __init__(self, lin, l1):
		super(Dqn, self).__init__()
		self.LIN_N = lin
		self.L1_N = l1
		self.LOUT_N = ACTION_SPACE_SIZE
		# Input shape is the size of state [8]
		self.lIn = layers.Dense(self.LIN_N, activation=tf.nn.relu, input_shape=(STATE_SPACE_SIZE,))
		self.l1 = layers.Dense(self.L1_N, activation=tf.nn.relu)
		self.lOut = layers.Dense(self.LOUT_N)

	def shapeStr(self):
		return "%d->%d->%d->%d" % (self.LIN_N, self.L1_N, self.L2_N, self.LOUT_N)

	def call(self, x):
		x = self.lIn(x)
		x = self.l1(x)
		output = self.lOut(x)
		return output

class Lander:
	def __init__(self, loadPath, loadEpisode):
		self.logFile = os.path.join(loadPath,"log.csv")
		(lin, l1) = self.loadNetConfig()

		self.WEIGHTS_FILE_NAME = "weights"
		self.DIR_NAME = loadPath

		self.model = Dqn(lin, l1)
		self._load(loadEpisode)

	def loadNetConfig(self):
		with open(self.logFile, 'r') as logFile:
			l1 = logFile.readline()
			l1 = l1.split("=")[1]
			print("Net Config: %s" % l1)
			l1 = l1.split("->")
			return (l1[0], l1[1])

	def _reshapeState(self, state):
		return tf.reshape(state, [1, STATE_SPACE_SIZE])

	def policy(self, state):
		output = self.model(self._reshapeState(state))
		return np.argmax(output)

	def _load(self, episode):
		try:
			epsStr = str(episode)
			folder = os.path.join(self.DIR_NAME, epsStr)
			fPath = os.path.join(folder, self.WEIGHTS_FILE_NAME)
			print("Attempting %s" % fPath)
			self.model.load_weights(fPath)
			print("Loaded")
		except:
			print("### Failed to load model...")
			quit()

	# The plots for Reward and Time Steps are moving averages.
	def plotCsv(self):
		plt.figure(figsize=(18, 8))
		windowSize = 100
		data = np.genfromtxt(self.logFile, delimiter=',', skip_header=1, names=True)
		plt.subplot(2, 2, 1)
		plt.ylabel('Reward')
		plt.xlabel('Episode')
		y = np.convolve(data['Reward'], np.ones(windowSize), mode='valid') / windowSize
		y = np.insert(y, 0, windowSize * [None])
		plt.plot(np.arange(len(y)), y, 'r-')
		plt.plot(np.arange(len(y) + windowSize), y[windowSize] * np.ones(len(y) + windowSize), 'g:')

		plt.subplot(2, 2, 2)
		plt.ylabel('Time Steps')
		plt.xlabel('Episode')
		# y = data['LastT']
		y = np.convolve(data['LastT'], np.ones(windowSize), mode='valid') / windowSize
		y = np.insert(y, 0, windowSize * [None])
		plt.plot(np.arange(len(y)), y, 'b')
		plt.plot(np.arange(len(y) + windowSize), 100 * np.ones(len(y) + windowSize), 'g:')

		plt.subplot(2, 2, 3)
		plt.ylabel('Loss')
		plt.xlabel('Episode')
		y = data['Loss']
		# y = np.insert(y, 0, windowSize * [None])
		plt.plot(np.arange(len(y)), y, 'm')

		plt.subplot(2, 2, 4)
		plt.ylabel('Epsilon')
		plt.xlabel('Episode')
		y = data['Epsilon']
		# y = np.insert(y, 0, windowSize * [None])
		plt.plot(np.arange(len(y)), y, 'k')

		plt.show()


STATE_SPACE_SIZE = 8
ACTION_SPACE_SIZE = 4
MAX_EPISODES = 15000
MAX_EPISODE_LENGTH = 2000
LOAD_PATH = "2019-11-13-23-56-38"
LOAD_EPISODE = "600"

if __name__ == "__main__":
	env = gym.make('LunarLander-v2')
	NeilArmstrong = Lander(LOAD_PATH, LOAD_EPISODE)
	NeilArmstrong.plotCsv()

	for episode in range(MAX_EPISODES):
		episodeReward = 0
		state = env.reset()
		lastT = 0
		for t in range(MAX_EPISODE_LENGTH):
			env.render()
			action = NeilArmstrong.policy(state)
			(nextS, r, terminal, _) = env.step(action)
			episodeReward += r

			if terminal:
				lastT = t
				break
			state = nextS

		print("Episode %4s, Reward %6s, T %d" % (episode, str("%.3f" % episodeReward), lastT))
	env.close()
