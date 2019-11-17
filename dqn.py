import gym
# Tensorflow Library
import tensorflow as tf
import numpy as np
from collections import deque
from time import time, strftime, localtime
import os
tf.get_logger().setLevel('ERROR')

# Define variables for Keras components
ModelBase = tf.keras.Model
layers = tf.keras.layers
regularizers = tf.keras.regularizers

class Dqn(ModelBase):
	def __init__(self):
		super(Dqn, self).__init__()
		self.LIN_N = 128
		self.L1_N = 256
		self.LOUT_N = ACTION_SPACE_SIZE
		# Input shape is the size of state [8]
		self.lIn = layers.Dense(self.LIN_N, activation=tf.nn.relu, name="Lin", activity_regularizer=regularizers.l1_l2(l1=0.1, l2=0.01))
		self.l1 = layers.Dense(self.L1_N, activation=tf.nn.relu, name="L1", activity_regularizer=regularizers.l1_l2(l1=0.1, l2=0.01))
		self.lOut = layers.Dense(self.LOUT_N, name="Lout")

	def shapeStr(self):
		return "%d->%d->%d" % (self.LIN_N, self.L1_N, self.LOUT_N)

	def call(self, x):
		x = self.lIn(x)
		x = self.l1(x)
		output = self.lOut(x)
		return output

# 16: Attempt a much faster epsilon decay
class Lander:
	def __init__(self):
		self.GAMMA = 0.99
		self.EPS_MAX = 1.0
		self.EPS_MIN = 0.01
		self.ADJUSTER = 0.9995
		self.EXPERIENCE_RELAY_SIZE = 300000
		self.BATCH_SIZE = 512
		self.TARGET_NET_UPDATE_FREQ = 1
		self.SAVE_FREQ = 25
		self.WEIGHTS_FILE_NAME = "weights"
		self.DIR_NAME = strftime("%Y-%m-%d-%H-%M-%S", localtime())
		self.LOG_FILE = None

		self.epsilon = self.EPS_MAX
		self.experienceRelay = deque(maxlen=self.EXPERIENCE_RELAY_SIZE)
		self.model = Dqn()
		self.target = Dqn()
		self.loss = tf.keras.losses.MeanSquaredError()
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
		self.trainLoss = tf.keras.metrics.Mean(name='train_loss')
		self._initLogger()

	def _initLogger(self):
		if not os.path.isdir(self.DIR_NAME):
			os.mkdir(self.DIR_NAME)
		self.LOG_FILE = open(os.path.join(self.DIR_NAME, "log.csv"), 'w')
		self.logLine("SHAPE=%s" % self.model.shapeStr())
		self.logLine("Episode, Epsilon, Reward, Loss, LastT")

	def logLine(self, msg):
		if not self.LOG_FILE: print("LOG FILE IS NONE")
		self.LOG_FILE.write("%s\n" % msg)

	def close(self, episode):
		self.LOG_FILE.close()
		self.save(episode)

	def policy(self, state):
		if np.random.sample() > self.epsilon:
			output = self.model(state[None,])
			return np.argmax(output)
		return np.random.choice(ACTION_SPACE_SIZE)

	def save(self, episode):
		if episode == 0 or episode % self.SAVE_FREQ != 0: return
		epsStr = str(episode)
		folder = os.path.join(self.DIR_NAME, epsStr)
		if os.path.isdir(folder):
			print("ALREADY SAVED, SKIPPING SAVE")
		os.mkdir(folder)
		self.model.save_weights(os.path.join(folder, self.WEIGHTS_FILE_NAME))

	def addToExpRelay(self, state, action, reward, nextS, isTerminal):
		self.experienceRelay.append((state, action, reward, nextS, isTerminal))

	def getBatchFromExpRelay(self):
		if len(self.experienceRelay) < self.BATCH_SIZE: return None
		indices = np.arange(len(self.experienceRelay))
		chosenIndices = np.random.choice(indices, self.BATCH_SIZE)
		batch = np.array(list(self.experienceRelay))[chosenIndices]
		bState = batch[:, 0].tolist()
		bAction = batch[:, 1].tolist()
		bReward = batch[:, 2].tolist()
		bNextS = batch[:, 3].tolist()
		bIsTerminal = batch[:, 4].tolist()
		return (bState, bAction, bReward, bNextS, bIsTerminal)

	def updateTargetNet(self, episode):
		# No training done yet
		if not self.target.built: return
		if episode == 0 or episode % self.TARGET_NET_UPDATE_FREQ != 0: return
		self.target.set_weights(self.model.get_weights())

	def updateEpsilon(self):
		self.epsilon = max(self.EPS_MIN, self.epsilon * self.ADJUSTER)

	def getLabels(self, bReward, bNextS, bIsTerminal):
		# Get target Q
		targetPredictions = self.target(np.array(bNextS))
		labels = np.zeros(self.BATCH_SIZE)
		# Calculate target value
		# For the sampled actions, add the reward
		for i in range(self.BATCH_SIZE):
			labels[i] = bReward[i] + (self.GAMMA * np.max(targetPredictions[i]) if not bIsTerminal[i] else 0)
		return labels

	def train(self, bState, bAction, labels):
		with tf.GradientTape() as tape:
			# Get Q
			predictions = self.model(np.array(bState))
			predictions = tf.gather_nd(predictions, tf.stack((tf.range(self.BATCH_SIZE), bAction), axis=1))
			# Take loss
			loss = self.loss(labels, predictions)
		gradients = tape.gradient(loss, self.model.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
		self.trainLoss(loss)

STATE_SPACE_SIZE = 8
ACTION_SPACE_SIZE = 4
MAX_EPISODES = 6000
MAX_EPISODE_LENGTH = 1000 # The simulator will cut an episode off after 999 time steps

if __name__ == "__main__":
	env = gym.make('LunarLander-v2')
	NeilArmstrong = Lander()

	for episode in range(MAX_EPISODES):
		episodeReward = 0
		state = env.reset()
		lastT = 0
		for t in range(MAX_EPISODE_LENGTH):
			action = NeilArmstrong.policy(state)
			(nextS, r, terminal, _) = env.step(action)
			episodeReward += r

			NeilArmstrong.addToExpRelay(state, action, r, nextS, terminal)
			batch = NeilArmstrong.getBatchFromExpRelay()
			if batch:
				(bState, bAction, bReward, bNextS, bIsTerminal) = batch
				NeilArmstrong.train(bState, bAction, NeilArmstrong.getLabels(bReward, bNextS, bIsTerminal))
				NeilArmstrong.updateEpsilon()
			if terminal:
				lastT = t
				break
			state = nextS
		NeilArmstrong.updateTargetNet(episode)
		NeilArmstrong.logLine("%d, %.5f, %.5f, %.5f, %d" % (episode, NeilArmstrong.epsilon, episodeReward, NeilArmstrong.trainLoss.result(), lastT))
		print("Episode %4s, Epsilon %5s, Reward %6s, Loss %5s, T %d" % (episode, str("%.4f" % NeilArmstrong.epsilon), str("%.3f" % episodeReward), str("%.3f" % NeilArmstrong.trainLoss.result()), lastT))
		NeilArmstrong.save(episode)
	NeilArmstrong.close(episode)
	env.close()
