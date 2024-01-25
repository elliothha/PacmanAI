# PacmanAI

A single-agent reinforcement learning project utilizing a convolutionasl dueling Deep Q-Network (DDQN) architecture to play Atari 2600's "Ms. Pacman" in OpenAI's Gym environment. An epsilon-greedy strategy is used for balancing exploration and exploitation in the environment. Hyperparameters used for the networks include:

|   HYPERPARAMETER   | VALUE |
|:------------------:|:-----:|
|    Learning Rate   |  1e-4 |
|     Batch Size     |   32  |
|   Initial Epsilon  |  1.0  |
|    Final Epsilon   |  0.1  |
|    Epsilon Decay   |  0.95 |
|  Update Frequency  |  1e+5 |
|   Discount Factor  |  0.95 |
| Number of Episodes |  1e+6 |
|     Buffer Size    |  1e+6 |

## Environment Description

This project leverages a [Gymnasium](https://gymnasium.farama.org/) environment, which is a maintained fork of OpenAI's [Gym](https://github.com/openai/gym) library. Specifically, The ["ALE/MsPacman-v5"](https://gymnasium.farama.org/environments/atari/ms_pacman/#mspacman) Atari environment is used to provide an environment for the agent to act within. Actions, observation states, and rewards are all taken from the Gymnasium API. The discrete action space of 9 actions is used instead of the full 18-dimensional space typical for Atari environments. For reference, the 9 possible actions are:

| VALUE | MEANING | VALUE |  MEANING  | VALUE |  MEANING |
|:-----:|:-------:|:-----:|:---------:|:-----:|:--------:|
|   0   |   NOOP  |   1   |     UP    |   2   |   RIGHT  |
|   3   |   LEFT  |   4   |    DOWN   |   5   |  UPRIGHT |
|   6   |  UPLEFT |   7   | DOWNRIGHT |   8   | DOWNLEFT |

The observation space is the default "RGB" observation type "Box(0, 255, (210, 160, 3), uint8)", meaning that each returned observation from the environment is an uint8 array of height 210 pixels, width 160 pixels, and 3 channels representing RGB color channels. 

Finally, no additional difficulties or modes are added to the "ALE/MsPacman-v5" environment.

## Building the RL Agent

The RL agent built follows the convolutional Dueling Deep Q-Network (Dueling DQN) architecture from [Wang et. al, 2016](https://arxiv.org/abs/1511.06581). At a high level, this architecture differs from traditional Q-Learning or DQNs by decomposing Q-value estimations into separate Value and Advantage streams as opposed to estimating the Q-values directly. The value stream represents the value of being in a certain state, regardless of the action taken, while the advantage stream provides the extra value of taking a particular action in comparison to other actions in that state. By first normalizing the advantage stream to have zero mean and then adding it to the value stream, this resultant Q-value estimation (analogous to the Bellman equation in traditional Q-Learning) makes it such that the network can learn the value of each state without having to learn the effect of each action in each state. Ideally, this provides more nuanced learning about the environment space, leading to improved policy evaluation, especially in environments with many similar-valued actions or where the value of the state significantly impacts the overall return.

### Preprocessing Inputs

In essence, the Dueling DQN architecture is used to provide Q-value estimations for state-action pairs. And as such, the input to the Dueling DQN will be an observation "state" taken from the environment. Here, a state is defined to be a batched Tensor of shape [batch_size, 4, 84, 84]. As mentioned in the environment description, a single observation frame is a (210, 160, 3) RGB array. Each frame is resized to be (84, 84) from (210, 160), then converted to grayscale to reduce dimensionality as color is not an integral part of playing this specific game, and finally normalized. Four consecutive observation frames are stacked together in order to give the architecture a sense of what is happening in temporal space. Further frame stacking techniques like frame skipping have not been implemented.

### Architecture

The input to the Dueling DQN architecture is sent through three convolutional layers, with ReLU activation functions following each convolution. The [batch_size, 4, 84, 84] input Tensor is convoluted into [batch_size, 32, 20, 20], then [batch_size, 64, 9, 9], then [batch_size, 64, 7, 7]. 

After the last convolution, this feature Tensor is flattened into a [batch_size, 64*7*7 = 3136] Tensor and it is here that the separation into the value and advantage stream occurs. 

With the latent dimension hyperparameter being 512, the value stream has N = 1 and is transformed into a Tensor of shape [batch_size, 1], representing a single value scalar, V(s) for each of the batched inputs. This scalar, V(s), is the expected return in the state, s, under the optimal policy. 

The advantage strean has N = env.action_space.n = 9 and is transformed into a Tensor of shape [batch_size, 9], representing the advantage stream, A(s,a), which is a batched Tensor indicating how much more (or less) beneficial each action is compared to a baseline, which in this case is the value of the state itself. A(s,a) can thus be interpreted as a measure of the relative importance of each action given the current state. 

|   LAYER   | IN CHANNELS | OUT CHANNELS | KERNEL SIZE | STRIDE | PADDING |
|:---------:|:-----------:|:------------:|:-----------:|:------:|:-------:|
| Conv2D #1 |      4      |      32      |      8      |    4   |    1    |
|    ReLU   |             |              |             |        |         |
| Conv2D #2 |      32     |      64      |      4      |    2   |    0    |
|    ReLU   |             |              |             |        |         |
| Conv2D #3 |      64     |      64      |      3      |    1   |    0    |
|    ReLU   |             |              |             |        |         |
| Linear #1 |     3136    |      512     |             |        |         |
|    ReLU   |             |              |             |        |         |
| Linear #2 |     512     |       N      |             |        |         |

Finally, the Q-value estimations are calculated by simply adding V(s) and A(s,a) together, and then returned. 

In summary, the Dueling DQN architecture takes a temporal observation state input, and outputs a Tensor with the Q-value estimations for each action in the environment space.

### Training

Our objective is to learn the weights for the Dueling DQN network such that the agent utilizing this network for Q-values chooses optimal Q-values for play. A primary/target network model is used in training for stability, with the weights of the primary network being copied over to the training network every N episodes. 

For training, experiences are sampled from a pre-populated replay buffer, which is filled with episodes of entirely random exploration. The epsilon-greedy strategy is used to select actions to explore the environment space, with epsilon linearly decaying over the number of episodes the agent is being trained for. 

The loss function upon which we optimize our network is the Mean Squared Error loss between the primary network's returned Q-value predictions, and the "target Q-values", which we define to be the immediate rewards returned by the action plus the target network's prediction for the maximal Q-value of the **next** state multiplied by a discount factor, gamma. The target Q-values can be intuitively thought of as the immediate benefit of taking action, a, in state, s, (the reward), added to the "best possible action-state pair that leads to the maximal future reward", multiplied by a discount factor. This formula balances immediate benefits with potential future gains, which encourages the agent to make decisions that are beneficial both in the short term and in the long run.

## Authors

* **Elliot Ha** - [LinkedIn](https://www.linkedin.com/in/elliothha/) | [GitHub](https://github.com/elliothha)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
