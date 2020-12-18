import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from .ReplayBuffer import ReplayBuffer
from .ActorNetwork import ActorNetwork
from .CriticNetwork import CriticNetwork

BUFFER_SIZE = 1000000
BATCH_SIZE = 1024
GAMMA = 0.98                    # Discount for rewards.
TAU = 0.05                      # Target network update rate.
LEARNING_RATE_ACTOR = 0.0001
LEARNING_RATE_CRITIC = 0.001


class EpsilonNormalActionNoise(object):
    """A class for adding noise to the actions for exploration."""

    def __init__(self, mu, sigma, epsilon):
        """Initialize the class.

        Args:
            mu: (float) mean of the noise (probably 0).
            sigma: (float) std dev of the noise.
            epsilon: (float) probability in range [0, 1] with
            which to add noise.
        """
        self.mu = mu
        self.sigma = sigma
        self.epsilon = epsilon

    def __call__(self, action):
        """With probability epsilon, adds random noise to the action.
        Args:
            action: a batched tensor storing the action.
        Returns:
            noisy_action: a batched tensor storing the action.
        """
        if np.random.uniform() > self.epsilon:
            return action + np.random.normal(self.mu, self.sigma)
        else:
            return np.random.uniform(-1.0, 1.0, size=action.shape)


# noinspection DuplicatedCode
class DDPG(object):
    """A class for running the DDPG algorithm."""

    def __init__(self, env, outfile_name):
        """Initialize the DDPG object.

        Args:
            env: an instance of gym.Env on which we aim to learn a policy.
            outfile_name: (str) name of the output filename.
        """
        action_dim = len(env.action_space.low)
        state_dim = len(env.observation_space.low)
        np.random.seed(1337)
        self.env = env

        gpu_ops = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_ops)
        self.sess = tf.Session(config=config)

        # self.sess = tf.Session()
        tf.keras.backend.set_session(self.sess)

        self.actor = ActorNetwork(sess=self.sess, state_size=state_dim, action_size=action_dim,
                                  batch_size=BATCH_SIZE, tau=TAU, learning_rate=LEARNING_RATE_ACTOR)
        self.critic = CriticNetwork(sess=self.sess, state_size=state_dim, action_size=action_dim,
                                    batch_size=BATCH_SIZE, tau=TAU, learning_rate=LEARNING_RATE_CRITIC)

        self.replay_memory = ReplayBuffer(buffer_size=BUFFER_SIZE)

        self.epsilon = 0.2
        self.mu = 0
        self.sigma = 0.01*2

        self.exploration = EpsilonNormalActionNoise(mu=self.mu, sigma=self.sigma, epsilon=self.epsilon)

        self.test_mean_reward = []
        self.test_sigma_reward = []
        self.test_success_ratio = []
        self.test_TD_error = []
        self.outfile_name = outfile_name

    def evaluate(self, num_episodes):
        """Evaluate the policy. Noise is not added during evaluation.

        Args:
            num_episodes: (int) number of evaluation episodes.
        Returns:
            success_rate: (float) fraction of episodes that were successful.
            average_return: (float) Average cumulative return.
        """
        test_rewards = []
        success_vec = []
        plt.figure(figsize=(12, 12))
        for i in range(num_episodes):
            s_vec = []
            state = self.env.reset()
            s_t = np.array(state)
            total_reward = 0.0
            done = False
            step = 0
            success = False
            while not done:
                s_vec.append(s_t)
                a_t = self.actor.model.predict(s_t[None])[0]
                new_s, r_t, done, info = self.env.step(a_t)
                if done and "goal" in info["done"]:
                    success = True
                new_s = np.array(new_s)
                total_reward += r_t
                s_t = new_s
                step += 1
            success_vec.append(success)
            test_rewards.append(total_reward)
            if i < 9:
                plt.subplot(3, 3, i+1)
                s_vec = np.array(s_vec)
                pusher_vec = s_vec[:, :2]
                puck_vec = s_vec[:, 2:4]
                goal_vec = s_vec[:, 4:]
                plt.plot(pusher_vec[:, 0], pusher_vec[:, 1], '-o', label='pusher')
                plt.plot(puck_vec[:, 0], puck_vec[:, 1], '-o', label='puck')
                plt.plot(goal_vec[:, 0], goal_vec[:, 1], '*', label='goal', markersize=10)
                plt.plot([0, 5, 5, 0, 0], [0, 0, 5, 5, 0], 'k-', linewidth=3)
                plt.fill_between([-1, 6], [-1, -1], [6, 6], alpha=0.1,
                                 color='g' if success else 'r')
                plt.xlim([-1, 6])
                plt.ylim([-1, 6])
                if i == 0:
                    plt.legend(loc='lower left', fontsize=28, ncol=3, bbox_to_anchor=(0.1, 1.0))
                if i == 8:
                    # Comment out the line below to disable plotting.
                    plt.savefig('BoxPos.png')
                    # plt.show()
        return np.mean(success_vec), np.mean(test_rewards), np.std(test_rewards)

    def train(self, num_episodes, hindsight=False):
        """Runs the DDPG algorithm.

        Args:
            num_episodes: (int) Number of training episodes.
            hindsight: (bool) Whether to use HER.
        """

        if hindsight:
            test_reward_figure_filename = 'Reward-HER.png'
            # hindsight = False
        else:
            test_reward_figure_filename = 'Reward-DDPG.png'

        for i in range(num_episodes):
            state = self.env.reset()
            s_t = np.array(state)
            total_reward = 0.0
            done = False
            step = 0
            loss = 0
            store_states = []
            store_actions = []
            success = False

            store_rewards = []
            store_new_states = []
            store_done_vec = []

            while not done:
                # Collect one episode of experience, saving the states and actions
                # to store_states and store_actions, respectively.
                store_states.append(s_t.copy())

                a_t = self.exploration.__call__(action=self.actor.model.predict(s_t[None])[0])

                store_actions.append(a_t)
                new_s, r_t, done, info = self.env.step(a_t)
                if done and "goal" in info["done"]:
                    success = True
                new_s = np.array(new_s)

                total_reward += r_t

                self.replay_memory.add(s_t, a_t, r_t, new_s, done)

                store_rewards.append(r_t)
                store_new_states.append(new_s)
                store_done_vec.append(done)

                s_t = new_s
                step += 1

                sampled_experiences = self.replay_memory.get_batch(batch_size=BATCH_SIZE)
                states = np.asarray([experience[0] for experience in sampled_experiences])
                actions = np.asarray([experience[1] for experience in sampled_experiences])
                rewards = np.asarray([experience[2] for experience in sampled_experiences])
                new_states = np.asarray([experience[3] for experience in sampled_experiences])
                done_vec = np.asarray([experience[4] for experience in sampled_experiences])

                y_vec = rewards + (1-done_vec) * GAMMA * self.critic.target_model.predict([new_states, self.actor.target_model.predict(new_states)]).squeeze()

                loss += self.critic.model.train_on_batch([states, actions], y_vec)

                actions_optimize = self.actor.model.predict(states)
                gradients_action = self.critic.gradients(states=states, actions=actions_optimize)
                self.actor.train(states=states, action_grads=gradients_action)

                self.critic.update_target()
                self.actor.update_target()
            # end_while

            if hindsight:
                # For HER, we also want to save the final next_state.
                store_states.append(new_s)
                self.add_hindsight_replay_experience(np.array(store_states), np.array(store_actions))

            del store_states, store_actions
            store_states, store_actions = [], []

            # Logging
            print("Episode %d: Total reward = %d\t,\tTD loss = %.2f\t,\tSteps = %d; Info = %s\t,\tSuccess = %s" %
                  (i, total_reward, loss / step, step, info['done'], success))
            if i % 100 == 0:
                successes, mean_rewards, std_rewards = self.evaluate(10)
                print('Evaluation: success = %.2f; return = %.2f' % (successes, mean_rewards))
                with open(self.outfile_name, "a") as f:
                    f.write("%.2f, %.2f, %.2f\n" % (successes, mean_rewards, std_rewards))

                self.test_mean_reward.append(mean_rewards)
                self.test_sigma_reward.append(std_rewards)
                self.test_success_ratio.append(successes)
                self.test_TD_error.append([loss, step])  # From Last Training

                plt.figure()
                x_plotting = np.linspace(0, i, len(self.test_mean_reward))
                y_plotting = np.asarray(self.test_mean_reward)
                y_error = np.asarray(self.test_sigma_reward)
                plt.plot(x_plotting, y_plotting)
                plt.fill_between(x_plotting,
                                 y_plotting - y_error,
                                 y_plotting + y_error, alpha=0.5)
                plt.xlabel('Number of Training Episodes')
                plt.ylabel('Average Test Reward')
                plt.savefig(test_reward_figure_filename)
                plt.close('all')
                np.save('data_TestRewardMean', y_plotting)
                np.save('data_TestRewardStd', y_error)
                np.save('data_TestSuccessRatio', np.asarray(self.test_success_ratio))
                np.save('data_TestTDError_Train', np.asarray(self.test_TD_error))

    def add_hindsight_replay_experience(self, states, actions):
        """Relabels a trajectory using HER.

        Args:
            states: a list of states.
            actions: a list of states.
        """
        her_states, her_rewards = self.env.apply_hindsight(np.copy(states))
        acts = np.copy(actions)

        for time_step in range(len(actions)):
            s_t = her_states[time_step]
            r_t = her_rewards[time_step]
            a_t = acts[time_step]
            new_s = her_states[time_step + 1]
            done = False if r_t else True

            self.replay_memory.add(s_t, a_t, r_t, new_s, done)
        # end_for
