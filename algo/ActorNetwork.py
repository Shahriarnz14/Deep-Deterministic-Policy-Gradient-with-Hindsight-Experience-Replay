import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 400


def create_actor_network(state_size, action_size):
    """Creates an actor network.

    Args:
        state_size: (int) size of the input.
        action_size: (int) size of the action.
    Returns:
        model: an instance of tf.keras.Model.
        state_input: a tf.placeholder for the batched state.
    """
    state_input = Input(shape=[state_size])

    hidden_layer_1 = Dense(HIDDEN1_UNITS, activation='relu')(state_input)
    hidden_layer_2 = Dense(HIDDEN2_UNITS, activation='relu')(hidden_layer_1)
    output = Dense(action_size, activation='tanh')(hidden_layer_2)

    model = Model(inputs=state_input, outputs=output)
    return model, state_input


class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, batch_size,
                 tau, learning_rate):
        """Initialize the ActorNetwork.
        This class internally stores both the actor and the target actor nets.
        It also handles training the actor and updating the target net.

        Args:
            sess: A Tensorflow session to use.
            state_size: (int) size of the input.
            action_size: (int) size of the action.
            batch_size: (int) the number of elements in each batch.
            tau: (float) the target net update rate.
            learning_rate: (float) learning rate for the critic.
        """
        self.sess = sess

        tf.keras.backend.set_session(self.sess)

        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.tau = tau
        self.learning_rate = learning_rate

        self.model, self.state_input = create_actor_network(self.state_size, self.action_size)
        self.target_model, self.target_state_input = create_actor_network(self.state_size, self.action_size)

        self.gradients_action = tf.placeholder(tf.float32, [None, self.action_size])
        self.gradients_weight = tf.gradients(self.model.output, self.model.trainable_weights, -self.gradients_action)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.gradients_weight,
                                                                                       self.model.trainable_weights))
        self.sess.run(tf.initialize_all_variables())

    def train(self, states, action_grads):
        """Updates the actor by applying dQ(s, a) / da.

        Args:
            states: a batched numpy array storing the state.
            action_grads: a batched numpy array storing the
                gradients dQ(s, a) / da.
        """
        self.sess.run(self.optimize, feed_dict={self.state_input: states, self.gradients_action: action_grads})

    def update_target(self):
        """Updates the target net using an update rate of tau."""
        weights_target_model = self.target_model.get_weights()
        weights_model = self.model.get_weights()

        for weight_idx in range(len(weights_model)):
            weights_target_model[weight_idx] = self.tau * weights_model[weight_idx] + \
                                               (1 - self.tau) * weights_target_model[weight_idx]
        self.target_model.set_weights(weights_target_model)
