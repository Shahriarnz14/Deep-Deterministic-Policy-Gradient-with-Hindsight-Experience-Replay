import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 400


def create_critic_network(state_size, action_size, learning_rate):
    """Creates a critic network.

    Args:
        state_size: (int) size of the input.
        action_size: (int) size of the action.
        learning_rate: (float) learning rate for the critic.
    Returns:
        model: an instance of tf.keras.Model.
        state_input: a tf.placeholder for the batched state.
        action_input: a tf.placeholder for the batched action.
    """
    state_input = Input(shape=[state_size])
    state_hidden_1 = Dense(HIDDEN1_UNITS, activation='relu')(state_input)

    action_input = Input(shape=[action_size])
    action_hidden_2 = Dense(HIDDEN2_UNITS, activation='linear')(action_input)
    state_hidden_2 = Dense(HIDDEN2_UNITS, activation='linear')(state_hidden_1)

    state_action_layer = Concatenate()([state_hidden_2, action_hidden_2])
    hidden_layer_2 = Dense(HIDDEN2_UNITS, activation='relu')(state_action_layer)
    value = Dense(1, activation='linear')(hidden_layer_2)

    model = tf.keras.Model(inputs=[state_input, action_input], outputs=value)
    model.compile(loss="mse", optimizer=Adam(lr=learning_rate))
    return model, state_input, action_input


class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, batch_size,
                 tau, learning_rate):
        """Initialize the CriticNetwork.
        This class internally stores both the critic and the target critic
        nets. It also handles computation of the gradients and target updates.

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

        self.model, self.state_input, self.action_input = create_critic_network(self.state_size, self.action_size,
                                                                                learning_rate=self.learning_rate)
        self.target_model, self.target_state_input, self.target_action_input = \
            create_critic_network(self.state_size, self.action_size, learning_rate=self.learning_rate)
        self.gradients_action = tf.gradients(self.model.output, self.action_input)
        self.sess.run(tf.initialize_all_variables())

    def gradients(self, states, actions):
        """Computes dQ(s, a) / da.

        Args:
            states: a batched numpy array storing the state.
            actions: a batched numpy array storing the actions.
        Returns:
            grads: a batched numpy array storing the gradients.
        """
        return self.sess.run(self.gradients_action, feed_dict={self.state_input: states, self.action_input: actions})[0]

    def update_target(self):
        """Updates the target net using an update rate of tau."""
        weights_target_model = self.target_model.get_weights()
        weights_model = self.model.get_weights()

        for weight_idx in range(len(weights_model)):
            weights_target_model[weight_idx] = self.tau * weights_model[weight_idx] + \
                                               (1 - self.tau) * weights_target_model[weight_idx]
        self.target_model.set_weights(weights_target_model)
