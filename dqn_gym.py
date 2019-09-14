import tensorflow as tf
import numpy as np
import gym
import os
import time


reward_decay_rate = 0
learning_rate = 0.0001
num_iteration = 1000000
animate = False
ckpt_dir = './checkpoint_dqn/'

tf.set_random_seed(0)
np.random.seed(0)
env = gym.make('CartPole-v0')

observation = tf.placeholder(tf.float32, shape=[None]+list(env.observation_space.shape))
action = tf.placeholder(tf.uint8, shape=[None])
reward = tf.placeholder(tf.float32, shape=[None])
accumulated_reward = tf.placeholder(tf.float32, shape=[None])
q_next_state = tf.placeholder(tf.float32, shape=[None, env.action_space.n])


def accumulate_reward(reward_input):
    accumulated_r = reward_input.copy()
    accumulated_r[-1] = reward_input[-1]
    for i in range(len(accumulated_r)-2, -1, -1):
        accumulated_r[i] = reward_input[i] + (1 - reward_decay_rate) * accumulated_r[i+1]
    return accumulated_r


def choose_action(action_q):

    return np.argmax(action_q)

    def softmax(x):
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x)
        return softmax_x

    action_q_softmax = softmax(action_q)
    final_action = np.random.choice(env.action_space.n, 1, p=action_q_softmax)
    return final_action[0]


def q_network(scope):

    with tf.variable_scope(scope, reuse=False):
        net_q = tf.layers.dense(observation, 16)
        net_q = tf.layers.dense(net_q, 32)
        q = tf.layers.dense(net_q, env.action_space.n)

    return q


def train():

    q_training = q_network('training')
    q_target = q_network('target')

    q_action = tf.reduce_sum(tf.multiply(q_training, tf.one_hot(action, env.action_space.n)), axis = 1)
    max_q_next_state = tf.reduce_max(q_next_state)
    q_training_label = tf.add(reward, (1-reward_decay_rate) * max_q_next_state)

    loss_q = tf.losses.mean_squared_error(q_action, q_training_label)
    tvars = tf.trainable_variables(scope='training')
    grads_q, _ = tf.clip_by_global_norm(tf.gradients(loss_q, tvars), 1)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).apply_gradients(zip(grads_q, tvars))

    train_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='training')
    target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target')
    update_target_op = []
    for var, var_target in zip(sorted(train_vars, key=lambda v: v.name),
                               sorted(target_vars, key=lambda v: v.name)):
        update_target_op.append(var_target.assign(var))
    update_target_op = tf.group(*update_target_op)

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if not ckpt:
        print(); print("checkpoint not found."); print()
    sess.run(tf.global_variables_initializer())
    if ckpt:
        saver.restore(sess, ckpt.model_checkpoint_path)

    for iteration in range(num_iteration):

        observation_input = []
        action_input = []
        reward_input = []
        q_next_state_input = []

        o_input = env.reset()
        done = False
        num_step = 0

        while not done:
            if animate:
                env.render()
                time.sleep(0.05)

            observation_input.append(o_input)

            q_training_output = \
                sess.run(q_training,
                         feed_dict={observation: [o_input]})
            q_action_output = choose_action(q_training_output[0])
            action_input.append(q_action_output)

            o_input, r, done, _ = env.step(q_action_output)
            reward_input.append(r)

            num_step += 1

            if done:
                q_target_output = np.zeros(env.action_space.n)
                q_next_state_input.append(q_target_output)
                break
            else:
                # q_target_output = \
                #     sess.run(q_training,
                #              feed_dict={observation: [o_input]})
                q_target_output = \
                    sess.run(q_target,
                             feed_dict={observation: [o_input]})
                q_next_state_input.append(q_target_output[0])

        accumulated_reward_input = accumulate_reward(reward_input)

        _, loss_q_output = \
            sess.run([optimizer, loss_q],
                     feed_dict={observation: observation_input,
                                action: action_input,
                                reward: reward_input,
                                accumulated_reward: accumulated_reward_input,
                                q_next_state: q_next_state_input})

        if iteration % 10 == 0:
            print('iteration:', iteration, 'q_loss:', str(loss_q_output)[:6], 'episode length:', num_step)

        if iteration % 5 == 0:
            sess.run(update_target_op)

        if iteration % 100 == 0:
            if not os.path.exists(ckpt_dir):
                os.mkdir(ckpt_dir)
            checkpoint_path = os.path.join(ckpt_dir, "model.ckpt")
            filename = saver.save(sess, checkpoint_path)
            # print("Model saved in file: %s" % filename)


if __name__ == '__main__':
    train()