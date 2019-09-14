import numpy as np
import tensorflow as tf
import gym
import time
import os


learning_rate = 0.0001
num_iteration = 2000000
reward_decay_rate = 0
animate = False
ckpt_dir = './checkpoint_pg/'

tf.set_random_seed(0)
np.random.seed(0)
env = gym.make('CartPole-v0')

observation = tf.placeholder(tf.float32, shape=[None, env.observation_space.shape[0]])
action_label = tf.placeholder(tf.int8, shape=[None, env.action_space.n])
accumulated_advantage = tf.placeholder(tf.float32, shape=[None])
baseline_label = tf.placeholder(tf.float32, shape=[None])


def chooseAction(action_output):
    def softmax(x):
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x)
        return softmax_x
    
    action_output_softmax = softmax(action_output)
    final_action = np.random.choice(env.action_space.n, 1, p=action_output_softmax)
    return final_action[0]


def oneHot(x, depth):
    y = np.zeros(depth)
    y[x] = 1
    return y


def AccumulateReward(reward, gamma):
    reward_length = len(reward)
    final_accumulated_reward = reward.copy()
    accumulated_reward = reward[-1]

    for i in range(reward_length-2, -1, -1):
        final_accumulated_reward[i] += accumulated_reward * gamma
        accumulated_reward = final_accumulated_reward[i]
    
    return final_accumulated_reward


def normalize(x, mu, sigma):
    unit_x = (x - np.mean(x))/(np.std(x) + 1e-8)
    return mu + sigma * unit_x


def build_model():

    net_action = tf.layers.dense(observation, 16)
    net_action = tf.layers.dense(net_action, 32)
    action = tf.layers.dense(net_action, env.action_space.n)

    cross_entropy_action = tf.nn.softmax_cross_entropy_with_logits_v2(labels=action_label, logits=action)
    loss_action = tf.reduce_sum(tf.multiply(cross_entropy_action, accumulated_advantage))

    # baseline is different from Value function
    net_baseline = tf.layers.dense(observation, 16)
    net_baseline = tf.layers.dense(net_baseline, 32)
    baseline = tf.squeeze(tf.layers.dense(net_baseline, 1))

    loss_baseline = tf.reduce_mean(tf.square(baseline - baseline_label))

    tvars = tf.trainable_variables()
    grads_action, _ = tf.clip_by_global_norm(tf.gradients(loss_action, tvars), 1)
    grads_baseline, _ = tf.clip_by_global_norm(tf.gradients(loss_baseline, tvars), 1)
    optimizer_action = tf.train.GradientDescentOptimizer(learning_rate).apply_gradients(zip(grads_action, tvars))
    optimizer_baseline = tf.train.GradientDescentOptimizer(learning_rate).apply_gradients(zip(grads_baseline, tvars))

    return optimizer_action, optimizer_baseline, loss_action, loss_baseline, action, baseline


def train():

    optimizer_action, optimizer_baseline, loss_action, loss_baseline, action, baseline = build_model()

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
        action_label_input = []
        baseline_label_input = []
        accumulated_reward_input = []
        reward = []

        o_input = env.reset()
        done = False

        while not done:
            if animate:
                env.render()
                time.sleep(0.05)

            action_output, baseline_output = \
                sess.run([action, baseline],
                            feed_dict={observation: [o_input]})

            observation_input.append(o_input)
            final_action = chooseAction(action_output[0])
            action_label_input.append(oneHot(final_action, env.action_space.n))
            baseline_label_input.append(baseline_output)

            o_input, r, done, _ = env.step(final_action)
            reward.append(r)

        accumulated_reward_input_episode = AccumulateReward(reward, 1-reward_decay_rate)
        for i in accumulated_reward_input_episode:
            accumulated_reward_input.append(i)

        # # additional norm trick
        # baseline_label_input = normalize(baseline_label_input,
        #                                  np.mean(accumulated_reward_input), np.std(accumulated_reward_input))
        
        # advantage_length = len(baseline_label_input)
        # accumulated_advantage_input = np.zeros(advantage_length)
        # accumulated_advantage_input[-1] = accumulated_reward_input[-1] - baseline_label_input[-1]
        # for i in range(advantage_length-2, -1, -1):
        #     accumulated_advantage_input[i] = \
        #         accumulated_reward_input[i] - accumulated_reward_input[i+1] * (1 - reward_decay_rate) + \
        #             baseline_label_input[i+1] * (1 - reward_decay_rate) - baseline_label_input[i]

        # accumulated_advantage_input = normalize(accumulated_advantage_input, 0, 1)

        # baseline_label_input = normalize(accumulated_reward_input, 0, 1)

        _, _, loss_action_output, loss_baseline_output = \
            sess.run([optimizer_action, optimizer_baseline, loss_action, loss_baseline],
                     feed_dict={observation: observation_input,
                                action_label: action_label_input,
                                accumulated_advantage: accumulated_reward_input,
                                # accumulated_advantage: accumulated_advantage_input,
                                baseline_label: baseline_label_input})
        
        if iteration % 10 == 0:
            print('iteration:', iteration, 'action loss:', str(loss_action_output)[:6],
                #   'baseline loss:', str(loss_baseline_output)[:6],
                  'episode length:', len(accumulated_reward_input))

        if iteration % 100 == 0:
            if not os.path.exists(ckpt_dir):
                os.mkdir(ckpt_dir)
            checkpoint_path = os.path.join(ckpt_dir, "model.ckpt")
            filename = saver.save(sess, checkpoint_path)
            # print("Model saved in file: %s" % filename)


if __name__== '__main__':
    train()