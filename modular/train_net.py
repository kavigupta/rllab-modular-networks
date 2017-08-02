from collections import defaultdict
from time import sleep

import numpy as np
import tensorflow as tf
from scipy.misc import imresize
import matplotlib.pyplot as plt

from rllab.envs.mujoco.reach_env import ReachEnv
from rllab.envs.mujoco.arm_env import COLORS, ALL_CONDITIONS
from modular.modular_architecture import construct_network
from modular.get_rollouts import good_environments, get_rollouts, run_sim_policy
from modular.display_network import show_graph

def train_network(sess, env, net, iterations, cost_callback=print):
    print("Start training", env)
    data_block_locations, data_robot_end_effector, data_joint_angles, data_image, action, last_images = get_rollouts(env)
    print("Get rollout")
    def resize_all(data_images):
        return np.array([imresize(x, (80, 80, 3)) for x in data_images])
    sample_time = list(np.random.choice(list(range(150)), 30))
    print("Sampled time")
    feed_dict = {
        # net.input.obs_image[()] : resize_all(data_image)[sample_time],
        net.input.block_locations[()] : data_block_locations[sample_time],
        net.input.joint_angles[env.number_links] : data_joint_angles[sample_time],
        net.label.action[env.number_links]: action[sample_time],
        # net.label.end_image[()] : resize_all(last_images)[sample_time]
    }
    print("Construct feed dictionary")
    cost = net.loss.action[("state",) + env.tensorcloud_key]
    #get_unified_cost(env, net.loss.action, net.loss.end_image, input_types=["state"], output_types=["joint"])
    with tf.device('/gpu:0'):
        train_op = initialized_adam(sess, cost)
    print("Initialize adam")
    for _ in range(iterations):
        sess.run(train_op, feed_dict=feed_dict)
        cost_callback(env, sess.run(cost, feed_dict=feed_dict))

def get_unified_cost(env, loss_joint, loss_image, input_types=("state", "images"), output_types=("joint", "end_image")):
    outputs = lambda vals: {
        "joint" : loss_joint[(vals,) + env.tensorcloud_key],
        "end_image" : 1e-2 * loss_image[(vals,) + env.task_type_key] / (80 * 80 * 3 * 256^2)
    }
    return sum((print(vals, output_type), outputs(vals)[output_type])[1]
                for output_type in output_types
                for vals in input_types)

def initialized_adam(sess, cost, *args, **kwargs):
    temp = set(tf.all_variables())
    train_step = tf.train.AdamOptimizer(*args, **kwargs).minimize(cost)
    sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))
    return train_step

def main():
    plt.ion()
    sess = tf.Session()
    net = construct_network(
        hidden_size=64, number_layers=1,
        conv_size=64, n_conv_layers=2,
        image_width=80, image_height=80)
    print("Constructed Network")
    show_graph(path="/home/abhigupta/log.html")
    print("Graph shown")
    sess.run(tf.variables_initializer(tf.trainable_variables()))
    print("Initialize all variables")
    environments = [env
                    for cls in (ReachEnv,)
                    for condition in range(len(ALL_CONDITIONS))
                    for num_links in (3, 4, 5)
                    for env in cls.all_envs(is_3d=True, condition=condition, number_links=num_links)]
    envs = good_environments(environments, plot=False)
    print("Filtered environments")
    training_trajectories = defaultdict(list)
    starting_cost = {}
    for _ in range(100):
        for env in envs:
            def cost_callback(env, cost):
                if len(training_trajectories[str(env)]) % 100 == 0:
                    print("*", end="")
                if str(env) not in starting_cost:
                    starting_cost[str(env)] = cost
                training_trajectories[str(env)].append(cost)
            print("Start iteration")
            train_network(sess, env(), net, 1000, cost_callback=cost_callback)
            print()
            plt.cla()
            for env_name, trajectory in training_trajectories.items():
                plt.plot(np.array(trajectory) / starting_cost[env_name], label=env_name)
            plt.legend(bbox_to_anchor=(3, 0))
            plt.draw()
            plt.pause(1)
            print("Finished iteration")

if __name__ == '__main__':
    main()
