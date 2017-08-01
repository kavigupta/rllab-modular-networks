import argparse

import joblib
import tensorflow as tf
from pickle import dump

from rllab.misc.console import query_yes_no
from rllab.sampler.utils import rollout

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--max_path_length', type=int, default=1000,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1,
                        help='Speedup')
    parser.add_argument('--rollout', type=str, default=None,
                        help='rollout locatino')
    args = parser.parse_args()

    # If the snapshot file use tensorflow, do:
    # import tensorflow as tf
    # with tf.Session():
    #     [rest of the code]
    with tf.Session() as sess:
        data = joblib.load(args.file)
        policy = data['policy']
        env = data['env']
        env.image_list = []
        env.setup_camera()
        path = rollout(env, policy, max_path_length=args.max_path_length,
                       animated=True, speedup=args.speedup, always_return_paths=True)
        if args.rollout is not None:
            with open(args.rollout, "wb") as f:
                dump([path, env.image_list], f)
