import os
from sandbox.rocky.tf.baselines.gaussian_conv_baseline import GaussianConvBaseline
from sandbox.rocky.tf.policies.conv_nn_policy import ConvNNPolicy
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.gym_env import GymEnv
import itertools

stub(globals())

# Param ranges
seeds = range(5)

for seed in seeds:
    mdp = TfEnv(normalize(env=GymEnv('Box3dReachPixel-v2',record_video=False, \
    log_dir='/tmp/gym_test',record_log=False)))
    
    policy = ConvNNPolicy(
        "conv_policy",
        env_spec=mdp.spec,
        conv_filters=(32, 32, 32, 32),
        conv_filter_sizes=((3,3),(3,3),(3,3),(3,3)),
        conv_strides=(2, 2, 2, 2),
        conv_pads=('SAME', 'SAME', 'SAME', 'SAME'),
        hidden_sizes=(256,),
    )

    # baseline = ZeroBaseline(mdp.env_spec)
    batch_size = 5000
    algo = TRPO(
        env=mdp,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        whole_paths=True,
        max_path_length=500,
        n_itr=1000,
        step_size=0.01,
        subsample_factor=1.0,
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix='trpo_box3d_pixel_v2_tf',
        n_parallel=1,
        snapshot_mode="last",
        seed=seed,
        mode="local"
    )
