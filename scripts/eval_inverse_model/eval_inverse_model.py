from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.gym_env import GymEnv
import numpy as np
import tensorflow as tf

import joblib
import argparse


import matplotlib.pyplot as plt

STATE_ENV = 'Box3dReach-v12'
PIXEL_ENV = 'Box3dReachPixel-v11'

# STATE_PATH = [
#   '/home/dianchen/rail-rl/scripts/analysis/inverse_test_data/state_v10_handeye.pkl',
# ]
STATE_PATH = [
    '/home/dianchen/rail-rl/scripts/eval_inverse_model/data/state_handeye',
    # '/home/dianchen/rail-rl/scripts/eval_inverse_model/data/state_table'
    # '/home/dianchen/rail-rl/scripts/eval_inverse_model/data/state_policy_dist',
    # '/home/dianchen/rail-rl/scripts/eval_inverse_model/data/state_random_dist',
    '/home/dianchen/rail-rl/scripts/eval_inverse_model/data/state_boxes',

]

STATE_MULTISTEP_PATH = [
    '/home/dianchen/rail-rl/scripts/eval_inverse_model/data/state_multistep_4repeat', # This is for v17
    # '/home/dianchen/rail-rl/scripts/eval_inverse_model/data/state_multistep' # This is for v12
    # '/home/dianchen/rail-rl/scripts/eval_inverse_model/data/state_multistep_normed', # This is for normalized v12
    # '/home/dianchen/rail-rl/scripts/eval_inverse_model/data/state_multistep_normed_4step',
]


## TODO: Add pixel dataset
PIXEL_PATH = [
    '/z/dianchen/rail-rl/scripts/eval_inverse_model/data/pixel_random_dist',
    '/z/dianchen/rail-rl/scripts/eval_inverse_model/data/pixel_boxes',
]

PIXEL_MULTISTEP_PATH = [
    '/z/dianchen/rail-rl/scripts/eval_inverse_model/data/pixel_multistep'
]


### Helper functions ####

def load_dataset(is_pixel, is_multistep):
    datasets = []
    if is_pixel:
        if is_multistep:
            dataset_paths = PIXEL_MULTISTEP_PATH
        else:
            dataset_paths = PIXEL_PATH
    else:
        if is_multistep:
            dataset_paths = STATE_MULTISTEP_PATH
        else:
            dataset_paths = STATE_PATH

    for path in dataset_paths:
        print ("Loading %s ..." %path)
        dataset = joblib.load(path)
        datasets.append(dataset)
        print ("Dataset size: %d"%np.prod(dataset['states'].shape[:2]))
    return dataset_paths, datasets

def set_state(env, state, qpos_dim):
    inner_env = env.wrapped_env._wrapped_env.env.env
    inner_env.set_state(state[:qpos_dim], state[qpos_dim:])

def get_ef_pos(env):
    d = env._wrapped_env.wrapped_env.env.env.model.data
    xpos = d.site_xpos.flatten()
    return xpos

def get_state(env):
	d = env.wrapped_env._wrapped_env.env.env.model.data
	qpos = d.qpos.flatten()
	qvel = d.qvel.flatten()

	return np.concatenate([qpos, qvel])

def get_render_img(env):
    return env.wrapped_env._wrapped_env.env.env.render(mode='rgb_array')

def get_qpos(env):
    inner_env = env.wrapped_env._wrapped_env.env.env
    return inner_env.model.data.qpos.flatten()

def load_data(data_dict, pixel=False):
    states = data_dict['states']
    next_states = data_dict['next_states']
    obs = data_dict['obs']
    next_obs = data_dict['next_obs']
    actions = data_dict['actions']

    states = states.reshape(states.shape[0]*states.shape[1], states.shape[2])
    next_states = next_states.reshape(next_states.shape[0]*next_states.shape[1], next_states.shape[2])
    obs = obs.reshape([obs.shape[0]*obs.shape[1]] + list(obs.shape[2:]))
    next_obs = next_obs.reshape([next_obs.shape[0]*next_obs.shape[1]] + list(next_obs.shape[2:]))
    actions = actions.reshape(actions.shape[0]*actions.shape[1], actions.shape[2])

    if pixel:
        obs = obs / 255.0 - 0.5
        next_obs = next_obs / 255.0 - 0.5

    return states, next_states, obs, next_obs, actions

def load_data_multistep(data_dict, step_size=5, pixel=False):
    raw_states = data_dict['states']
    raw_next_states = data_dict['next_states']
    raw_obs = data_dict['obs']
    raw_next_obs = data_dict['next_obs']
    raw_actions = data_dict['actions']

    assert raw_states.shape[1] % step_size == 0, "%d is not divisible by %d!" % (raw_states.shape[2], step_size)

    if pixel:
        raw_obs = raw_obs / 255.0 - 0.5
        raw_next_obs = raw_next_obs / 255.0 - 0.5


    num_batches = int(raw_states.shape[1] / step_size)

    states_batches = np.concatenate([np.split(raw_state_traj, num_batches) for raw_state_traj in raw_states])
    next_states_batches = np.concatenate([np.split(raw_next_state_traj, num_batches) for raw_next_state_traj in raw_next_states])
    obs_batches = np.concatenate([np.split(raw_obs_traj, num_batches) for raw_obs_traj in raw_obs])
    next_obs_batches = np.concatenate([np.split(raw_next_obs_traj, num_batches) for raw_next_obs_traj in raw_next_obs])
    actions_batches = np.concatenate([np.split(raw_action_traj, num_batches) for raw_action_traj in raw_actions])

    return states_batches, next_states_batches, obs_batches, next_obs_batches, actions_batches


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, default='path to snapshot file')
    parser.add_argument('--pixel', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--multistep', action='store_true')
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--zero_action', action='store_true')
    parser.add_argument('--gt_action', action='store_true')

    args = parser.parse_args()

    with tf.Session() as sess:
        data = joblib.load(args.file)
        _encoder = data['encoder']
        _inverse_model = data['inverse_model']
        _forward_model = data['forward_model']

        if args.pixel:
            env = TfEnv(normalize(env=GymEnv(PIXEL_ENV,record_video=False, \
            log_dir='/tmp/gym_test',record_log=False)))
        else:
            env = TfEnv(normalize(env=GymEnv(STATE_ENV,record_video=False, \
            log_dir='/tmp/gym_test',record_log=False)))

        # Rebuild models
        act_space = env.action_space
        obs_space = env.observation_space
        qpos_dim = env.wrapped_env._wrapped_env.env.env.init_qpos.shape[0]

        s1_ph = tf.placeholder(tf.float32, [None] + list(obs_space.shape))
        s2_ph = tf.placeholder(tf.float32, [None] + list(obs_space.shape))
        a_ph = tf.placeholder(tf.float32, [None, act_space.flat_dim])
        
        clipped_a = tf.clip_by_value(a_ph, -1.0, 1.0)
        encoder1 = _encoder.get_weight_tied_copy(observation_input=s1_ph)
        encoder2 = _encoder.get_weight_tied_copy(observation_input=s2_ph)
        inverse_model = _inverse_model.get_weight_tied_copy(feature_input1=encoder1.output, 
                                                            feature_input2=encoder2.output)
        forward_model = _forward_model.get_weight_tied_copy(feature_input=encoder1.output,
                                                            action_input=clipped_a)

        # Load test data
        dataset_paths, datasets = load_dataset(args.pixel, args.multistep)


        env.reset()
        for dataset_path, data_dict in zip(dataset_paths, datasets):
            
            ef_xyz_pred_diff = []
            ef_xyz_diff = []
            action_diff = []
            qpos_diff = []
            qpos_pred_diff = []
            if args.multistep:
                print ("===== Using multisteping testing, stepsize: %d" % args.step_size)
            
            print ("========================================")
            print ("===== Evaluating inverse model on %s" % dataset_path)
            # states = data_dict['states']
            # next_states = data_dict['next_states']
            # obs = data_dict['obs']
            # next_obs = data_dict['next_obs']
            # actions = data_dict['actions']
            if args.multistep:
                states, next_states, obs, next_obs, actions = load_data_multistep(data_dict, pixel=args.pixel, step_size=args.step_size)
            else:
                states, next_states, obs, next_obs, actions = load_data(data_dict, args.pixel)
            actions = np.clip(actions, -1.0, 1.0)

            if args.render:
                fig, [ax1, ax2, ax3] = plt.subplots(1, 3)
                plt.ion()
                ax1.set_title("t=0")
                ax2.set_title("t=1 after action")
                ax3.set_title("t=1 after predicted action")


            for state, next_state, ob, next_ob, action in zip(states, next_states, obs, next_obs, actions):
                # print (state.shape)
                if args.multistep:
                    # Set state, get real img1
                    set_state(env, state[0], qpos_dim)
                    _end_ef_pos = get_ef_pos(env)
                    _qpos = get_qpos(env)
                    if args.render:
                        img = get_render_img(env)

                    o = ob[0]
                    # next_o = next_ob[0]
                    next_o = next_ob[-1]
                    for _ in range(args.step_size):
                        # Get predicted action from inverse model
                        pred_action = sess.run(inverse_model.output, {
                            s1_ph: [o],
                            s2_ph: [next_o],
                        })[0]
                        if args.gt_action:
                        	pred_action = action[_]

                        if args.zero_action:
                        	pred_action = np.zeros_like(action[_])

                        # ob = next_o
                        # next_o = next_ob[_]

                        # Step predicted action
                        o, r, d, env_info = env.step(pred_action)

                    # Get sim_img2 and sim ef position
                    s_end_ef_pos = get_ef_pos(env)
                    s_qpos = get_qpos(env)
                    if args.render:
                        s_img = get_render_img(env)


                    # Get real img2 and real ef position
                    set_state(env, next_state[args.step_size-1], qpos_dim)
                    o_end_ef_pos = get_ef_pos(env)
                    o_qpos = get_qpos(env)
                    if args.render:
                        o_img = get_render_img(env)


                else:
                    # Set state, get real img1
                    # import pdb; pdb.set_trace()
                    set_state(env, state, qpos_dim)
                    _end_ef_pos = get_ef_pos(env)
                    # print ("Real: ", _end_ef_pos)
                    _qpos = get_qpos(env)
                    if args.render:
                        img = get_render_img(env)

                    # Get predicted action from inverse model
                    pred_action = sess.run(inverse_model.output, {
                        s1_ph: [ob],
                        s2_ph: [next_ob],
                    })[0]

                    if args.zero_action:
                        pred_action = np.zeros_like(pred_action)
                    if args.gt_action:
                    	pred_action = action


                    # Step action
                    env.step(pred_action)

                    # print (np.linalg.norm(next_state - get_state(env)))

                    # Get sim_img2 and sim ef position
                    s_end_ef_pos = get_ef_pos(env)
                    # print ("Sim pos", s_end_ef_pos)
                    s_qpos = get_qpos(env)
                    if args.render:
                        s_img = get_render_img(env)

                    # Get real img2 and real ef position
                    set_state(env, next_state, qpos_dim)
                    o_end_ef_pos = get_ef_pos(env)
                    o_qpos = get_qpos(env)

                    # print (np.linalg.norm(s_qpos - o_qpos))
                    
                    # print (np.linalg.norm(o_end_ef_pos - s_end_ef_pos))

                if args.render:
                    o_img = get_render_img(env)
                
                if args.render:
                    ax1.imshow(img)
                    ax2.imshow(o_img)
                    ax3.imshow(s_img)
                    plt.show()
                    plt.pause(0.1)

                    # print ("Actual action: ", action)
                    # print ("Predicted action: ", pred_action)

                ef_xyz_pred_diff.append(np.linalg.norm(o_end_ef_pos - s_end_ef_pos))
                ef_xyz_diff.append(np.linalg.norm(o_end_ef_pos - _end_ef_pos))
                qpos_pred_diff.append(np.linalg.norm(o_qpos - s_qpos))
                qpos_diff.append(np.linalg.norm(o_qpos - _qpos))
                
                action_diff.append(((action - pred_action)**2).mean())

            # print ("===== 1. real s1, real s2 end effector position L2 distance       mean:  %.5f, std: %.5f" % (np.mean(ef_xyz_diff), np.std(ef_xyz_diff)))
            # print ("===== 2. real s2, sim  s2 end effector position L2 distance       mean:  %.5f, std: %.5f" % (np.mean(ef_xyz_pred_diff), np.std(ef_xyz_pred_diff)))
            # print ("===== 3. real s1, real s2 joint position        L2 distance       mean:  %.5f, std: %.5f" % (np.mean(qpos_diff), np.std(qpos_diff)))
            # print ("===== 4. real s2, sim  s2 joint position        L2 distance       mean:  %.5f, std: %.5f" % (np.mean(qpos_pred_diff), np.std(qpos_pred_diff)))
           # if not args.multistep:
                #print ("===== 5. action - pred_action (per dim)      sq L2 distance       mean:  %.5f, std: %.5f" % (np.mean(action_diff), np.std(action_diff)))
            # print ("===== 6. action                                                   mean:  %.5f, std: %.5f" % (np.mean(np.abs(actions).mean(axis=1)), np.std(actions.mean(axis=1))))

            print ("===== 1. real s1, real s2 end effector position L2 distance       med:  %.5f, std: %.5f" % (np.median(ef_xyz_diff), np.std(ef_xyz_diff)))
            print ("===== 2. real s2, sim  s2 end effector position L2 distance       med:  %.5f, std: %.5f" % (np.median(ef_xyz_pred_diff), np.std(ef_xyz_pred_diff)))
            print ("===== 3. real s1, real s2 joint position        L2 distance       med:  %.5f, std: %.5f" % (np.median(qpos_diff), np.std(qpos_diff)))
            print ("===== 4. real s2, sim  s2 joint position        L2 distance       med:  %.5f, std: %.5f" % (np.median(qpos_pred_diff), np.std(qpos_pred_diff)))
            if not args.multistep:
                    print ("===== 5. action - pred_action (per dim)      sq L2 distance       med:  %.5f, std: %.5f" % (np.median(action_diff), np.std(action_diff)))
            print ("===== 6. action                                                   med:  %.5f, std: %.5f" % (np.median(np.abs(np.median(actions, axis=1))), np.std(np.median(actions, axis=1))))

            # print ("==== 7. Median of action error: %.5f" % np.median(action_diff))
if __name__ == '__main__':
    main()
