from rllab.sampler.utils import rollout
from railrl.misc.icm_util import test_icm, test_state_hist, test_icm_cmaes, investigate_inverse_loss, investigate_forward_loss, plot_forward, plot_action, get_max_reward, get_time_to_first_contact, analyze_forward_loss, analyze_ef_range

import argparse
import joblib
import uuid
import tensorflow as tf
# import envs

filename = str(uuid.uuid4())



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--max_path_length', type=int, default=1000,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1,
                        help='Speedup')
    parser.add_argument('--icm', action='store_true')
    parser.add_argument('--test_state_hist', action='store_true')
    parser.add_argument('--cmaes', action='store_true')
    parser.add_argument('--test_inverse_loss', action='store_true')
    parser.add_argument('--test_forward_loss', action='store_true')
    parser.add_argument('--plot_forward', action='store_true')
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--plot_action', action='store_true')
    parser.add_argument('--get_max_reward', action='store_true')
    parser.add_argument('--time_contact', action='store_true')
    parser.add_argument('--analyze_forward', action='store_true')
    parser.add_argument('--analyze_ef_range', action='store_true')
    parser.add_argument('--data_path', type=str, default='/tmp/data/')
    args = parser.parse_args()

    policy = None
    env = None

    with tf.Session() as sess:
        data = joblib.load(args.file)
       
        if 'policy' in data:
            policy = data['policy']
        elif 'optimizable_qfunction' in data:
            qf = data['optimizable_qfunction']
            policy = qf.implicit_policy
        if 'inverse_model' in data:
            encoder = data['encoder']
            print (encoder)
            inverse_model = data['inverse_model']
            forward_model = data['forward_model']
        elif args.test_state_hist:
            pass

        # else:
        #     raise ValueError("Unsupported snapshot!")

        env = data['env']
        if args.test_state_hist:
            test_state_hist(env)
        elif args.get_max_reward:
            get_max_reward(env, policy, num_trajs=200)
        elif args.icm:
            if args.cmaes:
                from railrl.policies.cmaes_icm import CMAESPolicy
                policy = CMAESPolicy(env.spec, encoder, inverse_model, forward_model, sess=sess)
                test_icm_cmaes(encoder, inverse_model, forward_model, env, policy, sess)
            elif args.test_inverse_loss:
                investigate_inverse_loss(encoder, inverse_model, forward_model, env, policy, sess, img_path=args.data_path, num_trajs=100, animate=args.render)
            elif args.test_forward_loss:
                if policy is None:
                    # TODO: Remove this hack after CoRL deadline
                    from rllab.policies.uniform_control_policy import UniformControlPolicy
                    policy = UniformControlPolicy(env.spec)
                investigate_forward_loss(encoder, inverse_model, forward_model, env, policy, sess, data_path=args.data_path, num_trajs=200, animate=args.render, num_top=50)
            elif args.plot_forward:
                plot_forward(encoder, inverse_model, forward_model, env, policy, sess)
            elif args.time_contact:
                get_time_to_first_contact(env, policy, is_random=False, num_trajs=500)
            elif args.analyze_forward:
                # analyze_forward_loss(encoder, forward_model, inverse_model, env, policy, sess, data_path=args.data_path)
                # from rllab.policies.uniform_control_policy import UniformControlPolicy
                # policy = UniformControlPolicy(env.spec)
                # analyze_forward_loss(encoder, forward_model, inverse_model, env, policy, sess, data_path='/home/dianchen/corl/data/state_analysis/icm_data_policy_on_policy.pkl', num_sample=5000)
                analyze_forward_loss(encoder, forward_model, inverse_model, env, policy, sess, data_path=args.data_path, num_sample=5000)
            elif args.analyze_ef_range:
                analyze_ef_range(env, policy, num_sample=100000, data_path=args.data_path)
            else:
                from rllab.policies.uniform_control_policy import UniformControlPolicy
                policy = UniformControlPolicy(env.spec)
                test_icm(encoder, inverse_model, forward_model, env, policy, sess, animate=False)
        elif args.plot_action:
            plot_action(env, policy, '/Users/dianchen/plot_actions.npy')
        else:
            while True:
                try:
                    path = rollout(env, policy, max_path_length=args.max_path_length,
                                   animated=True, speedup=args.speedup)
                # Hack for now. Not sure why rollout assumes that close is an
                # keyword argument
                except TypeError as e:
                    if (str(e) != "render() got an unexpected keyword "
                                  "argument 'close'"):
                        raise e


