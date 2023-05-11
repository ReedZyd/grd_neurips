import time
import numpy as np
from test import Tester
from envs import make_env, envs_collection
from algorithm import create_agent, algorithm_collection
from algorithm.replay_buffer import create_buffer
from learner import create_learner
from utils.os_utils import get_arg_parser, get_logger, str2bool

def get_args():
    parser = get_arg_parser()

    # basic arguments
    parser.add_argument('--tag', help='the terminal tag in logger', type=str, default='')
    parser.add_argument('--env', help='gym env id', type=str, default='Ant-v2')
    parser.add_argument('--env_type', help='the type of environment', type=str, default='ep_rews')
    parser.add_argument('--alg', help='backend algorithm', type=str, default='rrd', choices=algorithm_collection.keys())
    parser.add_argument('--new', help='if use new setting, constrain the speed', action='store_true', default=False)

    # training arguments
    parser.add_argument('--epochs', help='the number of epochs', type=np.int32, default=3)
    parser.add_argument('--cycles', help='the number of cycles per epoch', type=np.int32, default=100)
    parser.add_argument('--iterations', help='the number of iterations per cycle', type=np.int32, default=100)
    parser.add_argument('--timesteps', help='the number of timesteps per iteration', type=np.int32, default=100)

    # testing arguments
    parser.add_argument('--test_rollouts', help='the number of rollouts to test per cycle', type=np.int32, default=10)
    parser.add_argument('--test_timesteps', help='the number of timesteps per rollout', type=np.int32, default=1000)
    parser.add_argument('--save-rews', help='whether to save cumulative rewards', type=str2bool, default=True)

    # buffer arguments
    parser.add_argument('--buffer_size', help='the number of transitions in replay buffer', type=np.int32, default=1000000)
    parser.add_argument('--batch_size', help='the size of sample batch', type=np.int32, default=256)
    parser.add_argument('--warmup', help='the number of timesteps for buffer warmup', type=np.int32, default=10000)



    # WandB
    parser.add_argument('--with-wandb', default=False, action='store_true', help='Enables Weights and Biases')
    parser.add_argument('--wandb-entity', default='kcl_coopai', type=str, help='WandB username (entity).')
    parser.add_argument('--wandb-project', default='FactoredRD_new', type=str, help='WandB "Project"')
    parser.add_argument('--wandb-group', default=None, type=str, help='WandB "Group". Name of the env by default.')
    parser.add_argument('--wandb-job_type', default='train', type=str, help='WandB job type')
    parser.add_argument('--wandb-tags', default=[], type=str, nargs='*', help='Tags can help finding experiments')
    parser.add_argument('--wandb-key', default=None, type=str, help='API key for authorizing WandB')
    parser.add_argument('--wandb-dir', default=None, type=str, help='the place to save WandB files')
    parser.add_argument('--wandb-experiment', default='', type=str, help='Identifier to specify the experiment')


    args, _ = parser.parse_known_args()

    # env arguments
    parser.add_argument('--gamma', help='discount factor', type=np.float32, default=0.99)

    def mujoco_args():
        pass # no special arguments
    

    env_args_collection = {
        'mujoco': mujoco_args,
    }
    env_category = envs_collection[args.env]
    env_args_collection[env_category]()

    # algorithm arguments
    def q_learning_args():
        parser.add_argument('--train_batches', help='the number of batches to train per iteration', type=np.int32, default=25)
        parser.add_argument('--train_target', help='the frequency of target network updating', type=np.int32, default=8000)

        parser.add_argument('--eps_l', help='the beginning percentage of epsilon greedy explorarion', type=np.float32, default=1.00)
        parser.add_argument('--eps_r', help='the final percentage of epsilon greedy explorarion', type=np.float32, default=0.01)
        parser.add_argument('--eps_decay', help='the number of steps to decay epsilon', type=np.int32, default=250000)

        parser.add_argument('--optimizer', help='the optimizer to use', type=str, default='adam', choices=['adam', 'rmsprop'])
        args, _ = parser.parse_known_args()
        if args.optimizer=='adam':
            parser.add_argument('--q_lr', help='the learning rate of value network', type=np.float32, default=0.625e-4)
            parser.add_argument('--Adam_eps', help='the epsilon factor of Adam optimizer', type=np.float32, default=1.5e-4)
        elif args.optimizer=='rmsprop':
            parser.add_argument('--q_lr', help='the learning rate of value network', type=np.float32, default=2.5e-4)
            parser.add_argument('--RMSProp_decay', help='the decay factor of RMSProp optimizer', type=np.float32, default=0.95)
            parser.add_argument('--RMSProp_eps', help='the epsilon factor of RMSProp optimizer', type=np.float32, default=1e-2)


    def dqn_args():
        q_learning_args()
        parser.add_argument('--double', help='whether to use double trick', type=str2bool, default=False)
        parser.add_argument('--dueling', help='whether to use dueling trick', type=str2bool, default=False)

    def actor_critic_args():
        parser.add_argument('--train_batches', help='the number of batches to train per iteration', type=np.int32, default=100)
        parser.add_argument('--train_target', help='the frequency of target network updating', type=np.int32, default=2)

        parser.add_argument('--pi_lr', help='the learning rate of actor network', type=np.float32, default=3e-4)
        parser.add_argument('--q_lr', help='the learning rate of critic network', type=np.float32, default=3e-4)
        parser.add_argument('--polyak', help='interpolation factor in polyak averaging for target network updating', type=np.float32, default=0.995)
        parser.add_argument('--obs_normalization', help='whether to normalize observation vector', type=str2bool, default=False)

    def dpg_args():
        actor_critic_args()
        parser.add_argument('--std_act', help='the standard deviation of uncorrelated gaussian exploration', type=np.float32, default=0.1)

    def ddpg_args():
        dpg_args()
        parser.set_defaults(obs_normalization=True, pi_lr=1e-4, q_lr=1e-3, polyak=0.999)
        parser.add_argument('--q_reg', help='the l2-regularization of critic network', type=np.float32, default=1e-2)

    def td3_args():
        dpg_args()
        parser.add_argument('--pi_t_std', help='the standard deviation of target policy smoothing', type=np.float32, default=0.2)
        parser.add_argument('--pi_t_clip', help='the clipping range of target policy smoothing', type=np.float32, default=0.5)
        parser.add_argument('--pi_delay_freq', help='the frequency to train actor network', type=np.int32, default=2)

    def sac_args():
        actor_critic_args()
        parser.set_defaults(train_target=1)
        parser.add_argument('--alpha_lr', help='the learning rate of temperature value', type=np.float32, default=3e-4)
        parser.add_argument('--alpha_init', help='the initial value of temperature variable', type=np.float32, default=1.0)

    def basis_alg_args():
        from algorithm import basis_algorithm_collection
        defaul_basis_alg = {'mujoco': 'sac'}[env_category]
        parser.add_argument('--basis_alg', help='the basis algorithm of RRD/IRCR', type=str, default=defaul_basis_alg, choices=basis_algorithm_collection.keys())
        args, _ = parser.parse_known_args()
        global algorithm_args_collection
        algorithm_args_collection[args.basis_alg]()

    def ircr_args():
        basis_alg_args()
        if env_category=='mujoco':
            args, _ = parser.parse_known_args()
            if args.env in ['Humanoid-v2']:
                # This configuration is more stable for Humanoid.
                parser.set_defaults(alpha_lr=0, alpha_init=0.1)
            parser.set_defaults(buffer_size=300000, polyak=0.999, batch_size=512)

    def rrd_args():
        basis_alg_args()
        args, _ = parser.parse_known_args()
        default_batch_size, default_sample_size = {'mujoco': (256, 64)}[env_category]
        parser.add_argument('--r_lr', help='the learning rate of reward network', type=np.float32, default=3e-4)
        parser.add_argument('--rrd_batch_size', help='the size of sample batch for reward regression', type=np.int32, default=default_batch_size)
        parser.add_argument('--rrd_sample_size', help='the size of sample for reward regression', type=np.int32, default=default_sample_size)
        parser.add_argument('--rrd-bias-correction', help='whether to use bias-correction loss', type=str2bool, default=False)

    def causal_args():
        basis_alg_args()
        args, _ = parser.parse_known_args()
        default_batch_size, default_sample_size = {'mujoco': (256, 64)}[env_category]
        parser.add_argument('--r_lr', help='the learning rate of reward network', type=np.float32, default=3e-4)
        parser.add_argument('--causal-batch-size', help='the size of sample batch for reward regression', type=np.int32, default=default_batch_size)
        parser.add_argument('--causal-sample-size', help='the size of sample for reward regression', type=np.int32, default=default_sample_size)
        parser.add_argument('--causal-bias-correction', help='whether to use bias-correction loss', type=str2bool, default=False)
        parser.add_argument('--policy-learning-with-causal', help='if use causality to facilitate policy learning', action='store_true')
        parser.add_argument('--zr-sparsity-coef', help='the coef of sparsity loss', type=np.float32, default=0.1)
        parser.add_argument('--ar-sparsity-coef', help='the coef of sparsity loss', type=np.float32, default=0.1)
        parser.add_argument('--zz-sparsity-coef', help='the coef of sparsity loss', type=np.float32, default=1e-4)
        parser.add_argument('--zz-sparsity-coef-aux', help='the coef of sparsity loss', type=np.float32, default=1e-4)
        parser.add_argument('--az-sparsity-coef', help='the coef of sparsity loss', type=np.float32, default=1e-5)
        parser.add_argument('--temperature', help='the coef of sparsity loss', type=np.float32, default=0.01)
        parser.add_argument('--temperature-adjust-automatically', help='assume that all the binary mask in the causal structure equal to 1', action='store_true')
        parser.add_argument('--reward-loss-sh', help='the coef of sparsity loss', type=np.float32, default=0.0001)
        parser.add_argument('--dyn-loss-sh', help='the coef of sparsity loss', type=np.float32, default=0.5)

        parser.add_argument('--causal-discovery-type', help='increase/reduce sparsity', type=str, default='none', choices=['increase_sparsity', 'none', 'reduce_sparsity'])
        parser.add_argument('--sparsity-loss-type', help='loss for sparsity',type=str, default='cross_entropy', choices=['l1', 'cross_entropy'])
        
        # set to True by default in the code
        parser.add_argument('--full-structure-initial', help='assume that all the binary mask in the causal structure equal to 1', action='store_true')
        parser.add_argument('--apply-accurate-loss', help='if use accurate loss for approximating the reward function', action='store_true')
        parser.add_argument('--obs-norm', help='if use accurate loss for approximating the reward function', action='store_true')

    def ground_args():
        basis_alg_args()
        args, _ = parser.parse_known_args()

    global algorithm_args_collection
    algorithm_args_collection = {
        'dqn': dqn_args,
        'ddpg': ddpg_args,
        'td3': td3_args,
        'sac': sac_args,
        'ircr': ircr_args,
        'rrd': rrd_args,
        'causal':causal_args,
        'ground': ground_args,
    }
    algorithm_args_collection[args.alg]()

    args = parser.parse_args()
    args.env_category = envs_collection[args.env]

    logger_name = args.alg+'-'+args.env
    if args.tag!='': logger_name = args.tag+'-'+logger_name
    args.timestamp =  time.strftime('-(%Y-%m-%d-%H_%M_%S)') + '_' + str(np.random.randint(100))

    if args.alg == 'causal':
        # args.full_structure_initial = True
        args.apply_accurate_loss = True
    args.logger = get_logger(logger_name + args.timestamp)

    for key, value in args.__dict__.items():
        if key!='logger':
            args.logger.info('{}: {}'.format(key,value))

    return args

def experiment_setup(args):
    env = make_env(args)
    args.acts_dims = env.acts_dims
    args.obs_dims = env.obs_dims
    args.state_dim = args.obs_dims
    args.env_instance = env
    args.buffer = buffer = create_buffer(args)
    args.agent = agent = create_agent(args)
    args.agent_graph = agent.graph
    args.learner = learner = create_learner(args)
    args.logger.info('*** network initialization complete ***')
    args.tester = tester = Tester(args)
    args.logger.info('*** tester initialization complete ***')
    return env, agent, buffer, learner, tester


