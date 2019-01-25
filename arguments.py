import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--algo', default='a2c',
                        help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 7e-4)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--num-processes', type=int, default=8,
                        help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--num-steps', type=int, default=5,
                        help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--ppo-epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=32,
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip-param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--save-interval', type=int, default=100)
    parser.add_argument('--num-frames', type=int, default=10e7)
    parser.add_argument('--result-dir', default='./trained_models/',
                        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--recurrent-policy', action='store_true', default=True,
                        help='use a recurrent policy')
    parser.add_argument('--use-rmsprop', action='store_true', default=True,
                        help='use RMSprop optimizer for A2C')
    parser.add_argument('--reward-scale', type=float, default=0.01)
    parser.add_argument('--frame-skip', type=int, default=4,
                        help='number of frames to skip for each step')
    parser.add_argument('--game-config', default="ViZDoom_map/my_health.cfg")
    parser.add_argument('--use-depth', action='store_true', default=False)
    parser.add_argument('--disable-rgb', action='store_true', default=False)
    parser.add_argument('--jitter-rgb', action='store_true', default=False)
    parser.add_argument('--continue-training', action='store_true', default=False)

    parser.add_argument('--mse-coef', type=float, default=1.0)
    parser.add_argument('--kl-coef', type=float, default=1.0)

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args
