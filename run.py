import os
import gym
import argparse
from labs import *
import numpy as np
import torch
from wrapper import AtariWrapper
from baseline import test_game


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="PongNoFrameskip-v4",
                        choices=["PongNoFrameskip-v4", "BreakoutNoFrameskip-v4", "Hopper-v3", "Ant-v3"])
    parser.add_argument("--agent_type", type=str, default="DuelingDQN", choices=["DuelingDQN", "PPO"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", type=str, default="test", choices=["train+test", "test"])
    return parser.parse_args()


def main():
    os.add_dll_directory("C:\\Users\\Alr_k\\.mujoco\\mujoco200\\bin")   # add you own mujoco path
    args = parse_args()
    try:
        env = gym.make(args.env_name)
    except gym.error.UnregisteredEnv:
        print(f"The environment {args.env_name} is not recognized.")
        return
    if args.env_name == "PongNoFrameskip-v4" or args.env_name == "BreakoutNoFrameskip-v4":
        args.agent_type = "DuelingDQN"
    else:
        args.agent_type = "PPO"

    if args.mode == "train":
        # wrapper the environment
        if args.agent_type == "DuelingDQN":
            env = AtariWrapper(env)
            env.seed(args.seed)
        elif args.agent_type == "PPO":
            env = env.env

        # Seed the environment and PyTorch for reproducibility
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        if args.agent_type == "DuelingDQN":
            agent = DuelingDQN_Agent(env.observation_space, env.action_space)
        elif args.agent_type == "PPO":
            agent = PPO_Agent(env.observation_space, env.action_space)
        else:
            raise ValueError("Invalid agent_type: " + args.agent_type)

        train_agent(agent, env, args.episodes, args.agent_type)

        video_path = "video"
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        test_agent(agent, env, 1, args.agent_type, video_path)
    else:
        test_game(args.env_name, args.agent_type, args.mode)

    env.close()


if __name__ == "__main__":
    main()

