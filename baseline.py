import os
import gym
import numpy as np
import torch
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from gym.wrappers.monitoring.video_recorder import VideoRecorder


def tuning_model(env, env_id, agent_type, device):
    if agent_type == "DuelingDQN":
        if env_id == "BreakoutNoFrameskip-v4":
            model = DQN(policy='CnnPolicy', env=env, buffer_size=10000, learning_rate=1e-4, batch_size=32,
                        learning_starts=100000, target_update_interval=1000, train_freq=4, gradient_steps=1,
                        device=device, tensorboard_log=f"./DQN_{env_id}_tensorboard/")
        else:
            model = DQN(policy='CnnPolicy', env=env, buffer_size=10000, learning_rate=1e-4, batch_size=32,
                        learning_starts=100000, target_update_interval=1000, train_freq=4, gradient_steps=1,
                        device=device, tensorboard_log=f"./DQN_{env_id}_tensorboard/")
        iters = 1e7
    elif env_id == "Hopper-V2":
        model = PPO(policy='MlpPolicy', env=env, batch_size=32, n_steps=512, gamma=0.999,
                    learning_rate=1e-4, ent_coef=2e-3, n_epochs=5, max_grad_norm=0.7,
                    device=device, tensorboard_log=f"./PPO_{env_id}_tensorboard/")
        iters = 1e6
    else:
        model = PPO(policy='MlpPolicy', env=env, batch_size=32, n_steps=512, gamma=0.98,
                    learning_rate=2e-5, ent_coef=5e-7, n_epochs=10, max_grad_norm=0.6,
                    device=device, tensorboard_log=f"./PPO_{env_id}_tensorboard/")
        iters = 1e7
    return env, model, iters


def run_test(env, model, ep, total, video_name, render_available):
    obs = env.reset()
    if render_available:
        recorder = VideoRecorder(env, path=video_name)
    tot_rewards = 0
    done = False
    num_iter = 0
    max_iter = 10000
    while not done and num_iter <= max_iter:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        tot_rewards += np.mean(rewards)
        num_iter += 1
        if isinstance(done, np.ndarray):
            flag = True
            for d in done:
                flag = flag and d
            done = flag
        if render_available:
            env.render('rgb_array')
            recorder.capture_frame()
    print(f'In the episode {ep} / {total}, the rewards: {tot_rewards:.2f}')
    if render_available:
        recorder.close()


def test_game(env_id, agent_type, mode):
    # Create the environment
    if agent_type == "PPO":
        env = gym.make(env_id)
    else:
        env = make_atari_env(env_id, n_envs=4, seed=0)
        env = VecFrameStack(env, n_stack=4)
    video_path = "video"
    if not os.path.exists(video_path):
        os.makedirs(video_path)
    model_path = "models"
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # required before you can step the environment
    env.reset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env, model, iters = tuning_model(env, env_id, agent_type, device)
    if mode == "train+test":
        epoch_num = 10000
        save_epoch = 1000
        iter_num = int(iters)//epoch_num
        for epoch in range(epoch_num):
            model.learn(total_timesteps=iter_num, reset_num_timesteps=False)
            run_test(env, model, epoch, epoch_num, None, False)
            if epoch % save_epoch == 0:
                model.save(f"{model_path}/{env_id}_{agent_type}.zip")
        model.save(f"{model_path}/{env_id}_{agent_type}.zip")
    else:
        model_saved = f"{model_path}/{env_id}_{agent_type}.zip"
        if agent_type == "PPO":
            model = PPO.load(model_saved, env=env)
        else:
            model = DQN.load(model_saved, env=env, optimize_memory_usage=False)

    gl_available = False    # If GELW is available, set to True.
    render_available = gl_available or agent_type != "PPO"
    episodes = 3
    model.env = env
    for ep in range(episodes):
        video_name = f"{video_path}/{env_id}_{agent_type}_{ep}.mp4"
        run_test(env, model, ep, episodes, video_name, render_available)

    env.close()
