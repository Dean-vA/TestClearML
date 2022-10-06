#%%
import gym

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

import wandb
from wandb.integration.sb3 import WandbCallback
import torch

from clearml import Task
task = Task.init(project_name='Test', task_name='CartPole1')
task.set_base_docker('nvidia-cuda-11.7.0-base-ubuntu20.04-x86_64-sha256-e585d433c8cac09b487dfe9807364d9035be4a1ec4b59b550b31622bcb798ad3.tgz')
task.execute_remotely(queue_name="CPU1")#,docker_image="11.8.0-cudnn8-runtime-ubuntu20.04")pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

device = torch.device('cpu')


#%%
config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 250000,
    "env_name": "CartPole-v1",
}
#run = wandb.init(
#    project="sb3CartPole",
#    config=config,
#    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
#    monitor_gym=True,  # auto-upload the videos of agents playing the game
#    save_code=True,  # optional
#)

def make_env():
    env = gym.make(config["env_name"])
    env = Monitor(env)  # record stats such as returns
    return env
#%%
env = DummyVecEnv([make_env]) # Only one environment
#env = VecVideoRecorder(env, f"videos/{run.id}", record_video_trigger=lambda x: x % 2000 == 0, video_length=200)
#%%

model = PPO(config["policy_type"], env, verbose=1)#, tensorboard_log=f"runs/{run.id}")
model.learn(
    device=device,
    total_timesteps=config["total_timesteps"],
    #callback=WandbCallback(
        #gradient_save_freq=100,
        #model_save_path=f"models/{run.id}",
        #model_save_path="models/test",
        #verbose=2,
    #),
)
#run.finish()
model.save("ppo_cartpole")