#%%
import gym

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

import wandb
from wandb.integration.sb3 import WandbCallback

from clearml import Task
task = Task.init(project_name='Test', task_name='CartPole1')
task.execute_remotely(queue_name="CPU1")

#%%
config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 250000,
    "env_name": "CartPole-v1",
}
run = wandb.init(
    project="sb3CartPole",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)

def make_env():
    env = gym.make(config["env_name"])
    env = Monitor(env)  # record stats such as returns
    return env
#%%
env = DummyVecEnv([make_env]) # Only one environment
#env = VecVideoRecorder(env, f"videos/{run.id}", record_video_trigger=lambda x: x % 2000 == 0, video_length=200)
#%%

model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{run.id}",
        verbose=2,
    ),
)
run.finish()
model.save("ppo_cartpole")