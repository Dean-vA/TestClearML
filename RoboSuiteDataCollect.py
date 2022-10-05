#%%
import numpy as np
import gym
import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

directory = 'ObjDetectData_PickPlace'

joint_dim = 6
options = {}
# Choose controller
controller_name = 'OSC_POSE'
# Load the desired controller
options["controller_configs"] = suite.load_controller_config(default_controller=controller_name)

# create environment instance
env = suite.make(**options,
    env_name="Lift", # try with other tasks like "Stack" and "Door", "PickPlace"
    robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=False,
    has_offscreen_renderer=True,
    control_freq=20,                        # 20 hz control for applied actions
    horizon=500,                            # each episode terminates after 200 steps
    use_object_obs=True,                    # provide object observations to agent
    use_camera_obs=True,                   # don't provide image observations to agent
    reward_shaping=True,                    # use a dense reward signal for learning
    camera_segmentations = 'element'
)

runs = 500
steps = 2

images = np.empty((runs*steps,256,256,3),dtype='uint8')
seg_images = np.empty((runs*steps,256,256),dtype='uint8')
coords = np.empty((runs*steps,6))
index = -1

for i in range(runs):
    for step in range(steps):
        index +=1
        action = np.random.randn(env.robots[0].dof-1)*2 # sample random action
        obs, reward, done, info = env.step(action)
        #plt.imshow(np.flipud(obs['agentview_image']))
        #plt.show()
        #plt.imshow(np.flipud(obs['agentview_segmentation_element']))
        #plt.show()
        #plt.imsave(directory+'/Orig'+str(step*(i+1))+'.png', np.flipud(obs['agentview_image']))
        #plt.imsave(directory+'/Seg'+str(step*(i+1))+'.png', np.squeeze(np.flipud(obs['agentview_segmentation_element'])))
        images[index] = np.flipud(obs['agentview_image'])
        seg_images[index] = np.squeeze(np.flipud(obs['agentview_segmentation_element']))
        #print(obs['object-state'][0:6])
        xyz = obs['object-state'][:3]
        quat = obs['object-state'][3:7]
        ypr = R.from_quat(quat).as_euler('zyx', degrees=True)/180
        coords[index] = np.hstack((xyz,ypr))
        #print(np.round(coords[index],8))
        #env.render()
        #print(obs)
        if done:
            obs = env.reset()
    obs = env.reset()
    print(i)

np.savez('Object_coord_data9.npz',images,seg_images,coords)
#print(obs.keys())
env.close()
# %%
