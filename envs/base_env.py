import ctypes
import random
import subprocess
import time
import gym
import numpy as np
import mmap
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import DataStructs as ds


class UE4Env(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self,
                 name,
                 action_mem_sz,
                 state_mem_sz,
                 visual,
                 rank=0,
                 frame_skip=1,
                 log=False,
                 action_space=None):

        self.rank = rank

        if visual:
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 4), dtype=np.uint8)
        else:
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,))

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,)) if not action_space else action_space

        _log = '-log' if log else ''
        self.proc = subprocess.Popen([f'builds/{name}/Host.exe', '-windowed', '-nosound',
                                      '-worker_id', str(rank), 'ResX=128', 'ResY=128', _log,
                                      f'WinX={random.randint(100, 1100)} WinY={random.randint(10, 700)}'])

        time.sleep(rank + 5 ) # wait for full initialization
        self.action_shmem = mmap.mmap(-1, action_mem_sz, f"SHM_ACTION_{self.rank}")
        self.obs_shmem = mmap.mmap(-1, state_mem_sz, f"SHM_OBS_{self.rank}")
        self.img_shmem = mmap.mmap(-1, ctypes.sizeof(ds.ImageStateSpec), f"SHM_IMAGE_{self.rank}")

        self.state_mem_sz = state_mem_sz
        self.action_mem_sz = action_mem_sz

        self.current_frame = 0
        self.frame_skip = frame_skip
        self.visual = visual
        self.image = np.zeros((84, 84, 4), dtype=np.uint8)

    def reset(self):
        assert self.stateType and  self.actionType
        self.actionType.from_buffer(self.action_shmem).reset = True

        # wait for isRunning
        isRunning = False
        while not isRunning:
            data = self.stateType.from_buffer(self.obs_shmem)
            isRunning = data.is_running
            img = ds.ImageStateSpec.from_buffer(self.img_shmem)

        self.image = np.frombuffer(img, dtype=np.uint8).reshape((84, 84, 4))
        self.current_frame = data.frame_number

        if self.visual:
            return self.image
        else:
            return np.array(ds.to_list(data))

    def step(self, action):
        self.update_action(action)
        next_frame = self.current_frame  # where to perform frame skip

        while self.current_frame + self.frame_skip != next_frame:
            data = self.stateType.from_buffer(self.obs_shmem)
            next_frame = data.frame_number
            done = not data.is_running
            if done:
                break

        self.current_frame = next_frame
        img = ds.ImageStateSpec.from_buffer(self.img_shmem)
        self.image = np.frombuffer(img, dtype=np.uint8).reshape((84, 84, 4))

        done = not data.is_running
        reward = data.reward

        if self.visual:
            new_obs =  self.image
        else:
            new_obs = np.array(ds.to_list(data))

        # time.sleep(2)

        return new_obs, reward, done, {}

    def update_action(self, action):
        return NotImplemented

    def render(self, mode='rgb_array'):
        img = np.copy(self.image)
        if mode == 'rgb_array':
            return img[: , : , -1]  # return RGB frame suitable for video
        elif mode == 'human':
            ...  # pop up a window and render
            plt.imshow(img)
            plt.show(block=False)
            plt.pause(0.001)
        else:
            super(UE4Env, self).render(mode=mode)  # just raise an exception

    def close(self):
        self.proc.terminate()


class UECharacterEnv(UE4Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, rank=0, frame_skip=1, visual=False, log=False):

        action_space = gym.spaces.MultiDiscrete([3, 3])
        super().__init__('test_character', ctypes.sizeof(ds.CharacterActionSpec),
                         ctypes.sizeof(ds.CharacterStateSpec), visual,
                         rank, frame_skip, log, action_space)

        self.stateType = ds.CharacterStateSpec
        self.actionType = ds.CharacterActionSpec

    def update_action(self, action):
        data = self.actionType.from_buffer(self.action_shmem)
        data.forward = -1 if action[0] == 0 else 0 if action[0] == 1 else 1
        data.right = -1 if action[1] == 0 else 0 if action[1] == 1 else 1