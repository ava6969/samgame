import random
import subprocess
import sys
import mmap
import ctypes
from matplotlib import pyplot as plt
import time
import numpy as np
import envs.DataStructs as ds
import matplotlib

matplotlib.use('TkAgg')

if __name__ == '__main__':

    StateType = ds.CharacterStateSpec
    ActionType = ds.CharacterActionSpec
    # proc = subprocess.Popen([f'builds/test_character/Host.exe', '-windowed', '-nosound',
    #                          '-worker_id 0', 'ResX=128', 'ResY=128', '-log',
    #                          f'WinX={random.randint(100, 1100)} WinY={random.randint(10, 700)}'])
    # time.sleep(5)

    action_shmem = mmap.mmap(-1, ctypes.sizeof(ActionType), "SHM_ACTION_0")
    obs_shmem = mmap.mmap(-1, ctypes.sizeof(StateType), "SHM_OBS_0")
    img_shmem = mmap.mmap(-1, ctypes.sizeof(ds.ImageStateSpec), "SHM_IMAGE_0")

    for i in range(100):

        print('Episode ', i + 1)
        done = False
        reward = 0
        # state = env.reset()
        ActionType.from_buffer(action_shmem).reset = True
        fig, (ax1, ax2) = plt.subplots(2)
        # wait for isRunning
        isRunning = False
        while not isRunning:
            data = StateType.from_buffer(obs_shmem)
            img = ds.ImageStateSpec.from_buffer(img_shmem)
            isRunning = data.is_running
            # print('waiting for reset')

        ds.print_state(data)

        current_frame = data.frame_number
        done = False
        while not done:
            # get random action

            # env.step(action)
            data = ActionType.from_buffer(action_shmem)
            data.forward = random.choice([-1, 0, 1])
            data.right = random.choice([-1, 0, 1])
            # print('sent :', end=' ')
            # ds.print_action(data)

            # next observation
            next_frame = current_frame
            while current_frame == next_frame:
                data = StateType.from_buffer(obs_shmem)
                next_frame = data.frame_number
                done = not data.is_running
                if done:
                    break

            img = ds.ImageStateSpec.from_buffer(img_shmem)

            image_fov = np.frombuffer(img.fov, dtype=np.uint8).reshape((256, 256, 4))
            image_tov = np.frombuffer(img.tov, dtype=np.uint8).reshape((256, 256, 4))

            ax1.imshow(image_fov[:, :, :3])

            ax2.imshow(image_tov[:, :, :3])

            plt.show(block=False)
            plt.pause(0.001)

            current_frame = next_frame
            # reward and is running
            reward += data.reward
            done = not data.is_running
        print('total reward', reward)
