import ctypes


class ImageStateSpec(ctypes.Structure):
    _fields_ = [('fov', ctypes.c_uint8 * 256 * 256 * 4), ('tov', ctypes.c_uint8 * 256 * 256 * 4)]


class CharacterStateSpec(ctypes.Structure):
    _fields_ = [('actor_pos_x', ctypes.c_float), ('actor_pos_y', ctypes.c_float), ('actor_pos_z', ctypes.c_float),
                ('actor_vel_x', ctypes.c_float), ('actor_vel_y', ctypes.c_float), ('actor_vel_z', ctypes.c_float),
                ('is_running', ctypes.c_bool), ('reward', ctypes.c_float), ('frame_number', ctypes.c_int)]


class CharacterActionSpec(ctypes.Structure):
    _fields_ = [('spawn_plant', ctypes.c_int), ('quake', ctypes.c_int), ('reset', ctypes.c_bool)]


def print_state(data):
    print(f'actor_pos_x: {data.actor_pos_x} '
          f'actor_pos_y: {data.actor_pos_y} '
          f'actor_pos_z: {data.actor_pos_z} '
          f'actor_vel_x: {data.actor_vel_x} '
          f'actor_vel_y: {data.actor_vel_y} '
          f'actor_vel_z: {data.actor_vel_z} '
          f'is running: {data.is_running}'
          f'reward: {data.reward}',
          f'frame_number: {data.frame_number}')


def to_list(data):
    obs= []
    obs.append(data.actor_pos_x)
    obs.append(data.actor_pos_y)
    obs.append(data.actor_pos_z)
    obs.append(data.actor_vel_x)
    obs.append(data.actor_vel_y)
    obs.append(data.actor_vel_z)
    return obs


def print_action(data):
    print(f'vel_x: {data.spawn_plant} '
          f'vel_y: {data.quake} '
          f'reset: {data.reset}')
