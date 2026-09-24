# Python standard libraries
from threading import Thread
from queue import Queue
import socket

# 3rd party libraries
import numpy as np

# VAHSimulator library
from .utils import quaternion_to_euler, eci_to_geodetic, eci_to_enu_attitude, eci_to_enu, eci_to_ned_attitude
from .parameters import max_Th, lat_ref, lon_ref, alt_ref
from . import performance_decorator

class Communication:
    def __init__(self, sim_id=0, buffer_size=10000):
        self.MCAST_GRP = '127.0.0.1'
        self.MCAST_PORT = 5007
        self.sim_id = sim_id
        self.buffer_size = buffer_size
        self.queue = Queue(maxsize=buffer_size)
        self.running = True
        self.thread = Thread(target=self._send_messages, daemon=True)
        self.thread.start()
        self.first_step = True

    @performance_decorator.time_execution_stats
    def step(self, t, phase, x, x_nav, By, Bz, delta_1, delta_2, delta_3, delta_4, delta_left, delta_right, alpha, beta, mach, pn_pred, Th):
        done = False
        if x[2, 0] >= 0 or self.first_step:
            done = True
            self.first_step = False
        
        position = x[:3]
        quat = x[3:7]
        roll, pitch, yaw = quaternion_to_euler(quat)
        attitude = np.array([roll, pitch, yaw])

        norm_Th = Th / max_Th

        # Compute state in ENU frame
        position_enu = eci_to_enu(position[0, 0], position[1, 0], position[2, 0], lat_ref, lon_ref, alt_ref, t)
        lat, lon, alt = eci_to_geodetic(position[0, 0], position[1, 0], position[2, 0], t)
        attitude_enu, roll_enu, pitch_enu, yaw_enu = eci_to_enu_attitude(roll, pitch, yaw, lat, lon, t)

        # Compute state in NED frame
        roll_ned, pitch_ned, yaw_ned = eci_to_ned_attitude(roll, pitch, yaw, lat, lon, t)

        position_nav = x_nav[4:7]
        quat_nav = x_nav[0:4]
        roll_nav, pitch_nav, yaw_nav = quaternion_to_euler(quat_nav)
        attitude_nav = np.array([roll_nav, pitch_nav, yaw_nav])

        '''
        t: 0
        position: 1 2 3
        position_nav: 4 5 6
        attitude: 7 8 9
        attitude_nav: 10 11 12
        done: 13
        quat: 14 15 16 17
        By: 18
        Bz: 19
        alpha: 20
        beta: 21
        mach: 22
        pn_pred: 23 24 25
        position_enu: 26 27 28
        attitude_enu: 29 30 31 32
        roll_ned: 33
        pitch_ned: 34
        yaw_ned: 35
        alt: 36
        norm_Th: 37
        delta_1: 38
        delta_2: 39
        delta_3: 40
        delta_4: 41
        delta_left: 42
        delta_right: 43
        sim_id: 44
        phase: 45
        '''

        message = np.concatenate(
            [np.array([t]).flatten(), 
             position.flatten(), position_nav.flatten(),
             attitude.flatten(), attitude_nav.flatten(),
             np.array([done]).flatten(), quat.flatten(),
             np.array([By]), np.array([Bz]), 
             np.array([alpha]), np.array([beta]),
             np.array([mach]), pn_pred.flatten(),
             position_enu.flatten(), attitude_enu.flatten(),
             np.array([np.rad2deg(roll_ned)]), np.array([np.rad2deg(pitch_ned)]), np.array([np.rad2deg(yaw_ned)]),
             np.array([alt]), np.array([norm_Th]),
             np.array([delta_1]), np.array([delta_2]), np.array([delta_3]), np.array([delta_4]), 
             np.array([delta_left]), np.array([delta_right]),
             np.array([self.sim_id]),
             np.array([phase])]).tobytes()

        if not self.queue.full():
            self.queue.put(message)

    def _send_messages(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)

        while self.running:
            try:
                message = self.queue.get(timeout=1)
                sock.sendto(message, (self.MCAST_GRP, self.MCAST_PORT))
            except Exception:
                continue

    def _clear_queue(self):
        while not self.queue.empty():
            self.queue.get()

    def stop(self):
        self.running = False
        self.thread.join()
