import socket
import struct
import threading
from math import atan2, asin

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from vahsimulator.utils import eci_to_ned, eci_to_geodetic, body_to_eci_euler, eci_to_ecef, ecef_to_ned_matrix
from vahsimulator.parameters import lat_ref, lon_ref, alt_ref, we

MCAST_GRP = '127.0.0.1'
MCAST_PORT = 5007

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind((MCAST_GRP, MCAST_PORT))

positions = []
attitudes = []
ts = []
velocity_history = []
nozzle_positions = []
AoA = 0
beta = 0
mach = 0
phase = 1
lat = lat_ref
lon = lon_ref
pred_pos = np.zeros((3,))
current_sim_id = None

delta_1_history = []
delta_2_history = []
delta_3_history = []
delta_4_history = []
Bz_history = []
By_history = []

lock = threading.Lock()


def receive_data():
    global velocity_history, AoA, beta, mach, positions, pred_pos, attitudes, ts, lat, lon, nozzle_positions, phase, current_sim_id
    global delta_1_history, delta_2_history, delta_3_history, delta_4_history, Bz_history, By_history
    prev_pos = None
    prev_t = None
    
    while True:
        data, _ = sock.recvfrom(1024)
        data = np.frombuffer(data, dtype=np.float64)

        t = data[0]
        position_eci = data[1:4]
        attitude = data[7:10]
        AoA = data[20]
        beta = data[21]
        mach = data[22]
        pred_pos = data[23:26]
        By = data[18] if len(data) > 18 else 0
        Bz = data[19] if len(data) > 19 else 0
        delta_1 = data[38] if len(data) > 38 else 0
        delta_2 = data[39] if len(data) > 39 else 0
        delta_3 = data[40] if len(data) > 40 else 0
        delta_4 = data[41] if len(data) > 41 else 0
        sim_id = int(data[-2]) if len(data) > 44 else 0
        if len(data) > 40:
            phase = int(data[-1])

        current_pos = eci_to_ned(position_eci[0], position_eci[1], position_eci[2], lat_ref, lon_ref, alt_ref, t)
        current_pos[2] = -current_pos[2]
        lat, lon, alt = eci_to_geodetic(position_eci[0], position_eci[1], position_eci[2], t)
        
        if prev_pos is not None and prev_t is not None:
            dt = t - prev_t
            if dt > 0:
                velocity = np.linalg.norm(current_pos - prev_pos) / dt
            else:
                velocity = 0
        else:
            velocity = 0
        
        prev_pos = current_pos.copy()
        prev_t = t
        
        roll, pitch, yaw = attitude
        r_ned, p_ned, y_ned = eci_to_ned_attitude(roll, pitch, yaw, lat, lon, t)
        
        Rz = np.array([[np.cos(y_ned), -np.sin(y_ned), 0],
                       [np.sin(y_ned), np.cos(y_ned), 0],
                       [0, 0, 1]])
        Ry = np.array([[np.cos(-p_ned), 0, np.sin(-p_ned)], 
                       [0, 1, 0],
                       [-np.sin(-p_ned), 0, np.cos(-p_ned)]])
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(r_ned), -np.sin(r_ned)],
                       [0, np.sin(r_ned), np.cos(r_ned)]])
        R = Rz @ Ry @ Rx
        
        # Simple approach: small fixed offset for trajectory line
        nozzle_offset_body = np.array([-0.5, 0, 0])
        nozzle_offset_ned = R @ nozzle_offset_body
        nozzle_pos = current_pos + nozzle_offset_ned

        with lock:
            if current_sim_id is not None and sim_id != current_sim_id:
                positions.clear()
                attitudes.clear()
                ts.clear()
                velocity_history.clear()
                nozzle_positions.clear()
                delta_1_history.clear()
                delta_2_history.clear()
                delta_3_history.clear()
                delta_4_history.clear()
                Bz_history.clear()
                By_history.clear()
                if hasattr(receive_data, 'positions_eci'):
                    receive_data.positions_eci = []
                if hasattr(receive_data, 'altitudes'):
                    receive_data.altitudes = []
                prev_pos = None
                prev_t = None
            
            current_sim_id = sim_id
            ts.append(t)
            positions.append(current_pos)
            attitudes.append(attitude)
            velocity_history.append(velocity)
            nozzle_positions.append(nozzle_pos)
            delta_1_history.append(delta_1)
            delta_2_history.append(delta_2)
            delta_3_history.append(delta_3)
            delta_4_history.append(delta_4)
            Bz_history.append(np.rad2deg(Bz))
            By_history.append(np.rad2deg(By))
            positions_eci = getattr(receive_data, 'positions_eci', [])
            altitudes = getattr(receive_data, 'altitudes', [])
            positions_eci.append(position_eci)
            altitudes.append(alt)
            receive_data.positions_eci = positions_eci
            receive_data.altitudes = altitudes


def eci_to_ned_attitude(roll, pitch, yaw, lat, lon, t):
    LIB = body_to_eci_euler(roll, pitch, yaw)    
    LEI = eci_to_ecef(we*t)    
    LNE = ecef_to_ned_matrix(lat, lon)

    LNB = np.matmul(LNE, np.matmul(LEI, LIB))
    
    r = atan2(LNB[2, 1], LNB[2, 2])
    p = -asin(LNB[2, 0])
    y = atan2(LNB[1, 0], LNB[0, 0])
    
    return r, p, y


def create_rocket_mesh(position, attitude, phase=1, radius=0.075, height=0.5):
    """Creates rocket mesh based on phase
    Phase 1: Motor+fins, intermediate module+canards, nosecone
    Phase 2: Intermediate module+canards, nosecone
    Phase 3: Intermediate module+canards, payload pyramid
    Phase 4: payload pyramid only
    """
    roll, pitch, yaw = attitude
    
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    Ry = np.array([[np.cos(-pitch), 0, np.sin(-pitch)], 
                   [0, 1, 0],
                   [-np.sin(-pitch), 0, np.cos(-pitch)]])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    R = Rz @ Ry @ Rx
    
    n_sections = 30
    n_theta = 40
    theta = np.linspace(0, 2*np.pi, n_theta)
    
    components = []
    
    motor_height = height * 0.40
    intermediate_height = height * 0.30
    nose_height = height * 0.30
    pyramid_height = height * 0.35
    
    if phase == 1:
        nozzle_height = motor_height * 0.15
        z_nozzle = np.linspace(-height/2, -height/2 + nozzle_height, n_sections//3)
        nozzle_x = np.outer(z_nozzle, np.ones(n_theta))
        nozzle_y = radius * 0.9 * np.outer(np.ones(len(z_nozzle)), np.sin(theta))
        nozzle_z = radius * 0.9 * np.outer(np.ones(len(z_nozzle)), np.cos(theta))
        
        for i in range(nozzle_x.shape[0]):
            for j in range(nozzle_x.shape[1]):
                v = np.array([nozzle_x[i, j], nozzle_y[i, j], nozzle_z[i, j]])
                vr = R @ v + position.flatten()
                nozzle_x[i, j], nozzle_y[i, j], nozzle_z[i, j] = vr
        
        components.append(('nozzle', nozzle_x, nozzle_y, nozzle_z))
        
        z_motor = np.linspace(-height/2 + nozzle_height, -height/2 + motor_height, n_sections)
        motor_x = np.outer(z_motor, np.ones(n_theta))
        motor_y = radius * np.outer(np.ones(len(z_motor)), np.sin(theta))
        motor_z = radius * np.outer(np.ones(len(z_motor)), np.cos(theta))
        
        for i in range(motor_x.shape[0]):
            for j in range(motor_x.shape[1]):
                v = np.array([motor_x[i, j], motor_y[i, j], motor_z[i, j]])
                vr = R @ v + position.flatten()
                motor_x[i, j], motor_y[i, j], motor_z[i, j] = vr
        
        components.append(('motor', motor_x, motor_y, motor_z))
        
        intermediate_radius = radius
        z_inter_start = -height/2 + motor_height
        z_inter = np.linspace(z_inter_start, z_inter_start + intermediate_height, n_sections)
        inter_x = np.outer(z_inter, np.ones(n_theta))
        inter_y = intermediate_radius * np.outer(np.ones(len(z_inter)), np.sin(theta))
        inter_z = intermediate_radius * np.outer(np.ones(len(z_inter)), np.cos(theta))
        
        for i in range(inter_x.shape[0]):
            for j in range(inter_x.shape[1]):
                v = np.array([inter_x[i, j], inter_y[i, j], inter_z[i, j]])
                vr = R @ v + position.flatten()
                inter_x[i, j], inter_y[i, j], inter_z[i, j] = vr
        
        components.append(('intermediate', inter_x, inter_y, inter_z))
        
    elif phase == 2:
        intermediate_radius = radius
        z_inter = np.linspace(-height/2, -height/2 + intermediate_height, n_sections)
        inter_x = np.outer(z_inter, np.ones(n_theta))
        inter_y = intermediate_radius * np.outer(np.ones(len(z_inter)), np.sin(theta))
        inter_z = intermediate_radius * np.outer(np.ones(len(z_inter)), np.cos(theta))
        
        for i in range(inter_x.shape[0]):
            for j in range(inter_x.shape[1]):
                v = np.array([inter_x[i, j], inter_y[i, j], inter_z[i, j]])
                vr = R @ v + position.flatten()
                inter_x[i, j], inter_y[i, j], inter_z[i, j] = vr
        
        components.append(('intermediate', inter_x, inter_y, inter_z))
    
    elif phase == 3:
        intermediate_radius = radius
        z_inter = np.linspace(-height/2, -height/2 + intermediate_height, n_sections)
        inter_x = np.outer(z_inter, np.ones(n_theta))
        inter_y = intermediate_radius * np.outer(np.ones(len(z_inter)), np.sin(theta))
        inter_z = intermediate_radius * np.outer(np.ones(len(z_inter)), np.cos(theta))
        
        for i in range(inter_x.shape[0]):
            for j in range(inter_x.shape[1]):
                v = np.array([inter_x[i, j], inter_y[i, j], inter_z[i, j]])
                vr = R @ v + position.flatten()
                inter_x[i, j], inter_y[i, j], inter_z[i, j] = vr
        
        components.append(('intermediate', inter_x, inter_y, inter_z))
    
    nose_x, nose_y, nose_z = None, None, None
    pyramid_polys = []
    
    if phase in [1, 2]:
        nose_start = -height/2 + motor_height + intermediate_height if phase == 1 else -height/2 + intermediate_height
        nose_length = height * 0.5
        n_nose = 25
        z_nose = np.linspace(0, nose_length, n_nose)
        
        nose_x = np.outer(z_nose + nose_start, np.ones(n_theta))
        nose_y = np.zeros_like(nose_x)
        nose_z = np.zeros_like(nose_x)
        
        nose_base_radius = radius
        for i, z_val in enumerate(z_nose):
            progress = z_val / nose_length
            nose_r = nose_base_radius * (1 - progress**2.2)
            nose_y[i, :] = nose_r * np.sin(theta)
            nose_z[i, :] = nose_r * np.cos(theta)
        
        for i in range(nose_x.shape[0]):
            for j in range(nose_x.shape[1]):
                v = np.array([nose_x[i, j], nose_y[i, j], nose_z[i, j]])
                vr = R @ v + position.flatten()
                nose_x[i, j], nose_y[i, j], nose_z[i, j] = vr
    
    elif phase in [3, 4]:
        # Same pyramid in both phases, just positioned differently
        base_radius = radius * 0.7
        
        if phase == 3:
            # Phase 3: pyramid sits ON TOP of intermediate module
            pyramid_base_x = -height/2 + intermediate_height
            pyramid_apex_x = pyramid_base_x + pyramid_height
        else:
            # Phase 4: pyramid centered (same size as phase 3)
            pyramid_base_x = -pyramid_height / 2
            pyramid_apex_x = pyramid_height / 2
        
        apex = np.array([pyramid_apex_x, 0.0, 0.0])
        b1 = np.array([pyramid_base_x, -base_radius, -base_radius])
        b2 = np.array([pyramid_base_x, base_radius, -base_radius])
        b3 = np.array([pyramid_base_x, 0.0, base_radius])
        
        verts = [apex, b1, b2, b3]
        transformed = [(R @ v) + position.flatten() for v in verts]
        transformed = np.array(transformed)
        
        pyramid_polys = [
            [transformed[0], transformed[1], transformed[2]],
            [transformed[0], transformed[2], transformed[3]],
            [transformed[0], transformed[3], transformed[1]],
            [transformed[1], transformed[2], transformed[3]]
        ]
    
    fin_polys = []
    canard_polys = []
    fin_angles = [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]
    
    if phase == 1:
        for fin_angle in fin_angles:
            fin_root_x = -height/2 + motor_height*0.15
            fin_tip_x = fin_root_x + motor_height*0.3
            fin_height = radius * 2.2
            
            # Base of fin starts at motor surface
            base_y = radius * np.sin(fin_angle)
            base_z = radius * np.cos(fin_angle)
            
            fin_verts_local = np.array([
                [fin_root_x, base_y, base_z],
                [fin_tip_x, base_y, base_z],
                [fin_root_x, fin_height*np.sin(fin_angle), fin_height*np.cos(fin_angle)]
            ])
            
            fin_verts_transformed = []
            for vert in fin_verts_local:
                vr = R @ vert + position.flatten()
                fin_verts_transformed.append(vr)
            
            fin_polys.append(fin_verts_transformed)
    
    if phase in [1, 2, 3]:
        intermediate_radius = radius
        canard_base_x = -height/2 + motor_height if phase == 1 else -height/2
        canard_base_x += intermediate_height * 0.15
        
        for fin_angle in fin_angles:
            canard_root_x = canard_base_x
            canard_tip_x = canard_root_x + intermediate_height*0.35
            canard_height = intermediate_radius * 1.8
            
            # Base of canard starts at intermediate module surface
            base_y = intermediate_radius * np.sin(fin_angle)
            base_z = intermediate_radius * np.cos(fin_angle)
            
            canard_verts_local = np.array([
                [canard_root_x, base_y, base_z],
                [canard_tip_x, base_y, base_z],
                [canard_root_x, canard_height*np.sin(fin_angle), canard_height*np.cos(fin_angle)]
            ])
            
            canard_verts_transformed = []
            for vert in canard_verts_local:
                vr = R @ vert + position.flatten()
                canard_verts_transformed.append(vr)
            
            canard_polys.append(canard_verts_transformed)
    
    return components, nose_x, nose_y, nose_z, fin_polys, canard_polys, pyramid_polys


def create_terrain(ax, x_range, y_range, z_min):
    x_terrain = np.linspace(x_range[0], x_range[1], 25)
    y_terrain = np.linspace(y_range[0], y_range[1], 25)
    X, Y = np.meshgrid(x_terrain, y_terrain)
    
    Z = np.full_like(X, z_min)
    
    ax.plot_surface(X, Y, Z, cmap='terrain', alpha=0.4, linewidth=0, 
                   antialiased=True, shade=True, vmin=-0.05, vmax=0.05)
    
    for x_line in x_terrain[::3]:
        ax.plot([x_line, x_line], [y_range[0], y_range[1]], [z_min, z_min], 
               color='#7cb342', linewidth=0.6, alpha=0.5, linestyle='-')
    for y_line in y_terrain[::3]:
        ax.plot([x_range[0], x_range[1]], [y_line, y_line], [z_min, z_min], 
               color='#7cb342', linewidth=0.6, alpha=0.5, linestyle='-')
    
    ax.scatter([0], [0], [z_min], c='#ff3d00', s=80, alpha=1.0, marker='o', 
              edgecolors='#bf360c', linewidths=3, depthshade=False)


def draw_velocity_vector(ax, position, velocity_vec, scale=0.5):
    if np.linalg.norm(velocity_vec) > 1:
        vel_norm = velocity_vec / np.linalg.norm(velocity_vec) * scale
        ax.quiver(position[0], position[1], position[2],
                 vel_norm[0], vel_norm[1], vel_norm[2],
                 color='#ff6f00', arrow_length_ratio=0.3, linewidth=2.5, alpha=0.9)


def create_sky_gradient(ax):
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    
    ax.xaxis.pane.set_facecolor('#e1f5fe')
    ax.yaxis.pane.set_facecolor('#e1f5fe')
    ax.zaxis.pane.set_facecolor('#81d4fa')
    
    ax.xaxis.pane.set_edgecolor('#0277bd')
    ax.yaxis.pane.set_edgecolor('#0277bd')
    ax.zaxis.pane.set_edgecolor('#0277bd')
    
    ax.xaxis.pane.set_alpha(0.3)
    ax.yaxis.pane.set_alpha(0.3)
    ax.zaxis.pane.set_alpha(0.5)


def update_plot(fig_canard, ax1_c, ax2_c, ax3_c, fig_tvc, ax1_t, ax2_t):
    global lat, lon
    
    fig = plt.figure(figsize=(10, 8))
    ax_3d = fig.add_subplot(111, projection='3d')
    
    ax_3d.set_facecolor('#f0f8ff')
    fig.patch.set_facecolor('#ffffff')
    
    create_sky_gradient(ax_3d)
    
    ax_3d.grid(False)
    
    ax_3d.set_xlabel('North (km)', fontsize=12, color='#01579b', fontweight='bold', labelpad=10)
    ax_3d.set_ylabel('East (km)', fontsize=12, color='#01579b', fontweight='bold', labelpad=10)
    ax_3d.set_zlabel('Altitude (km)', fontsize=12, color='#01579b', fontweight='bold', labelpad=10)
    ax_3d.tick_params(colors='#0277bd', labelsize=10)
    
    azim, elev = -65, 25
    xlim, ylim, zlim = None, None, None
    first_iter = True
    last_known_sim_id = None
    
    update_counter = 0
    update_interval = 5
    
    while True:
        with lock:
            if current_sim_id != last_known_sim_id and last_known_sim_id is not None:
                first_iter = True
                xlim, ylim, zlim = None, None, None
            last_known_sim_id = current_sim_id
            
            if len(positions) > 1:
                positions_np = np.array(positions) / 1000.0
                attitudes_np = np.array(attitudes)
                ts_np = np.array(ts)
                velocities_np = np.array(velocity_history)
                nozzle_positions_np = np.array(nozzle_positions) / 1000.0
                
                if positions_np.ndim == 2:
                    current_position = positions_np[-1]
                    x_center, y_center, z_center = current_position
                    current_velocity = velocities_np[-1] if len(velocities_np) > 0 else 0
                    
                    if hasattr(receive_data, 'altitudes') and len(receive_data.altitudes) > 0:
                        altitude_km = receive_data.altitudes[-1] / 1000.0
                    else:
                        altitude_km = z_center
                    
                    if not first_iter:
                        azim = ax_3d.azim
                        elev = ax_3d.elev
                        xlim = ax_3d.get_xlim()
                        ylim = ax_3d.get_ylim()
                        zlim = ax_3d.get_zlim()
                    
                    ax_3d.clear()
                    ax_3d.set_facecolor('#f0f8ff')
                    create_sky_gradient(ax_3d)
                    ax_3d.grid(False)
                    
                    if not first_iter:
                        x_min, x_max = xlim
                        if x_center < x_min:
                            x_min = x_center
                        if x_center > x_max:
                            x_max = x_center
                        new_x_lim = [min(x_min, -3), max(x_max, 3)]

                        y_min, y_max = ylim
                        if y_center < y_min:
                            y_min = y_center
                        if y_center > y_max:
                            y_max = y_center
                        new_y_lim = [min(y_min, -3), max(y_max, 3)]

                        z_min, z_max = zlim
                        if z_center > z_max:
                            z_max = z_center
                        new_z_lim = [min(z_min, -0.5), max(z_max, 10)]
                        
                        ax_3d.set_xlim(new_x_lim)
                        ax_3d.set_ylim(new_y_lim)
                        ax_3d.set_zlim(new_z_lim)
                        ax_3d.view_init(elev=elev, azim=azim)
                    else:
                        new_x_lim = [-3, 3]
                        new_y_lim = [-3, 3]
                        new_z_lim = [-0.5, 10]
                        ax_3d.set_xlim(new_x_lim)
                        ax_3d.set_ylim(new_y_lim)
                        ax_3d.set_zlim(new_z_lim)
                        first_iter = False
                    
                    if len(nozzle_positions_np) > 1:
                        ax_3d.plot(nozzle_positions_np[:, 0], nozzle_positions_np[:, 1], nozzle_positions_np[:, 2],
                                 color='#ff6f00', linewidth=2.5, alpha=0.85, zorder=1)
                    
                    if len(nozzle_positions_np) > 1:
                        ax_3d.plot(nozzle_positions_np[:, 0], nozzle_positions_np[:, 1], 
                                 np.full_like(nozzle_positions_np[:, 2], new_z_lim[0]),
                                 color='#1976d2', linewidth=1.2, alpha=0.4, linestyle=':')
                        
                        ax_3d.plot(nozzle_positions_np[:, 0], 
                                 np.full_like(nozzle_positions_np[:, 1], new_y_lim[0]), 
                                 nozzle_positions_np[:, 2],
                                 color='#1976d2', linewidth=1.2, alpha=0.4, linestyle=':')
                        
                        ax_3d.plot(np.full_like(nozzle_positions_np[:, 0], new_x_lim[0]), 
                                 nozzle_positions_np[:, 1], 
                                 nozzle_positions_np[:, 2],
                                 color='#1976d2', linewidth=1.2, alpha=0.4, linestyle=':')
                    
                    roll, pitch, yaw = attitudes_np[-1]
                    r, p, y = eci_to_ned_attitude(roll, pitch, yaw, lat, lon, ts_np[-1])
                    att_ned = np.array([r, p, y])
                    
                    # Calculate rocket size based on axis ranges to grow with scene
                    x_span = new_x_lim[1] - new_x_lim[0]
                    y_span = new_y_lim[1] - new_y_lim[0]
                    z_span = new_z_lim[1] - new_z_lim[0]
                    avg_span = (x_span + y_span + z_span) / 3.0
                    
                    rocket_height = max(0.5, avg_span * 0.15)
                    rocket_radius = rocket_height * 0.15
                    
                    components, nose_x, nose_y, nose_z, fin_polys, canard_polys, pyramid_polys = create_rocket_mesh(
                        positions_np[-1], att_ned, phase=phase, radius=rocket_radius, height=rocket_height)
                    
                    for comp_name, comp_x, comp_y, comp_z in components:
                        if comp_name == 'nozzle':
                            surf = ax_3d.plot_surface(comp_x, comp_y, comp_z, color='#37474f', 
                                              alpha=0.98, linewidth=0, antialiased=True, 
                                              shade=True, edgecolor='none')
                            surf.set_zorder(100)
                        elif comp_name == 'motor':
                            surf = ax_3d.plot_surface(comp_x, comp_y, comp_z, color='#eceff1', 
                                              alpha=0.98, linewidth=0, antialiased=True, 
                                              shade=True, edgecolor='none')
                            surf.set_zorder(100)
                        elif comp_name == 'intermediate':
                            surf = ax_3d.plot_surface(comp_x, comp_y, comp_z, color='#78909c', 
                                              alpha=0.98, linewidth=0, antialiased=True, 
                                              shade=True, edgecolor='none')
                            surf.set_zorder(100)
                    
                    if nose_x is not None:
                        surf = ax_3d.plot_surface(nose_x, nose_y, nose_z, color='#263238', 
                                          alpha=0.98, linewidth=0, antialiased=True, 
                                          shade=True, edgecolor='none')
                        surf.set_zorder(100)
                    
                    if fin_polys:
                        fin_collection = Poly3DCollection(fin_polys, facecolors='#c62828', 
                                                         edgecolors='#b71c1c', linewidths=2, 
                                                         alpha=0.95)
                        fin_collection.set_zorder(100)
                        ax_3d.add_collection3d(fin_collection)
                    
                    if canard_polys:
                        canard_collection = Poly3DCollection(canard_polys, facecolors='#1565c0', 
                                                            edgecolors='#0d47a1', linewidths=2, 
                                                            alpha=0.95)
                        canard_collection.set_zorder(100)
                        ax_3d.add_collection3d(canard_collection)
                    
                    if pyramid_polys:
                        pyramid_collection = Poly3DCollection(pyramid_polys, facecolors="#018708", 
                                                             edgecolors="#000000", linewidths=2, 
                                                             alpha=0.95)                        
                        pyramid_collection.set_zorder(100)                        
                        ax_3d.add_collection3d(pyramid_collection)
                    
                    if len(positions_np) > 5:
                        vel_vec = positions_np[-1] - positions_np[-5]
                        draw_velocity_vector(ax_3d, positions_np[-1], vel_vec, scale=0.8)
                    
                    x_pos, y_pos, z_pos = positions_np[-1]
                    ax_3d.plot([x_pos, x_pos], [y_pos, new_y_lim[0]], [z_pos, z_pos], 
                             color='#1976d2', linewidth=1.3, alpha=0.5, linestyle='--')
                    ax_3d.scatter([x_pos], [new_y_lim[0]], [z_pos], c='#1976d2', s=14, 
                                alpha=0.7, edgecolors='#0d47a1', linewidths=0.8)
                    
                    ax_3d.plot([x_pos, new_x_lim[0]], [y_pos, y_pos], [z_pos, z_pos], 
                             color='#1976d2', linewidth=1.3, alpha=0.5, linestyle='--')
                    ax_3d.scatter([new_x_lim[0]], [y_pos], [z_pos], c='#1976d2', s=14, 
                                alpha=0.7, edgecolors='#0d47a1', linewidths=0.8)
                    
                    ax_3d.plot([x_pos, x_pos], [y_pos, y_pos], [z_pos, new_z_lim[0]], 
                             color='#1976d2', linewidth=1.3, alpha=0.5, linestyle='--')
                    ax_3d.scatter([x_pos], [y_pos], [new_z_lim[0]], c='#1976d2', s=14, 
                                alpha=0.7, edgecolors='#0d47a1', linewidths=0.8)
                    
                    ax_3d.scatter(pred_pos[0] / 1000.0, pred_pos[1] / 1000.0, -pred_pos[2] / 1000.0, 
                                c='#7b1fa2', s=90, alpha=0.95, marker='D', 
                                edgecolors='#4a148c', linewidths=2.5, depthshade=False)
                    
                    ax_3d.set_xlabel('North (km)', fontsize=12, color='#01579b', fontweight='bold', labelpad=10)
                    ax_3d.set_ylabel('East (km)', fontsize=12, color='#01579b', fontweight='bold', labelpad=10)
                    ax_3d.set_zlabel('Altitude (km)', fontsize=12, color='#01579b', fontweight='bold', labelpad=10)
                    ax_3d.tick_params(colors='#0277bd', labelsize=10)
                    
                    current_time = ts_np[-1] if len(ts_np) > 0 else 0
                    
                    pos_x_km = current_position[0]
                    pos_y_km = current_position[1]
                    pos_z_km = current_position[2]
                    
                    downrange = np.sqrt(pos_x_km**2 + pos_y_km**2)
                    
                    phase_names = {1: "PHASE 1", 2: "PHASE 2", 
                                   3: "PHASE 3", 4: "PHASE 4"}
                    phase_name = phase_names.get(phase, "UNKNOWN")
                    
                    info_text = (f"╔═════════════════════════╗\n"
                                f"│ {phase_name:23} │\n"
                                f"╠═════════════════════════╣\n"
                                f"│ FLIGHT TELEMETRY        │\n"
                                f"╠═════════════════════════╣\n"
                                f"  Time:      {current_time:7.1f} s\n"
                                f"  Altitude:  {altitude_km:7.2f} km\n"
                                f"  Downrange: {downrange:7.2f} km\n"
                                f"  Velocity:  {current_velocity:7.1f} m/s\n"
                                f"  Mach:      {mach:7.2f}\n"
                                f"╠═════════════════════════╣\n"
                                f"│ ATTITUDE                │\n"
                                f"╠═════════════════════════╣\n"
                                f"  Roll:      {np.rad2deg(att_ned[0]):7.1f}°\n"
                                f"  Pitch:     {np.rad2deg(att_ned[1]):7.1f}°\n"
                                f"  Yaw:       {np.rad2deg(att_ned[2]):7.1f}°\n"
                                f"╠═════════════════════════╣\n"
                                f"│ AERODYNAMICS            │\n"
                                f"╠═════════════════════════╣\n"
                                f"  AoA:       {AoA:7.1f}°\n"
                                f"  Sideslip:  {beta:7.1f}°\n"
                                f"╠═════════════════════════╣\n"
                                f"│ POSITION (NED)          │\n"
                                f"╠═════════════════════════╣\n"
                                f"  North:     {pos_x_km:7.2f} km\n"
                                f"  East:      {pos_y_km:7.2f} km\n"
                                f"  Up:        {pos_z_km:7.2f} km\n"
                                f"╚═════════════════════════╝")
                    
                    ax_3d.text2D(0.02, 0.98, info_text, transform=ax_3d.transAxes, 
                               fontsize=8.5, color='#01579b', verticalalignment='top',
                               bbox=dict(boxstyle='round,pad=0.7', facecolor='#ffffff', 
                                       edgecolor='#0288d1', alpha=0.97, linewidth=2.5),
                               fontfamily='monospace', weight='bold', linespacing=1.3)
                    
                    x_range = ax_3d.get_xlim()
                    y_range = ax_3d.get_ylim()
                    z_range = ax_3d.get_zlim()
                    
                    x_span = x_range[1] - x_range[0]
                    y_span = y_range[1] - y_range[0]
                    z_span = z_range[1] - z_range[0]
                    
                    ax_3d.set_box_aspect([x_span, y_span, z_span])
        
        update_counter += 1
        should_update_plots = (update_counter % update_interval == 0)
        
        if should_update_plots and len(ts) > 0 and len(delta_1_history) > 0:
            t_data = np.array(ts)
            delta_1 = np.array(delta_1_history)
            delta_2 = np.array(delta_2_history)
            delta_3 = np.array(delta_3_history)
            delta_4 = np.array(delta_4_history)
            
            min_len = min(len(t_data), len(delta_1), len(delta_2), len(delta_3), len(delta_4))
            t_data = t_data[:min_len]
            delta_1 = delta_1[:min_len]
            delta_2 = delta_2[:min_len]
            delta_3 = delta_3[:min_len]
            delta_4 = delta_4[:min_len]
            
            delta_p = 0.25 * (-delta_1 - delta_2 + delta_3 + delta_4)
            delta_q = 0.25 * (delta_1 + delta_2 + delta_3 + delta_4)
            delta_r = 0.25 * (-delta_1 + delta_2 - delta_3 + delta_4)
            
            ax1_c.clear()
            ax2_c.clear()
            ax3_c.clear()
            
            ax1_c.plot(t_data, delta_p, color='#d32f2f', linewidth=2.0, label='δp')
            ax2_c.plot(t_data, delta_q, color='#1976d2', linewidth=2.0, label='δq')
            ax3_c.plot(t_data, delta_r, color='#388e3c', linewidth=2.0, label='δr')
            
            for ax in [ax1_c, ax2_c, ax3_c]:
                ax.set_facecolor('#f0f8ff')
                ax.grid(True, alpha=0.3, color='#0277bd', linestyle='--', linewidth=0.8)
                ax.tick_params(colors='#0277bd', labelsize=10)
                ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
            
            ax1_c.set_ylabel('δp (deg)', fontsize=11, color='#01579b', fontweight='bold')
            ax2_c.set_ylabel('δq (deg)', fontsize=11, color='#01579b', fontweight='bold')
            ax3_c.set_ylabel('δr (deg)', fontsize=11, color='#01579b', fontweight='bold')
            ax3_c.set_xlabel('Time (s)', fontsize=11, color='#01579b', fontweight='bold')
            
            fig_canard.canvas.draw_idle()
        
        if should_update_plots and len(ts) > 0 and len(Bz_history) > 0:
            t_data = np.array(ts)
            delta_q_data = np.array(Bz_history)
            delta_r_data = np.array(By_history)
            
            min_len = min(len(t_data), len(delta_q_data), len(delta_r_data))
            t_data = t_data[:min_len]
            delta_q_data = delta_q_data[:min_len]
            delta_r_data = delta_r_data[:min_len]
            
            ax1_t.clear()
            ax2_t.clear()
            
            ax1_t.plot(t_data, delta_q_data, color='#7b1fa2', linewidth=2.0, label='δq')
            ax2_t.plot(t_data, delta_r_data, color='#f57c00', linewidth=2.0, label='δr')
            
            for ax in [ax1_t, ax2_t]:
                ax.set_facecolor('#f0f8ff')
                ax.grid(True, alpha=0.3, color='#0277bd', linestyle='--', linewidth=0.8)
                ax.tick_params(colors='#0277bd', labelsize=10)
                ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
            
            ax1_t.set_ylabel('δq (deg)', fontsize=11, color='#01579b', fontweight='bold')
            ax2_t.set_ylabel('δr (deg)', fontsize=11, color='#01579b', fontweight='bold')
            ax2_t.set_xlabel('Time (s)', fontsize=11, color='#01579b', fontweight='bold')
            
            fig_tvc.canvas.draw_idle()

        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)


def main():
    plt.ion()
    
    fig_canard, (ax1_c, ax2_c, ax3_c) = plt.subplots(3, 1, figsize=(10, 8))
    fig_canard.patch.set_facecolor('#ffffff')
    for ax in [ax1_c, ax2_c, ax3_c]:
        ax.set_facecolor('#f0f8ff')
        ax.grid(True, alpha=0.3, color='#0277bd', linestyle='--', linewidth=0.8)
        ax.tick_params(colors='#0277bd', labelsize=10)
    ax1_c.set_ylabel('δp (deg)', fontsize=11, color='#01579b', fontweight='bold')
    ax2_c.set_ylabel('δq (deg)', fontsize=11, color='#01579b', fontweight='bold')
    ax3_c.set_ylabel('δr (deg)', fontsize=11, color='#01579b', fontweight='bold')
    ax3_c.set_xlabel('Time (s)', fontsize=11, color='#01579b', fontweight='bold')
    fig_canard.suptitle('Canard Control Deflections', fontsize=14, color='#01579b', fontweight='bold')
    plt.tight_layout()
    
    fig_tvc, (ax1_t, ax2_t) = plt.subplots(2, 1, figsize=(10, 6))
    fig_tvc.patch.set_facecolor('#ffffff')
    for ax in [ax1_t, ax2_t]:
        ax.set_facecolor('#f0f8ff')
        ax.grid(True, alpha=0.3, color='#0277bd', linestyle='--', linewidth=0.8)
        ax.tick_params(colors='#0277bd', labelsize=10)
    ax1_t.set_ylabel('δq (deg)', fontsize=11, color='#01579b', fontweight='bold')
    ax2_t.set_ylabel('δr (deg)', fontsize=11, color='#01579b', fontweight='bold')
    ax2_t.set_xlabel('Time (s)', fontsize=11, color='#01579b', fontweight='bold')
    fig_tvc.suptitle('TVC Deflections', fontsize=14, color='#01579b', fontweight='bold')
    plt.tight_layout()
    
    plt.show(block=False)
    
    threading.Thread(target=receive_data, daemon=True).start()
    update_plot(fig_canard, ax1_c, ax2_c, ax3_c, fig_tvc, ax1_t, ax2_t)


if __name__ == '__main__':
    main()
