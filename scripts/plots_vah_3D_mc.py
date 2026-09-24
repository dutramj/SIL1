import socket
import threading
import time
import random
from math import atan2, asin

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from vahsimulator.utils import eci_to_ned, eci_to_geodetic, body_to_eci_euler, eci_to_ecef, ecef_to_ned_matrix
from vahsimulator.parameters import lat_ref, lon_ref, alt_ref, we

mpl.rcParams['axes.formatter.useoffset'] = False
mpl.rcParams['axes.formatter.limits'] = (-99, 99)

MCAST_GRP = '127.0.0.1'
MCAST_PORT = 5007

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind((MCAST_GRP, MCAST_PORT))

lat = lat_ref
lon = lon_ref
current_sim_id = None
last_update_time = {}
active_simulations = {}
lock = threading.Lock()
INACTIVITY_TIMEOUT = 3.0


def receive_data():
    global lat, lon, current_sim_id, last_update_time, active_simulations
    
    while True:
        data, _ = sock.recvfrom(1024)
        data = np.frombuffer(data, dtype=np.float64)

        t = data[0]
        position_eci = data[1:4]
        attitude = data[7:10]
        sim_id = int(data[-2]) if len(data) > 44 else 0
        phase = int(data[-1]) if len(data) > 44 else 1
        
        current_time = time.time()
        lat_current, lon_current, alt = eci_to_geodetic(position_eci[0], position_eci[1], position_eci[2], t)
        lat = lat_current
        lon = lon_current
        
        current_pos = eci_to_ned(position_eci[0], position_eci[1], position_eci[2], lat_ref, lon_ref, alt_ref, t)
        current_pos[2] = -current_pos[2]
        
        with lock:
            if sim_id not in active_simulations:
                active_simulations[sim_id] = {
                    'positions': [],
                    'attitudes': [],
                    'geo_positions': [],
                    'ts': [],
                    'velocity_history': [],
                    'positions_eci': [],
                    'prev_pos': None,
                    'prev_t': None,
                    'phase': phase
                }
            
            sim_data = active_simulations[sim_id]
            last_update_time[sim_id] = current_time
            sim_data['phase'] = phase
            
            if sim_data['prev_pos'] is not None and sim_data['prev_t'] is not None:
                dt = t - sim_data['prev_t']
                velocity = np.linalg.norm(current_pos - sim_data['prev_pos']) / dt if dt > 0 else 0
            else:
                velocity = 0
            
            sim_data['prev_pos'] = current_pos.copy()
            sim_data['prev_t'] = t
            
            # Converter atitude ECI para NED
            roll, pitch, yaw = attitude
            r_ned, p_ned, y_ned = eci_to_ned_attitude(roll, pitch, yaw, lat_current, lon_current, t)
            
            sim_data['positions'].append(current_pos)
            sim_data['attitudes'].append(np.array([r_ned, p_ned, y_ned]))
            sim_data['geo_positions'].append(np.array([lat_current, lon_current, alt]))
            sim_data['ts'].append(t)
            sim_data['velocity_history'].append(velocity)
            sim_data['positions_eci'].append(position_eci)


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
        base_radius = radius * 0.7
        
        if phase == 3:
            pyramid_base_x = -height/2 + intermediate_height
            pyramid_apex_x = pyramid_base_x + pyramid_height
        else:
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


def update_plot():
    global lat, lon, current_sim_id, active_simulations, last_update_time
    
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
    
    azim, elev = -60, 30
    
    # Extended color palette for more variety
    base_colors = [
        '#ff6f00', '#1976d2', '#388e3c', '#d32f2f', '#7b1fa2', '#f57c00', '#0288d1', '#689f38',
        '#e91e63', '#9c27b0', '#673ab7', '#3f51b5', '#00bcd4', '#009688', '#4caf50', '#8bc34a',
        '#cddc39', '#ffeb3b', '#ffc107', '#ff9800', '#ff5722', '#795548', '#607d8b', '#f06292'
    ]
    random.shuffle(base_colors)
    colors = base_colors
    sim_colors = {}
    first_iter = True
    
    while True:
        with lock:
            current_time = time.time()
            completed_sims = []
            
            for sim_id in list(active_simulations.keys()):
                if current_time - last_update_time.get(sim_id, 0) > INACTIVITY_TIMEOUT:
                    completed_sims.append(sim_id)
            
            for sim_id in completed_sims:
                if sim_id in active_simulations:
                    del active_simulations[sim_id]
                if sim_id in last_update_time:
                    del last_update_time[sim_id]
                if sim_id in sim_colors:
                    del sim_colors[sim_id]
            
            if not active_simulations:
                plt.draw()
                plt.pause(0.5)
                continue
            
            if not first_iter:
                azim = ax_3d.azim
                elev = ax_3d.elev
            
            ax_3d.cla()
            ax_3d.set_facecolor('#f0f8ff')
            create_sky_gradient(ax_3d)
            ax_3d.grid(False)
            
            ax_3d.set_xlabel('North (km)', fontsize=12, color='#01579b', fontweight='bold', labelpad=10)
            ax_3d.set_ylabel('East (km)', fontsize=12, color='#01579b', fontweight='bold', labelpad=10)
            ax_3d.set_zlabel('Altitude (km)', fontsize=12, color='#01579b', fontweight='bold', labelpad=10)
            ax_3d.tick_params(colors='#0277bd', labelsize=10)
            
            # Calculate max ranges across all active simulations
            max_x, max_y, max_z = 3, 3, 10
            min_x, min_y, min_z = -3, -3, -0.5
            
            for sim_id, sim_data in active_simulations.items():
                if len(sim_data['positions']) > 1:
                    positions_np = np.array(sim_data['positions']) / 1000.0
                    max_x = max(max_x, np.max(positions_np[:, 0]) + 1)
                    max_y = max(max_y, np.max(positions_np[:, 1]) + 1)
                    max_z = max(max_z, np.max(positions_np[:, 2]) + 1)
                    min_x = min(min_x, np.min(positions_np[:, 0]) - 1)
                    min_y = min(min_y, np.min(positions_np[:, 1]) - 1)
                    min_z = min(min_z, np.min(positions_np[:, 2]) - 1)
            
            # Set axis limits
            ax_3d.set_xlim([min_x, max_x])
            ax_3d.set_ylim([min_y, max_y])
            ax_3d.set_zlim([min_z, max_z])
            
            # Calculate rocket size based on axis ranges
            x_span = max_x - min_x
            y_span = max_y - min_y
            z_span = max_z - min_z
            avg_span = (x_span + y_span + z_span) / 3.0
            
            rocket_height = max(0.5, avg_span * 0.15)
            rocket_radius = rocket_height * 0.15
            
            # Set aspect ratio
            ax_3d.set_box_aspect([x_span, y_span, z_span])
            
            if not first_iter:
                ax_3d.view_init(elev=elev, azim=azim)
            else:
                first_iter = False
            
            for idx, (sim_id, sim_data) in enumerate(active_simulations.items()):
                positions = sim_data['positions']
                attitudes = sim_data['attitudes']
                
                if len(positions) > 1:
                    positions_np = np.array(positions) / 1000.0
                    attitudes_np = np.array(attitudes)
                    current_phase = sim_data['phase']
                    
                    # Assign persistent colors to this simulation
                    if sim_id not in sim_colors:
                        sim_colors[sim_id] = {
                            'trajectory': colors[len(sim_colors) % len(colors)],
                            'nozzle': random.choice(colors),
                            'motor': random.choice(colors),
                            'intermediate': random.choice(colors),
                            'nose': random.choice(colors),
                            'pyramid': random.choice(colors),
                            'fins': random.choice(colors),
                            'canards': random.choice(colors)
                        }
                    
                    rocket_colors = sim_colors[sim_id]
                    
                    # Draw trajectory line
                    ax_3d.plot(positions_np[:, 0], positions_np[:, 1], positions_np[:, 2],
                             color=rocket_colors['trajectory'], linewidth=2.0, alpha=0.7)
                    
                    # Draw current rocket at latest position
                    current_position = positions_np[-1]
                    current_attitude = attitudes_np[-1]
                    
                    components, nose_x, nose_y, nose_z, fin_polys, canard_polys, pyramid_polys = create_rocket_mesh(
                        current_position, current_attitude, phase=current_phase, 
                        radius=rocket_radius, height=rocket_height
                    )
                    
                    # Draw body components
                    for comp_name, comp_x, comp_y, comp_z in components:
                        if comp_name == 'nozzle':
                            ax_3d.plot_surface(comp_x, comp_y, comp_z, color=rocket_colors['nozzle'], 
                                             alpha=0.95, linewidth=0, shade=True)
                        elif comp_name == 'motor':
                            ax_3d.plot_surface(comp_x, comp_y, comp_z, color=rocket_colors['motor'], 
                                             alpha=0.95, linewidth=0, shade=True)
                        elif comp_name == 'intermediate':
                            ax_3d.plot_surface(comp_x, comp_y, comp_z, color=rocket_colors['intermediate'], 
                                             alpha=0.95, linewidth=0, shade=True)
                    
                    # Draw nose cone
                    if nose_x is not None:
                        ax_3d.plot_surface(nose_x, nose_y, nose_z, color=rocket_colors['nose'], 
                                         alpha=0.95, linewidth=0, shade=True)
                    
                    # Draw pyramid
                    if pyramid_polys:
                        pyramid_collection = Poly3DCollection(pyramid_polys, facecolors=rocket_colors['pyramid'], 
                                                             linewidths=0.5, edgecolors='k', alpha=0.95)
                        ax_3d.add_collection3d(pyramid_collection)
                    
                    # Draw fins
                    if fin_polys:
                        for fin_verts in fin_polys:
                            fin_collection = Poly3DCollection([fin_verts], facecolors=rocket_colors['fins'], 
                                                            linewidths=0.5, edgecolors='k', alpha=0.95)
                            ax_3d.add_collection3d(fin_collection)
                    
                    # Draw canards
                    if canard_polys:
                        for canard_verts in canard_polys:
                            canard_collection = Poly3DCollection([canard_verts], facecolors=rocket_colors['canards'], 
                                                               linewidths=0.5, edgecolors='k', alpha=0.95)
                            ax_3d.add_collection3d(canard_collection)
            
            active_count = len(active_simulations)
            info_text = (f"╔═══════════════════════════════╗\n"
                        f"│ MONTE CARLO SIMULATION        │\n"
                        f"╠═══════════════════════════════╣\n"
                        f"  Active Trajectories: {active_count:3d}\n"
                        f"╚═══════════════════════════════╝")
            
            ax_3d.text2D(0.02, 0.98, info_text, transform=ax_3d.transAxes, 
                       fontsize=10, color='#01579b', verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.7', facecolor='#ffffff', 
                               edgecolor='#0288d1', alpha=0.97, linewidth=2.5),
                       fontfamily='monospace', weight='bold', linespacing=1.3)

        plt.tight_layout()
        plt.draw()
        pause_time = 0.5 * max(1, active_count)
        plt.pause(pause_time)


def main():
    threading.Thread(target=receive_data, daemon=True).start()
    update_plot()


if __name__ == '__main__':
    main()
