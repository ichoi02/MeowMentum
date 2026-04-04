import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
import time
import os

def load_telemetry_data():
    """
    Dummy telemtry data
    """
    num_steps = 500
    t = np.linspace(0, 10, num_steps)
    
    # Mock IMU Quaternions for front_body (MuJoCo uses [w, x, y, z])
    # Just making it wobble slightly for the visual
    w = np.cos(t)
    x = np.sin(t) * 0.2
    y = np.zeros_like(t)
    z = np.zeros_like(t)
    norms = np.sqrt(w**2 + x**2 + y**2 + z**2)
    front_quats = np.column_stack((w/norms, x/norms, y/norms, z/norms))
    rear_quats = np.column_stack((w/norms, (x+0.1)/norms, y/norms, z/norms))
    
    # Mock Encoder Data
    rot1 = np.sin(t * 2) * 1.5
    pitch = np.cos(t * 3) * 0.5
    rot2 = np.sin(t * 2 + 1) * 1.5
    tail = np.sin(t * 4) * 1.0
    
    return {
        "front_quat": front_quats,
        "rear_quat": rear_quats,
        "rot1": rot1,
        "pitch": pitch,
        "rot2": rot2,
        "tail": tail
    }

def load_telemetry_csv(filepath):
    columns = [
        "Time", "F_Q0", "F_Q1", "F_Q2", "F_Q3", "F_M1", "F_M2", "Cmd_F1", "Cmd_F2", 
        "B_Q0", "B_Q1", "B_Q2", "B_Q3", "B_M1", "B_M2", "Cmd_B1", "Cmd_B2"
    ]
    
    df = pd.read_csv(filepath, usecols=columns)
    
    front_quats = df[["F_Q0", "F_Q1", "F_Q2", "F_Q3"]].to_numpy()
    rear_quats = df[["B_Q0", "B_Q1", "B_Q2", "B_Q3"]].to_numpy()
    rot1 = df["F_M1"].to_numpy()
    pitch = df["F_M2"].to_numpy()
    rot2 = df["B_M1"].to_numpy()
    tail = df["B_M2"].to_numpy()
    time = df["Time"].to_numpy()
    
    return {
        "time": time,
        "front_quat": front_quats,
        "rear_quat": rear_quats,
        "rot1": rot1,
        "pitch": pitch,
        "rot2": rot2,
        "tail": tail
    }

def visualize():
    model_path = os.path.abspath("model/cat_viz.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Rear body ghost viz
    ghost_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rear_ghost")
    ghost_mocap_id = model.body_mocapid[ghost_body_id]
    real_rear_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rear_body")
    
    # Disable gravity
    model.opt.gravity[:] = [0, 0, 0]

    joint_names = ["rot1", "pitch", "rot2", "tail"]
    joint_qpos_idx = {
        name: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] 
        for name in joint_names
    }

    # Load telemetry
    telemetry = load_telemetry_csv("/Users/itak/Documents/CMU/24775_Bioinspired_Robot/MeowMentum/telemetry/telemetry_1775332381.csv")
    num_steps = len(telemetry["front_quat"])
    fps = 100
    
    print("Launching viewer.")
    
    # Launch the passive viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0, 0, 3]
        viewer.cam.distance = 2.0
        
        for step in range(num_steps):
            if not viewer.is_running():
                break
            
            # Root position
            data.qpos[0:3] = [0, 0, 3] 
            
            # Root orientation: quat - (w, x, y, z)
            data.qpos[3:7] = telemetry["front_quat"][step]
            
            # Joint angles
            data.qpos[joint_qpos_idx["rot1"]] = telemetry["rot1"][step]
            data.qpos[joint_qpos_idx["pitch"]] = telemetry["pitch"][step]
            data.qpos[joint_qpos_idx["rot2"]] = telemetry["rot2"][step]
            data.qpos[joint_qpos_idx["tail"]] = telemetry["tail"][step]
            
            # Update kinematics (no physics)
            mujoco.mj_kinematics(model, data)

            # Snap the ghost's position to the simulated rear body's position
            data.mocap_pos[ghost_mocap_id] = data.xpos[real_rear_body_id]
            # Set the ghost's orientation purely from the rear IMU telemetry
            data.mocap_quat[ghost_mocap_id] = telemetry["rear_quat"][step]
            
            viewer.sync()
            time.sleep(1.0 / fps)

if __name__ == "__main__":
    visualize()