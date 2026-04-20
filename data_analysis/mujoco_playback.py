"""
Run with mjpython (called automatically by data_analysis.py):
    mjpython data_analysis/mujoco_playback.py --file telemetry/Apr18_r90_1.csv

Speed control: type a number + Enter in the terminal while running (e.g. '0.25' for slow-mo).
Pass --save_video PATH to render headlessly and write an mp4 instead of opening the viewer.
"""
import os
import sys
import argparse
import time
import threading
import numpy as np
import pandas as pd
import mujoco
import mujoco.viewer

f_quat = ['F_Q0', 'F_Q1', 'F_Q2', 'F_Q3']
b_quat = ['B_Q0', 'B_Q1', 'B_Q2', 'B_Q3']

def _speed_controller(speed):
    print("Speed control: type a multiplier + Enter (e.g. '0.25' for slow-mo, '2' for 2x)")
    while True:
        try:
            val = float(input())
            speed[0] = max(0.05, val)
            print(f"  → speed: {speed[0]}x")
        except (ValueError, EOFError):
            pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True, help='Path to telemetry CSV')
    parser.add_argument('--end_idx', type=int, default=None, help='Last frame index to play')
    parser.add_argument('--save_video', default=None, help='If set, render headlessly and save to this path (.mp4)')
    args = parser.parse_args()

    df = pd.read_csv(args.file)
    end_idx = args.end_idx if args.end_idx is not None else len(df)

    front_quats = df[f_quat].iloc[0:end_idx].to_numpy()
    rear_quats  = df[b_quat].iloc[0:end_idx].to_numpy()
    joints = {
        "rot1":  df["F_M1"].iloc[0:end_idx].to_numpy(),
        "pitch": df["F_M2"].iloc[0:end_idx].to_numpy(),
        "rot2":  df["B_M2"].iloc[0:end_idx].to_numpy(),
        "tail":  df["B_M1"].iloc[0:end_idx].to_numpy(),
    }

    # Resolve model path relative to repo root (parent of this file's directory)
    repo_root  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(repo_root, "model", "cat_viz.xml")

    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data  = mujoco.MjData(mj_model)
    mj_model.opt.gravity[:] = [0, 0, 0]

    ghost_body_id  = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "rear_ghost")
    ghost_mocap_id = mj_model.body_mocapid[ghost_body_id]
    real_rear_id   = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "rear_body")

    joint_names = ["rot1", "pitch", "rot2", "tail"]
    jnt_idx = {n: mj_model.jnt_qposadr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, n)]
               for n in joint_names}

    def _set_frame(i):
        mj_data.qpos[0:3] = [0, 0, 3]
        mj_data.qpos[3:7] = front_quats[i]
        for n in joint_names:
            mj_data.qpos[jnt_idx[n]] = joints[n][i]
        mujoco.mj_kinematics(mj_model, mj_data)
        mj_data.mocap_pos[ghost_mocap_id]  = mj_data.xpos[real_rear_id]
        mj_data.mocap_quat[ghost_mocap_id] = rear_quats[i]

    if args.save_video:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0, 0, 3]
        cam.distance  = 2.0
        cam.azimuth   = 90.0
        cam.elevation = -20.0

        mj_model.vis.global_.offwidth  = 1280
        mj_model.vis.global_.offheight = 720
        import cv2
        os.makedirs(os.path.dirname(os.path.abspath(args.save_video)), exist_ok=True)
        with mujoco.Renderer(mj_model, height=720, width=1280) as renderer:
            writer = cv2.VideoWriter(
                args.save_video,
                cv2.VideoWriter_fourcc(*'mp4v'),
                50, (1280, 720),
            )
            for i in range(end_idx):
                _set_frame(i)
                renderer.update_scene(mj_data, camera=cam)
                frame = renderer.render()
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
        print(f"Video saved to {args.save_video}")

    else:
        speed = [0.1]
        # threading.Thread(target=_speed_controller, args=(speed,), daemon=True).start()

        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            viewer.cam.lookat[:] = [0, 0, 3]
            viewer.cam.distance  = 2.0
            viewer.cam.azimuth   = 90
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD

            while viewer.is_running():
                for i in range(end_idx):
                    if not viewer.is_running():
                        break
                    _set_frame(i)
                    viewer.sync()
                    time.sleep(1.0 / (50 * speed[0]))

if __name__ == '__main__':
    main()
