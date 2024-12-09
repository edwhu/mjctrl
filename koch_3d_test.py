import mujoco
import mujoco.viewer
import numpy as np
import time

# Integration timestep in seconds. This corresponds to the amount of time the joint
# velocities will be integrated for to obtain the desired joint positions.
integration_dt: float = 0.1

# Damping term for the pseudoinverse. This is used to prevent joint velocities from
# becoming too large when the Jacobian is close to singular.
damping: float = 1e-4

# Gains for the twist computation. These should be between 0 and 1. 0 means no
# movement, 1 means move the end-effector to the target in one integration step.
Kpos: float = 0.95
Kori: float = 0.95

# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Simulation timestep in seconds.
dt: float = 0.002

# Nullspace P gain.
Kn = np.asarray([10.0, 10.0, 10.0, 10.0, 5.0, 0.0])
Kn /= 10.0

# Maximum allowable joint velocity in rad/s.
max_angvel = 0.785


def diffik_nullspace(
    model,
    data,
    goal_pos,
    goal_quat,
    site_id,
    twist,
    site_quat,
    site_quat_conj,
    error_quat,
    eye,
    jac,
    diag,
    q0,
    dof_ids,
):
    # Spatial velocity (aka twist).
    # dx = data.mocap_pos[mocap_id] - data.site(site_id).xpos
    dx = goal_pos - data.site(site_id).xpos
    twist[:3] = Kpos * dx / integration_dt
    mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
    mujoco.mju_negQuat(site_quat_conj, site_quat)
    # mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
    mujoco.mju_mulQuat(error_quat, goal_quat, site_quat_conj)
    mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
    twist[3:] *= Kori / integration_dt

    # Jacobian.
    mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)

    # Damped least squares.
    dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, twist)

    # Nullspace control biasing joint velocities towards the home configuration.
    dq += (eye - np.linalg.pinv(jac) @ jac) @ (Kn * (q0 - data.qpos[dof_ids]))

    # Clamp maximum joint velocity.
    dq_abs_max = np.abs(dq).max()
    if dq_abs_max > max_angvel:
        dq *= max_angvel / dq_abs_max

    # Integrate joint velocities to obtain joint positions.
    q = data.qpos.copy()  # Note the copy here is important.
    mujoco.mj_integratePos(model, q, dq, integration_dt)
    np.clip(q, *model.jnt_range.T, out=q)
    return q


def main() -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    # Load the model and data.
    model = mujoco.MjModel.from_xml_path("low_cost_robot_6dof/reach_cube.xml")
    data = mujoco.MjData(model)

    # Enable gravity compensation. Set to 0.0 to disable.
    model.body_gravcomp[:] = float(gravity_compensation)
    model.opt.timestep = dt

    # End-effector site we wish to control.
    site_name = "attachment_site"
    site_id = model.site(site_name).id

    # Get the dof and actuator ids for the joints we wish to control. These are copied
    # from the XML file. Feel free to comment out some joints to see the effect on
    # the controller.
    joint_names = [
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",
    ]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])

    # Initial joint configuration saved as a keyframe in the XML file.
    key_name = "home"
    key_id = model.key(key_name).id
    q0 = model.key(key_name).qpos

    # Mocap body we will control with our mouse.
    mocap_name = "target"
    mocap_id = model.body(mocap_name).mocapid[0]

    # Pre-allocate numpy arrays.
    jac = np.zeros((6, model.nv))
    diag = damping * np.eye(6)
    eye = np.eye(model.nv)
    twist = np.zeros(6)
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)

    bottom_left = [-0.06, -0.10, 0.015]
    bottom_right = [-0.06, 0.10, 0.015]
    top_left = [-0.14, -0.10, 0.015]
    top_right = [-0.14, 0.10, 0.015]
    goals = [bottom_left, bottom_right, top_right, top_left]

    bottom_left = [-0.06, -0.10, 0.05]
    bottom_right = [-0.06, 0.10, 0.05]
    top_left = [-0.14, -0.10, 0.05]
    top_right = [-0.14, 0.10, 0.05]
    goals.extend([top_left, top_right, bottom_right, bottom_left])



    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        # Reset the simulation.
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        # Reset the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Enable site frame visualization.
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        # set goal quat to just the initial arm quat
        goal_quat = data.mocap_quat[mocap_id]
        steps = 0
        goal_interval = 500

        while viewer.is_running():
            step_start = time.time()

            # Spatial velocity (aka twist).
            if steps % goal_interval == 0:
                goal_pos = goals[ (steps // goal_interval) % len(goals)]   

            q = diffik_nullspace(
                model,
                data,
                goal_pos,
                goal_quat,
                site_id,
                twist,
                site_quat,
                site_quat_conj,
                error_quat,
                eye,
                jac,
                diag,
                q0,
                dof_ids,
            )

            # Set the control signal and step the simulation.
            data.ctrl[actuator_ids] = q[dof_ids]
            mujoco.mj_step(model, data)
            print(f"t: {steps} goal: {goal_pos}, arm: {data.site(site_id).xpos}",end='\r')
            viewer.sync()
            steps += 1
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
