import mujoco
import numpy as np
import time
from mujoco import viewer


def quaternion_to_euler(quat):
    """
    Convert a quaternion to Euler angles.

    Parameters:
    quat (numpy.ndarray): A quaternion (x, y, z, w) where w is the scalar part.

    Returns:
    numpy.ndarray: Euler angles (roll, pitch, yaw) in radians.
    """
    # Ensure the quaternion is normalized
    quat = quat / np.linalg.norm(quat)

    # Extract the components of the quaternion
    x, y, z, w = quat

    # Compute the Euler angles
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)

    return np.array([np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw)])


class MujocoEnvironment:
    def __init__(self, render: bool = False):
        self.render = render
        self.viewer = None
        self.model = None
        self.data = None

    def reset(self):
        x_object_size = np.random.uniform(0.1, 0.4)
        y_object_size = np.random.uniform(x_object_size*0.3, x_object_size*1.3)
        z_object_size = np.random.uniform(x_object_size*2.5, x_object_size*5)
        x_object_pos = np.random.uniform(-0.6, 0.6)
        z_object_pos = 2*z_object_size
        density = np.random.uniform(0.1, 0.5)

        kp_slide = np.random.uniform(500, 1000)
        kv_slide = np.random.uniform(kp_slide/6, kp_slide/3)
        while True:
            self.event_1_target = np.random.uniform(-3, 3)
            if np.linalg.norm(self.event_1_target) > 1:
                break
        while True:
            self.event_2_target = np.random.uniform(-3, 3)
            if np.linalg.norm(self.event_2_target - self.event_1_target) > 2:
                break

        xml = """
        <mujoco model="balance_cart">
            <compiler angle="degree" />
            <option timestep=".01"/>

            <worldbody>
                <light pos="0 -.4 1"/>

                <!-- Define the box body -->
                <body name="box" pos="0 0 0.05">
                    <!-- Define the box geometry -->
                    <geom name="box_geom" type="box" size="1 0.5 0.05" rgba="0.8 0.3 0.3 1" pos="0 0 -0.05" friction=".5"/>

                    <!-- Define the slide joint in x direction -->
                    <joint name="slide_x" type="slide" axis="1 0 0" frictionloss="0.5" damping="5"/>

                    <!-- Define the hinge joint in y direction with limits -->
                    <joint name="hinge_y" type="hinge" axis="0 1 0" range="-15 15" frictionloss="100" damping="5"/>

                    <!-- Define the touch sensors as box sites on top of the box -->
                    <site name="sensor1" type="box" pos="-0.9 0 -0.01" size="0.1 0.4 0.02" rgba="1 1 1 0.1" />
                    <site name="sensor2" type="box" pos="-0.7 0 -0.01" size="0.1 0.4 0.02" rgba="0.5 0.5 0.5 0.1" />
                    <site name="sensor3" type="box" pos="-0.5 0 -0.01" size="0.1 0.4 0.02" rgba="1 1 1 0.1" />
                    <site name="sensor4" type="box" pos="-0.3 0 -0.01" size="0.1 0.4 0.02" rgba="0.5 0.5 0.5 0.1" />
                    <site name="sensor5" type="box" pos="-0.1 0 -0.01" size="0.1 0.4 0.02" rgba="1 1 1 0.1" />
                    <site name="sensor6" type="box" pos="0.1 0 -0.01" size="0.1 0.4 0.02" rgba="0.5 0.5 0.5 0.1" />
                    <site name="sensor7" type="box" pos="0.3 0 -0.01" size="0.1 0.4 0.02" rgba="1 1 1 0.1" />
                    <site name="sensor8" type="box" pos="0.5 0 -0.01" size="0.1 0.4 0.02" rgba="0.5 0.5 0.5 0.1" />
                    <site name="sensor9" type="box" pos="0.7 0 -0.01" size="0.1 0.4 0.02" rgba="1 1 1 0.1" />
                    <site name="sensor10" type="box" pos="0.9 0 -0.01" size="0.1 0.4 0.02" rgba="0.5 0.5 0.5 0.1" />
                </body>
                
                <!-- Define the object body -->
                <body name="object" pos="{} 0 {}">
                    <geom name="object_geom" type="box" size="{} {} {}" rgba="0.3 0.3 0.8 1" density="{}" friction=".5"/>            
                    <joint name="object_free" type="free" />
                </body>
            </worldbody>


            <actuator>
                <!-- Position control actuators for the slide and hinge joints -->
                <position name="slide_x_actuator" joint="slide_x" kp="{}" kv="{}"/>
                <position name="hinge_y_actuator" joint="hinge_y" kp="1500" kv="250"/>
            </actuator>

            <sensor>
                <!-- Touch sensors connected to the sites -->
                <touch name="touch1" site="sensor1" />
                <touch name="touch2" site="sensor2" />
                <touch name="touch3" site="sensor3" />
                <touch name="touch4" site="sensor4" />
                <touch name="touch5" site="sensor5" />
                <touch name="touch6" site="sensor6" />
                <touch name="touch7" site="sensor7" />
                <touch name="touch8" site="sensor8" />
                <touch name="touch9" site="sensor9" />
                <touch name="touch10" site="sensor10" />
            </sensor>
        </mujoco>
        """.format(x_object_pos, z_object_pos, x_object_size, y_object_size, z_object_size, density, int(kp_slide), int(kv_slide))
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)

        if self.render:
            if self.viewer:
                self.viewer.close()
            self.viewer = viewer.launch_passive(self.model, self.data)
            self.viewer.cam.distance = self.model.stat.extent * 2.0

        # Observation Consists of: Sensor Readings (10), Current Board X Velocity and Acceleration (2), Current Board Angle, Angular Velocity and Angular Acceleration (3) => 15
        touch_sensor_data = [self.data.sensordata[i] for i in range(10)]
        # board velocity, board acceleration, board angle, board angular velocity, board angular acceleration
        board_data = [self.data.qvel[0], self.data.qacc[0], self.data.qpos[1], self.data.qvel[1], self.data.qacc[1]]
        observations = [*touch_sensor_data, *board_data]
        return observations

    def step(self, action: float):
        step_start = time.time()
        # Action Input
        self.data.ctrl[1] = action

        # Event 1
        if 1.0 < self.data.time < 1.01:
            self.data.ctrl[0] = self.event_1_target

        # Event 2
        if 3.0 < self.data.time < 3.01:
            self.data.ctrl[0] = self.event_2_target

        # Simulation Step
        mujoco.mj_step(self.model, self.data)

        # Observation Consists of: Sensor Readings (10), Current Board X Velocity and Acceleration (2), Current Board Angle, Angular Velocity and Angular Acceleration (3) => 15
        touch_sensor_data = [self.data.sensordata[i] for i in range(10)]
        # board velocity, board acceleration, board angle, board angular velocity, board angular acceleration
        board_data = [self.data.qvel[0], self.data.qacc[0], self.data.qpos[1], self.data.qvel[1], self.data.qacc[1]]
        observations = [*touch_sensor_data, *board_data]

        if np.abs(quaternion_to_euler(self.data.xquat[2])[1]) > 30:
            return observations, 0, True, False, ""
        if self.data.time > 5:
            return observations, 0, False, True, ""

        reward = 1 - np.abs(quaternion_to_euler(self.data.xquat[2])[1])/30

        if self.render:
            self.viewer.sync()
            time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

        return observations, reward, False, False, ""  # observation, reward, terminated, truncated, ""

    def close(self):
        if self.viewer:
            self.viewer.close()


if __name__ == '__main__':
    env = MujocoEnvironment(render=True)
    for i in range(10):
        env.reset()
        while True:
            observation, reward, terminated, truncated, info = env.step(0)
            if terminated or truncated:
                break
    env.close()