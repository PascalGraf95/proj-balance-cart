import mujoco
import mediapy as media
from mujoco import viewer
import time
import matplotlib.pyplot as plt
import numpy as np

# region - Model Definition -
xml = """
<mujoco model="balance_cart">
    <compiler angle="degree" />
    <option timestep=".001"/>
    
    <extension>
        <plugin plugin="mujoco.elasticity.solid"/>
    </extension>

    <worldbody>
        <!-- Define the ground plane
        <geom name="ground" type="plane" pos="0 0 0" size="10 10 0.1" rgba="0.9 0.9 0.9 1"/>
         -->

        <!-- Define the box body -->
        <body name="box" pos="0 0 0.05">
            <!-- Define the box geometry -->
            <geom name="box_geom" type="box" size="1 0.5 0.01" rgba="0.8 0.3 0.3 1" friction=".5"/>
            
            <!-- Define the slide joint in x direction -->
            <joint name="slide_x" type="slide" axis="1 0 0" frictionloss="0.5" damping="5"/>
            
            <!-- Define the hinge joint in y direction with limits -->
            <joint name="hinge_y" type="hinge" axis="0 1 0" range="-15 15" frictionloss="0.5" damping="5"/>
            
            <!-- Define the touch sensors as box sites on top of the box -->
            <site name="sensor1" type="box" pos="-0.8 0 0.0" size="0.1 0.4 0.01" rgba="1 1 1 0.1" />
            <site name="sensor2" type="box" pos="-0.6 0 0.0" size="0.1 0.4 0.01" rgba="0.5 0.5 0.5 0.1" />
            <site name="sensor3" type="box" pos="-0.4 0 0.0" size="0.1 0.4 0.01" rgba="1 1 1 0.1" />
            <site name="sensor4" type="box" pos="-0.2 0 0.0" size="0.1 0.4 0.01" rgba="0.5 0.5 0.5 0.1" />
            <site name="sensor5" type="box" pos="0 0 0.0" size="0.1 0.4 0.01" rgba="1 1 1 0.1" />
            <site name="sensor6" type="box" pos="0.2 0 0.0" size="0.1 0.4 0.01" rgba="0.5 0.5 0.5 0.1" />
            <site name="sensor7" type="box" pos="0.4 0 0.0" size="0.1 0.4 0.01" rgba="1 1 1 0.1" />
            <site name="sensor8" type="box" pos="0.6 0 0.0" size="0.1 0.4 0.01" rgba="0.5 0.5 0.5 0.1" />
            <site name="sensor9" type="box" pos="0.8 0 0.0" size="0.1 0.4 0.01" rgba="1 1 1 0.1" />
        </body>
        <!-- Define the object body -->
        <body name="object" pos="-0.3 0 0.65">
            <geom name="object_geom" type="box" size="0.3 0.15 0.6" rgba="0.3 0.3 0.8 1" mass="1" friction=".5"/>
            
            <joint name="object_free" type="free" />
        </body>
        <!--
        <flexcomp type="grid" count="3 3 3" spacing=".2 .2 .2" pos="0 0 .5" radius=".0" rgba="0 1 0 .5" name="object2" dim="3" mass="1">
            <edge damping="1"/>
            <plugin plugin="mujoco.elasticity.solid">
                <config key="poisson" value="0"/>
                <config key="young" value="2e4"/>
            </plugin>
        </flexcomp>
        -->
    </worldbody>

    <actuator>
        <!-- Position control actuators for the slide and hinge joints -->
        <position name="slide_x_actuator" joint="slide_x" kp="300" kv="100"/>
        <position name="hinge_y_actuator" joint="hinge_y" kp="1000" kv="200"/>
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
    </sensor>
</mujoco>
"""
# endregion

# region - Initialization -
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
try:
    model.joint()
except KeyError as e:
    print(e)
# random initial rotational velocity:
mujoco.mj_resetData(model, data)
# data.qvel[3:6] = 5 * np.random.randn(3)
# endregion


touch_sensor_readings = [[] for i in range(9)]
relative_object_position = []
cart_position = []
cart_angle = []
time_steps = []
object_id = model.geom("object_geom").id
cart_id = model.geom("box_geom").id

with (viewer.launch_passive(model, data) as viewer):
    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()

    # Setup camera
    # viewer.cam.type = 1
    # viewer.cam.trackbodyid = model.geom("red_box").id
    """
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

    # tweak scales of contact visualization elements
    model.vis.scale.contactwidth = 0.1
    model.vis.scale.contactheight = 0.03
    model.vis.scale.forcewidth = 0.05
    model.vis.map.force = 0.3
    """

    while viewer.is_running() and data.time < 10:
        step_start = time.time()

        # Break when the object falls from the cart
        if data.geom_xpos[object_id][2] < -2:
            break
        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        mujoco.mj_step(model, data)

        time_steps.append(data.time)
        for i in range(9):
            touch_sensor_readings[i].append(data.sensordata[i])
        relative_object_position.append(data.geom_xpos[object_id][0] - data.geom_xpos[cart_id][0])
        cart_position.append(data.geom_xpos[cart_id][0])
        cart_angle.append(data.qpos[1])

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        # print(data.time)
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

    # Create a figure and subplots
    fig1, axis = plt.subplots(9, 1, figsize=(15, 15), sharex=True)

    # Plot actual data
    for i, a in zip(range(9), axis):
        a.plot(time_steps, touch_sensor_readings[i], label="Sensor: {:02d}".format(i))
        a.set_title(label="Touch Sensor: {:02d}".format(i))
        a.set_ylabel("Sensor Readings")
        a.grid(True)

    # Add a title that spans all subplots
    fig1.suptitle('Touch Sensor Readings - Balance Cart', fontsize=16, fontweight='bold')

    fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15), sharex=True)
    ax1.plot(time_steps, relative_object_position, label="Relative Object Position")
    ax2.plot(time_steps, cart_position, label="Cart Position")
    ax3.plot(time_steps, np.rad2deg(cart_angle), label="Cart Angle")

    # Set the title of the plot
    ax1.set_title("Relative Object Position")
    ax2.set_title("Cart Position")
    ax3.set_title("Cart Angle")

    # Set the x- and y-axis labels
    ax3.set_xlabel("Time in s")

    # Add a title that spans all subplots
    fig2.suptitle('Positional and Joint Data - Balance Cart', fontsize=16, fontweight='bold')

    # Enable the grid
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)

    # Show the plot
    plt.show()