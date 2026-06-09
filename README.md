# xArm6 for Isaac Sim

The Isaac Sim-ready xArm6 asset is located at:

`assets/xarm6/xarm6.urdf`

## Import

1. Open Isaac Sim.
2. Choose **File > Import**.
3. Select `assets/xarm6/xarm6.urdf`.
4. Set **Robot Type** to `Manipulator`.
5. Set **Base Type** to `Fixed`.
6. Leave **Allow Self-Collision** disabled for the first import.
7. Set the USD output inside `assets/xarm6/usd/`, then import.

The URDF uses relative mesh paths, so keep `xarm6.urdf` and the `meshes`
directory together.

## Source

The model was generated from UFACTORY's official `xarm_ros2`
`xarm_description` package, Humble branch:

https://github.com/xArm-Developer/xarm_ros2

The generated URDF excludes ROS 2 control, Gazebo, and transmission elements
that are not required by Isaac Sim's URDF importer.
