# ros_langgraph

## Install

### Create the workspace

If you don't have a workscape create it with:

```ROS
mkdir -p ~/ros_langgraph/src
cd ~/ros_langgraph/src
```

### Clone the packages
Once in src you have to clone the SinfonIA Framework ROS messages and this package:
```bash
git clone https://github.com/SinfonIAUniandes/perception_msgs.git
git clone https://github.com/SinfonIAUniandes/robot_toolkit_msgs.git
git clone https://github.com/SinfonIAUniandes/navigation_msgs.git
git clone https://github.com/SinfonIAUniandes/manipulation_msgs.git
git clone https://github.com/SinfonIAUniandes/speech_msgs.git
git clone https://github.com/dcuevasa/ros_langgraph.git
```

### Compile workspace

```ROS
cd ~/sinfonia_ws
catkin_make
```

And source:

```ROS
source devel/setup.bash
```

# PATCH FOR ROSPY AND PYTHON3.11

This repo uses python3.11 which is not natively compatible with ROS Noetic, first you must apply this patch to your roslogging.py file in rospy for it to work:

```bash
sudo mv roslogging.py /opt/ros/noetic/lib/python3/dist-packages/rosgraph/roslogging.py
```


