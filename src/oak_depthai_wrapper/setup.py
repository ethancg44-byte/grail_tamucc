from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'oak_depthai_wrapper'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='grail',
    maintainer_email='grail@todo.todo',
    description='ROS 2 wrapper for Luxonis OAK-D Pro with DepthAI on-device inference',
    license='MIT',
    entry_points={
        'console_scripts': [
            'oak_camera_node = oak_depthai_wrapper.oak_camera_node:main',
        ],
    },
)
