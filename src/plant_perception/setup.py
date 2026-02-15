from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'plant_perception'

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
    description='Plant structure perception pipeline',
    license='MIT',
    entry_points={
        'console_scripts': [
            'perception_node = plant_perception.perception_node:main',
        ],
    },
)
