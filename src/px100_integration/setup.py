from setuptools import find_packages, setup

package_name = 'px100_integration'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='grail',
    maintainer_email='grail@todo.todo',
    description='PincherX-100 reachability checker (dry-run only)',
    license='MIT',
    entry_points={
        'console_scripts': [
            'reachability_checker = px100_integration.reachability_checker:main',
        ],
    },
)
