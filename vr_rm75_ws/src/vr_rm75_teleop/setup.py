"""Packaging metadata for the VR RM75 ROS 2 nodes."""

from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'vr_rm75_teleop'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py'),
        ),
        (
            os.path.join('share', package_name, 'docs'),
            glob('docs/*.md'),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='moyu',
    maintainer_email='1956853921@qq.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'meta_quest_bridge = vr_rm75_teleop.meta_quest_bridge:main',
            'quest_dual_ik_fusion = vr_rm75_teleop.quest_dual_ik_fusion:main',
            'rm75_state_node = vr_rm75_teleop.rm75_state_node:main',
            'collision_backend = vr_rm75_teleop.collision_backend_node:main',
        ],
    },
)
