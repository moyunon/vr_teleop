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
        ],
    },
)
