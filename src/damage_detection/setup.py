from setuptools import find_packages, setup
from glob import glob

package_name = 'damage_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Narayana Panda',
    maintainer_email='narayanapanda10@gmail.com',
    description='Live per-agent AI crack detection (offline training tools + ROS2 inference node)',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'infer_node = damage_detection.infer_node:main',
        ],
    },
)
