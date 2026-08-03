from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'simulation'


def model_data_files(model_dir):
    """Recursively collects every file under models/<model_dir>/ (including
    the materials/textures/ subdirectory PBR maps live in) and maps each one
    to its matching destination path under share/simulation/models/<model_dir>/
    -- a plain glob('models/<model_dir>/*') only grabs the top-level model.sdf/
    model.config and silently drops texture files in subdirectories."""
    entries = []
    base = os.path.join('models', model_dir)
    for root, _dirs, files in os.walk(base):
        if not files:
            continue
        dest = os.path.join('share', package_name, root)
        srcs = [os.path.join(root, f) for f in files]
        entries.append((dest, srcs))
    return entries


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')
        ),
        (
            os.path.join('share', package_name, 'worlds'),
            glob('worlds/*')
        ),
        (
            os.path.join('share', package_name, 'config'),
            glob('config/*')
        ),
        *model_data_files('auv'),
        *model_data_files('dam_structure'),
        *model_data_files('seafloor'),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Narayana Panda',
    maintainer_email='narayanapanda10@gmail.com',
    description='Simulation package',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [],
    },
)
