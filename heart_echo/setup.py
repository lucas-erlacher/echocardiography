from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

setup(
    name='Heart_Echo',
    version='1.1.4',
    description='Library that facilitates loading heart echo videos',
    url='https://gitlab.inf.ethz.ch/OU-VOGT/heart_echo',
    author='Kieran Chin-Cheong',
    author_email='kieran.chincheong@inf.ethz.ch',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],
    packages=find_packages(include=['heart_echo', 'heart_echo.*']),
    python_requires='>=3.5, <4',
    install_requires=['ffmpeg_python', 'opencv_python', 'tqdm', 'torch', 'matplotlib', 'pandas'],
    package_data={
        'heart_echo': ['Helpers/video_angles_list.csv', 'Helpers/video_angles_unused.csv'],
    }
)