"""Setup."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020, All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 12月 28日 星期一 14:29:37 CST
# ***
# ************************************************************************************/
#

from setuptools import setup

with open("README.md", "r") as file:
    long_description = file.read()

setup(
    name="image_likeness",
    version="1.0.0",
    author="Dell Du",
    author_email="18588220928@163.com",
    description="LPIPS Similarity metric",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/delldu/LPIPS",
    packages=["image_likeness"],
    package_data={"image_likeness": ["models/*.pth", "models/v0.0/*.pth", "models/v0.1/*.pth"]},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "torch >= 1.5.0",
        "torchvision >= 0.6.0",
    ],
)
