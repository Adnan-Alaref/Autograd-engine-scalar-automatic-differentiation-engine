from setuptools import setup, find_packages

setup(
    name="autograd-engine",
    version="1.0.0",
    author="Adnan Alaref",
    author_email="adnanalaref27@example.com",  
    description=(
        "A lightweight, educational, and fully functional automatic differentiation engine "
        "built from scratch in pure Python — inspired by micrograd but extended with "
        "Torch-like features, dynamic learning rate, and graph visualization."
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Adnan-Alaref/Autograd-engine-scalar-automatic-differentiation-engine.git",
    packages=find_packages(),
    install_requires=[
        "graphviz>=0.20",
        "torch>=2.0",  # optional, remove if not used at runtime
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    include_package_data=True,
)
