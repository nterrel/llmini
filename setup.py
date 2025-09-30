import os
from setuptools import setup, find_packages

# Run the setup script for external dependencies
if os.path.exists("scripts/setup_external.py"):
    exec(open("scripts/setup_external.py").read())

setup(
    name="llmini",
    version="1.0.0",
    author="Nick Terrel",
    description="A Tiny LLM Implementation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/nterrel/llmini",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "": ["external/*"],  # Include all files in the external directory
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "numpy",
        "tqdm",
    ],
)
