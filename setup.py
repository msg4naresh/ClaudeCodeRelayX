from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="relayx",
    version="0.1.0",
    author="M Naresh",
    author_email="msg4naresh@gmail.com",
    description="A hackable Claude API proxy for monitoring AI agent requests and connecting multiple LLM backends.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/msg4naresh/ClaudeCodeRelayX",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11',
    entry_points={
        'console_scripts': [
            'relayx=relayx.main:main',
        ],
    },
)
