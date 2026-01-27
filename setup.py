"""
Setup script for FiscalMind
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fiscal_mind",
    version="0.1.0",
    author="FiscalMind Team",
    description="面向财务BP的表格分析Agent - A Table Analysis Agent for Financial Business Partners",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Heng-Bian/FiscalMind",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'fiscal-mind=fiscal_mind.main:main',
        ],
    },
)
