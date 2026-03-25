from pathlib import Path

from setuptools import find_packages, setup


README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")


setup(
    name="rabbit-ai",
    version="0.1.0",
    description="A lightweight retrieval-first local AI assistant.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="PastaHimself",
    python_requires=">=3.11",
    packages=find_packages(include=["rabbit_ai", "rabbit_ai.*"]),
    install_requires=[],
    entry_points={
        "console_scripts": [
            "rabbit-ai=rabbit_ai.cli:run_cli",
        ]
    },
)
