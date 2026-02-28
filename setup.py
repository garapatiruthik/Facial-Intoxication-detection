from setuptools import setup, find_packages

setup(
    name="inerb",
    version="1.0.0",
    author="Ruthik garapati",
    author_email="garapatiruthik@gmail.com",
    description="Drunk/Sober detection using facial features and machine learning",
    long_description="INERB - Drunk/Sober Detection System",
    long_description_content_type="text/markdown",
    url="https://github.com/inerb/inerb",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "streamlit",
        "mediapipe",
        "opencv-python",
        "scikit-learn",
        "joblib",
        "numpy",
        "pandas",
        "pyyaml",
        "pytest",
    ],
    entry_points={
        "console_scripts": [
            "inerb-train=inerb.model:main_train",
            "inerb-predict=inerb.model:main_predict",
        ],
    },
)
