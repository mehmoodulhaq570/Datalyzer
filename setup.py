from setuptools import setup, find_packages

setup(
    name="smart-datalyzer",
    version="0.1.0",
    description="Data analysis and reporting toolkit",
        author="Mehmood Ul Haq",
        license="MIT",
    author_email="mehmoodulhaq1040@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pandas",
        "numpy",
        "scipy",
        "statsmodels",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "rich"
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "datalyzer = datalyzer.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    url="https://github.com/mehmoodulhaq570/smart-datalyzer",
    project_urls={
        "Bug Tracker": "https://github.com/mehmoodulhaq570/smart-datalyzer/issues",
    },
)
