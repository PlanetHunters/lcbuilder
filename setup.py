import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
version = "0.25.0"
setuptools.setup(
    name="lcbuilder", # Replace with your own username
    version=version,
    author="M. Dévora-Pajares",
    author_email="mdevorapajares@protonmail.com",
    description="Easy light curve builder from multiple sources",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PlanetHunders/lcbuilder",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ], zip_safe= False,
    python_requires='>=3.13',
    install_requires=['certifi==2025.1.31',
                        'Cython==3.0.6',
                        'everest-pipeline==2.0.12',
                        #'eleanor==2.0.5', included with submodule
                        'pandas==2.2.3',
                        "lightkurve==2.5.0",
                        "matplotlib==3.10.1",
                        "photutils==1.10.0",
                        "pybind11==2.11.1",
                        "requests==2.32.3",
                        "tess-point==0.9.2",
                        "foldedleastsquares==1.1.7",
                        'typing_extensions==4.13.2', #For astropy version
                        'uncertainties==3.2.2',
                        'urllib3==2.4.0',
                        "wotan==1.9",
    ]
)