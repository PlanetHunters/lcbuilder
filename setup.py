import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
version = "0.23.2"
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
    python_requires='>=3.10',
    install_requires=['numpy==1.23.5',
                        'astropy==5.3.4',
                      'certifi==2025.1.31',
                        'Cython==3.0.6',
                        'everest-pipeline==2.0.12',
                        #'eleanor==2.0.5', included with submodule
                        'pandas==1.5.3',
                        "lightkurve==2.4.2",
                        "matplotlib==3.8.2",
                        "photutils==1.10.0",
                        "pybind11==2.11.1",
                        "requests==2.32.3",
                        "scipy==1.11.4",
                        "statsmodels==0.13.5",
                        "tess-point==0.6.1",
                        "foldedleastsquares==1.1.6",
                        'typing_extensions==4.7.1', #For astropy version
                        'uncertainties==3.2.2',
                        'urllib3==2.4.0',
                        "wotan==1.9",
    ]
)