from setuptools import setup, find_packages

setup(
    name="cast",
    version="0.0.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "cast": ["*.so", "*.pyd", "*.dylib", "*.dll", "*.pyi"],
    },
    zip_safe=False,
)