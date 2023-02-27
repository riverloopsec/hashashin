import setuptools  # type: ignore

installed_api = False
try:
    import binaryninja  # type: ignore
except ImportError:
    import os

    if os.path.exists(
        "/Applications/Binary Ninja.app/Contents/Resources/scripts/install_api.py"
    ):
        installed_api = True
        print("Installing Binary Ninja API...")
        os.system(
            "python3 /Applications/Binary\ Ninja.app/Contents/Resources/scripts/install_api.py"
        )
    else:
        print(
            "Binary Ninja not found. Please install Binary Ninja using install_api.py first."
        )
        import sys

        sys.exit(1)

setuptools.setup(
    name="hashashin",
    version="0.1.0",
    license="MIT",
    author="Jonathan Prokos",
    author_email="jonathan.prokos@twosixtech.com",
    description="Binary Fingerprint Library",
    packages=setuptools.find_packages(),
    install_requires=[
        "tqdm",
        "numpy",
        "xxhash",
        "cbor2",
        "SQLAlchemy",
        "python-magic",
        "scikit-learn",
        # "matplotlib",
        # "pandas",
        # "seaborn",
    ],
    # extras_require={
    #     "dev": [
    #     ],
    # },
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "hashashin = hashashin.main:main",
            "flowslicer = flowslicer.flowslicer:Main",
        ]
    },
)

# TODO: This is not outputting during install
if installed_api:
    print(
        "Installed Binary Ninja API. To uninstall run 'python3 /Applications/Binary\ Ninja.app/Contents/Resources/scripts/install_api.py -u'"
    )
