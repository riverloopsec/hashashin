import setuptools  # type: ignore

installed_api = False
try:
    import binaryninja  # type: ignore
except ImportError as e:
    print("ImportError: {}".format(e))
    print("Attempting to install binja for you...")
    import os

    if os.path.exists(
        "/Applications/Binary Ninja.app/Contents/Resources/scripts/install_api.py"
    ):
        installed_api = True
        print("Installing Binary Ninja API...")
        os.system(
            "python3 /Applications/Binary\ Ninja.app/Contents/Resources/scripts/install_api.py"
        )
    elif os.path.exists("/binaryninja/scripts/install_api.py"):
        installed_api = True
        print("Installing Binary Ninja API...")
        os.system("python3 /binaryninja/scripts/install_api.py")
    else:
        print(
            "Binary Ninja not found. Please install Binary Ninja using install_api.py first."
        )
        import sys

        sys.exit(1)
print(f"Binary Ninja API {'' if installed_api else 'already '}installed.")

setuptools.setup(
    name="hashashin",
    version="0.1.1",
    license="MIT",
    author="Jonathan Prokos",
    author_email="jonathan.prokos@twosixtech.com",
    description="Binary Fingerprint Library",
    packages=["hashashin"],
    package_dir={"hashashin": "hashashin"},
    install_requires=[
        "tqdm",
        "numpy",
        "xxhash",
        "cbor2",
        "SQLAlchemy",
        "python-magic",
        "scikit-learn",
        "GitPython",
        "elasticsearch",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "hashashin = hashashin.cli:cli",
            "flowslicer = flowslicer.flowslicer:Main",
            "populate-db = hashashin.db:populate_db",
            "safedocs-demo = demos.safedocs_05_16_demo:safedocs_demo",
            "hashashin-lib-match = hashashin.db:get_closest_library_version_cli",
        ]
    },
    package_data={
        "hashashin": [
            "net-snmp*.pickle",
            "hashashin.db",
        ]
    },
)

# TODO: This is not outputting during install
if installed_api:
    print(
        "Installed Binary Ninja API. To uninstall run 'python3 /Applications/Binary\ Ninja.app/Contents/Resources/scripts/install_api.py -u'"
    )
