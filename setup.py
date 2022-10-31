import setuptools
installed_api = False
try:
    import binaryninja
except ImportError:
    import os
    if os.path.exists("/Applications/Binary Ninja.app/Contents/Resources/scripts/install_api.py"):
        installed_api = True
        print("Installing Binary Ninja API...")
        os.system("python3 /Applications/Binary\ Ninja.app/Contents/Resources/scripts/install_api.py")
    else:
        print("Binary Ninja not found. Please install Binary Ninja using install_api.py first.")
        import sys; sys.exit(1)

setuptools.setup(
    name="hashashin",
    version="0.0.1",
    author="Jonathan Prokos",
    author_email="jonathan.prokos@twosixtech.com",
    description="Binary Fingerprint Library",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "jupyter",
        "seaborn",
        "matplotlib",
        "tqdm",
        "python-magic",
    ],
    python_requires=">=3.6",
)

# TODO: This is not outputting during install
if installed_api:
    print("Installed Binary Ninja API. To uninstall run 'python3 /Applications/Binary\ Ninja.app/Contents/Resources/scripts/install_api.py -u'")
