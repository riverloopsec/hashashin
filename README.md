## Hashashin: A Fuzzy Matching Library for Binary Ninja

*NOTE: This readme may be out of date with regards to changes on the develop branch. The platform currently does not support hashing basic blocks.* 

### Setup
#### Prerequisites
A valid binary ninja license is required to use this library - please ensure Binary Ninja is installed and that the Python
API is available. `setup.py` will do its best to run this install script but may not work in all cases.

To install the Binary Ninja API, run the `install-api.py` script located in the the Binary Ninja installationation
directory:
- MacOS - `/Applications/Binary\ Ninja.app/Contents/Resources/scripts/install-api.py`
- Windows - `C:\Program Files\Vector35\BinaryNinja\scripts\install-api.py`
- Linux - May be in `/opt/binaryninja/scripts/install-api.py`

#### Installation
To install Hashashin as a library, simply run `pip install .` from within the root of this repository.
If you would like to run the provided tests or notebook, run `pip install .[dev]` instead.

### Demo Usage
![](demo.gif)
