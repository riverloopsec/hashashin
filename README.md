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


### CLI Usage

The CLI will only print the hash result to the screen by default, please use `hashashin -h` to see all available options.

#### Hash a single function
```shell
hashashin -b <path_to_binary> -f <function_name || function_address>
```

#### Hash a full binary
```shell
hashashin -b <path_to_binary>
```

### Python Usage

_Note this section is out of date and will be updated soon._

#### Hash a single function  

```python
>>> import hashashin
>>> import binaryninja
>>> bv = binaryninja.open_view('test')
>>> f = bv.get_functions_by_name('main')[0]
>>> hashashin.hash_function(f)
'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
```
#### Hash all functions in a binary
```python
>>> import hashashin
>>> import binaryninja
>>> bv = binaryninja.open_view('test')
>>> hashashin.hash_all(bv)
{
  '86ae180a048a9ed02b0f2413bb0d736fb372233fc93237f73f6fc4f17af696cc': <func: x86_64@0x630>, 
  'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855': <func: x86_64@0x974>, 
  'd3fe3a0e5dfacf36664a3c8478a50fb4f1aee744a9d9d839f65c86c7d91567d4': <func: x86_64@0x666>, 
  ...
}
```

#### Hash basic block (outdated)
```python
>>> import hashashin
>>> import binaryninja
>>> bv = binaryninja.open_view('test')
>>> f = bv.get_functions_by_name('main')[0]
>>> bb = f.basic_blocks[0]
>>> hashashin.hash_basic_block(bb)
'2e196bef7f9beffa99ffbf'
```
