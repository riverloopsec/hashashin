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

### `net-snmp` Matching
Once installed, you can run `hashashin-lib-match net-snmp <path_to_binary>` to estimate its closest stored `net-snmp` binary. If you do not have the appropriate pickle files it will generate them from the `.db` file assuming you have all 576 binaries in the database.
```
❯ hashashin-lib-match net-snmp hashashin/binary_data/pilot/downloads/7a738681-aec5-4ff0-9755-9e8d7bafadce/7da5e3d9-3aaf-4580-ab1a-1eb9ac2f57dc/sbin/snmpd/sbin/snmpd                                             ─╯
INFO:hashashin.db:Found 1 files matching net-snmp filename
INFO:hashashin.db:Loaded net-snmp signature matrix from /Users/jonathan.prokos/Documents/Projects/hashashin/hashashin/net-snmp_triage.pickle
WARNING:hashashin.classes:Wasting space here, can shorten array by 256 bytes by using uint32
INFO:hashashin.db:Closest net-snmp signature to hashashin/binary_data/pilot/downloads/7a738681-aec5-4ff0-9755-9e8d7bafadce/7da5e3d9-3aaf-4580-ab1a-1eb9ac2f57dc/sbin/snmpd/sbin/snmpd is 0.2. Continuing to robust matching.
INFO:hashashin.db:Loaded net-snmp signatures from /Users/jonathan.prokos/Documents/Projects/hashashin/hashashin/net-snmp_signatures.pickle
INFO:hashashin.db:Closest net-snmp function features is 0.3231333729039647 to v5.1.4.
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:03<00:00,  3.91s/it]
INFO:hashashin.db:hashashin/binary_data/pilot/downloads/7a738681-aec5-4ff0-9755-9e8d7bafadce/7da5e3d9-3aaf-4580-ab1a-1eb9ac2f57dc/sbin/snmpd/sbin/snmpd: 5.1.4
5.1.4
```

### Docker
Using the provided [Dockerfile](./Dockerfile) you can build an image for net-snmp matching with 

```docker build -t gcr.io/pilot-security-eng/research/hashashin -f Dockerfile .```

You must have a valid license `pilot_2023_headless_license.dat` in the TLD before building and it is recommended to generate `net-snmp_{signatures,triage}.pickle` before building as well. 

### Demo Usage
![](demo.gif)

### libcurl similarity matrix
![](libcurl_similarity_matrix.png)

### net-snmp similarity matrix
![](net-snmp-full-matrix.png)
