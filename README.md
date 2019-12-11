## Hashashin: A Fuzzy Matching Tool for Binary Ninja

This tool detects similar functions between binaries, and ports annotations (currently only in the form of tags)
 for matching basic blocks.

### Setup

#### Installation in Docker

Pre-requisites:
- Download a Linux version of [Binary Ninja](https://binary.ninja/) (commercial or headless) `BinaryNinja.zip` from Vector35 and place at `third-party/binaryninja/BinaryNinja.zip`
- Place your license file at `third-party/binaryninja/license.dat`

Then, build the provided Docker containers as described here:
```bash
docker build -t safedocs/base/static-light -f Dockerfile.base .
docker build -t safedocs/hashashin:local -f Dockerfile.dev .
```

Due to how the Developer docker is setup below, your code changes will sync into it and
 you do not need to rebuild unless dependencies change.
Launch and enter the container:
```bash
docker run -it --rm -v"$(pwd):/processor" --name dev-hashashin safedocs/hashashin:local
```

#### Installation Locally

The Docker method is supported to enable more consistent development envrionments.
However, you should be able to run locally as long as the dependencies are installed:
- BinaryNinja headless
- Python numpy


### Usage

### Generate Signatures

Provide a Binary Ninja database with tags applied to it which you want to identify.

```bash
./src/generate_signatures.py <input_bndb> <signature_file>
```

For example, `./src/generate_signatures.py tests/test1_annotated.bndb test1_annotated.sig`.

> NOTE: On sizeable files with many functions, this process will take time.

### Apply Signatures

```bash
./src/apply_signatures.py <input_binary> <signature_file>
```

This will output to a file at the same path as the `input_binary` with `.bndb` appended,
 which is the annotated Binary Ninja database as a result of applying signatures.
