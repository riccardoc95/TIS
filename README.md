[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-38/)
[![Python 3.8.5](https://img.shields.io/badge/python-3.8.5-blue.svg)](https://www.python.org/downloads/release/python-385/)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-39/)
# Topological Image Segmentation
© Copyright by

## Description
TIS is a Python library for the segmentation of astronomical images. It uses persistent homology to find and separate objects.


## License
TIS is free software: you can redistribute it and / or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

## Installation

1) Clone this repository:
```bash
git clone https://github.com/riccardoc95/TIS.git
```
2) Go to this repository:
```bash
cd TIS
```
3) (OPTIONAL) Create new python virtual-env:
```bash
python -m venv venv
```
```bash
source venv/bin/activate
```

4) Install TIS package:
```bash
pip install -e .
```

## How to use
1) (Default options) Run:

```
python -m tis
```
Default paths is `./data/img.fits` for image and `./data/rms.fits` for rms map.
To change inputs and outputs you can use tags:
* `--img`: to change the path of the image
* `--rms`: to change the path of the rms map
* `--output`: to specify the output folder

Example:
```
python -m tis --img /path/of/img.fits --rms /path/of/rms.fits --output /output_folder/
```

2) (Custom options) Run:
```
python -m tis --export_config .
```
to export the configuration file to the current directory. Then, edit the file `tis.conf` and run the script as in point 1 in the same directory as the configuration file.

## Input file format
The script takes only `.fits` files as input.

## Output file format
The segmentation is in `.fits` format while the file containing the information is in `.csv` format.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
