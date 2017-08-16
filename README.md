Code from publication @eScience 2017:
Adaptive Lossy Compression of Complex Environmental
Indices using Seasonal Auto Regressive Integrated
Moving Average Models.

# Requirements
This code has been tested on following machine:

```
Python: 3.6.1
OS: Debian 4.11.6-1 (2017-06-19) testing (buster)
CPU: Intel(R) Core(TM) i5-7200U CPU @ 2.50GHz
MEM: 16 GiB 2400MHz DDR4
```

To recreate the software environment you can use the provided
`spec-file.txt` and `requirements.txt` files. Currently only GNU\Linux is supported.

```bash
conda create -n ENVNAME --file spec-file.txt
conda activate ENVNAME
pip install -r requirements.txt
```

> macOS & Windows: Conda does not support cross-plattform export of package names incl. versions. As soon as this feature is added I'll generate the appropiate macOS and Windows environmental files.

# Project structure

```
- run.py
- data
    - arima_uncompressed : Lossily compressed and uncompressed residue data from ARIMA model output (numpy raw)
    - direct_uncompressed : Lossily compressed and uncompressed original data (numpy raw)
    - original : Daily and monthly environmental indices (numpy raw)
    - dm_weather.nc : Daily environmental indices (netcdf)
    - mm_weather.nc : Monthly environmental indices (netcdf)
- model : Parameters of used ARIMA model
- envelope.py
- manualsarima.py
- transport.py
```

# Run

To run the experiment simply execute `python run.py`
