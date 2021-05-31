# SAIL-ON protocols

## Requirements
* pipenv

## Installation

```
# Install tinker engine.
git clone git@github.com:tinker-engine/tinker-engine.git
cd tinker-engine
pipenv install -d

# Activate the pipenv shell.
pipenv shell

# Install sailon-protocols "sailon" package (for testing).
cd ..
git clone git@github.com:tinker-engine/sailon-protocols.git
cd sailon-protocols
pip install -e .
```

## Graph auto-encoder

Activate the Tinker engine `pipenv` environment from above, then:

```
# Clone and install the evm_based_novelty_detector repo.
cd evm_based_novelty_detector
pip install -e .
```

```
# Clone the graph-autoencoder repo and check out OND/Tinker port branch.
cd graph-autoencoder
git checkout ond-tinker-port


# Install dependencies.
pip install opencv-python pandas ubelt

# Install dgl, being sure the CUDA version matches the version of CUDA
# on the system (this example assumes CUDA 10.0).
pip install dgl-cu100

pip install -e .

# Enter the sailon-protocols directory and run the test config.
cd ../sailon-protocols
tinker -c configs/gae_nd_rd_config.yaml ond.py
```
