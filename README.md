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

# Clone the graph-autoencoder repo and check out OND/Tinker port branch.
cd graph-autoencoder
git checkout ond-tinker-port


# Install dependencies.
pip install opencv-python pandas ubelt

# Install dgl, being sure the CUDA version matches the version of CUDA
# on the system (this example assumes CUDA 10.0).
pip install dgl-cu100

pip install -e .
```

In
[dummy_interface.py](./sailon/dummy_interface.py), modify the `"red_light"`
metadata value in `DummyInterface.get_test_metadata` and modify the dataset
file list in `DummyInterface.dataset_request` to point to the desired test
video files. Modify the [GAE sample config](./configs/gae_nd_rd_config.yaml)
to point to the correct detector backbone weights, graph weights, and EVM
weights. Then, from the repo root directory, run the following:

```
# Enter the sailon-protocols directory and run the test config.
tinker -c configs/gae_nd_rd_config.yaml ond.py
```

**Sample output:**

```
tinker -c configs/gae_nd_rd_config.yaml  ond.py
Using backend: pytorch
Beginning round 0
/home/najam/.virtualenv/tinker-engine-dKqDR-tU/lib/python3.7/site-packages/dgl/base.py:45: DGLWarning: Recommend creating graphs by `dgl.graph(data)` instead of `dgl.DGLGraph(data)`.
  return warnings.warn(message, category=category, stacklevel=1)
session_id: 9dd59867-b2be-4dde-9e8e-89c51f2317e3 test_id: OND.sample.1 Round id: 0: Finished feature extraction 0/3
session_id: 9dd59867-b2be-4dde-9e8e-89c51f2317e3 test_id: OND.sample.1 Round id: 0: Finished feature extraction 1/3
session_id: 9dd59867-b2be-4dde-9e8e-89c51f2317e3 test_id: OND.sample.1 Round id: 0: Finished feature extraction 2/3
session_id: 9dd59867-b2be-4dde-9e8e-89c51f2317e3 test_id: OND.sample.1 Round id: 0: Starting to detect change in world
Detected red light image
session_id: 9dd59867-b2be-4dde-9e8e-89c51f2317e3 test_id: OND.sample.1 Round id: 0: Starting novelty classification
session_id: 9dd59867-b2be-4dde-9e8e-89c51f2317e3 test_id: OND.sample.1 Round id: 0: Starting to classify samples
Softmax scores: tensor([51,  0,  0])
EVM scores: tensor([45, 12, 51])
session_id: 9dd59867-b2be-4dde-9e8e-89c51f2317e3 test_id: OND.sample.1 Round id: 0: Finished classifying samples
Beginning round 1
session_id: 9dd59867-b2be-4dde-9e8e-89c51f2317e3 test_id: OND.sample.1: Starting to characterize samples
session_id: 9dd59867-b2be-4dde-9e8e-89c51f2317e3 test_id: OND.sample.1: self.Novel_dict is empty.
session_id: 9dd59867-b2be-4dde-9e8e-89c51f2317e3 test_id: OND.sample.1: Finished characterizing samples
```
