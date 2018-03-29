# Shift Operation CUDA Implementation

created by Peter Jin

Tradeoffs and further analysis can be found in the paper. If you find this work useful for your research, please consider citing:

    @inproceedings{shift,
        Author = {Bichen Wu and Alvin Wan and Xiangyu Yue and Peter Jin and Sicheng Zhao and Noah Golmant and Amir Gholaminejad and Joseph Gonzalez and Kurt Keutzer},
        Title = {Shift: A Zero FLOP, Zero Parameter Alternative to Spatial Convolutions},
        Journal = {arXiv:1711.08141},
        Year = {2017}
    }

Uses of Shift:
* [ShiftResNet](http://github.com/alvinwan/shiftresnet-cifar) for CIFAR-10, CIFAR-100 classification

# Installation

> If you have included this `shift` repository as a submodule in a separate repository, feel free to skip down to step 5.

1. If you have not already, setup a virtual environment with Python3, and activate it.

```
virtualenv shift --python=python3
source shift/bin/activate
```

Your prompt should now be prefaced with `(shift)`, as in

```
(shift) [user@server:~]$ 
```

2. Install `pytorch` and `torchvision`. Access [pytorch.org](http://pytorch.org), scroll down to the "Getting Started" section, and select the appropriate OS, package manager, Python, and CUDA build. For example, selecting Linux, pip, Python3.5, and CUDA 8 gives the following, as of the time of this writing

```
pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl 
pip3 install torchvision
```

3. Clone this repository.

```
git clone git@github.com:peterhj/shiftnet_cuda_v2.git
```

4. `cd` into the root of this repository.

```
cd shiftnet_cuda_v2
```

5. Install the Python requirements for this package.

```
pip3 install -r requirements.txt
```

6. Compile the Shift Layer implementation in C.

```
make
```

> **Getting `invalid_device_function`?** Update the architecture code in [`models/shiftnet_cuda_v2/Makefile`](https://github.com/alvinwan/shiftresnet-cifar/blob/master/models/shiftnet_cuda_v2/Makefile#L4), currently configured for a Titan X. e.g., A Tesla K80 is `sm-30`.

Your custom CUDA layer is now installed.

# Test

To check that the build completed successfully, run the test script

```
python test_shiftnet.py
```

After ~3s, the script should output a number of different tensors, where the last tensor has non-zero values only in the first column.

```
Columns 13 to 17
   89     0     0     0     0
  107     0     0     0     0
  125     0     0     0     0
  143     0     0     0     0
  161     0     0     0     0
  179     0     0     0     0
  197     0     0     0     0
  215     0     0     0     0
  233     0     0     0     0
  251     0     0     0     0
  269     0     0     0     0
  287     0     0     0     0
  305     0     0     0     0
  323     0     0     0     0
    0     0     0     0     0
    0     0     0     0     0
    0     0     0     0     0
    0     0     0     0     0
[torch.FloatTensor of size 18x18]
```
