# slip-detection-keras-i3d
This is a Keras implementation of Inflated 3d Inception architecture for bianry classification in fall detection, especially. If you would like to find more details, please refer to: Wu K, He S, Fernie G, Roshan Fekr A. [Deep Neural Network for Slip Detection on Ice Surface](https://www.mdpi.com/1424-8220/20/23/6883#cite). Sensors. 2020; 20(23):6883.

## Dataset
The dataset that is reported in the paper is not publicly released. 

## Environment setup
This code has been tested on Windows 10, Python 3.6, and CUDA 10.0.
- Clone the repository 
```
git clone https://github.com/Kentwhf/fall-detection-keras-i3d.git && cd fall-detection-keras-i3d
export fall-detection-keras-i3d=$PWD
```
- Setup python environment
```
conda create -n fall-detection-keras-i3d python=3.6
source activate fall-detection-keras-i3d
pip install -r requirements.txt
```

## Demo
Change checkpoint parameters for different experiments. 
```
python -m scripts.experiment --cross_validation_type subject-wise
```

## License
All code in this repository are licensed under the MIT license as specified by the LICENSE file.
