# Cognitive Radio Spectrum Sensing on the Edge: A Quantization-Aware Deep Learning Approach
_Hamza A. Abushahla, Dara Varam, and Dr. Mohamed I. AlHajri_

This repository contains code and resources for the paper: "[Cognitive Radio Spectrum Sensing on the Edge: A Quantization-Aware Deep Learning Approach](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=4234)"

## Introduction

We evaluate our models using the publicly available SDR and LTE datasets obtained from: https://github.com/wineslab/deepsense-spectrum-sensing-datasets

For SDR, in "[bin2hdf5.py]([https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=4234](https://github.com/wineslab/deepsense-spectrum-sensing-datasets/blob/main/sdr_wifi_code/bin2hdf5.py)", we set `nsamples_per_file = 50000` (to match reported # of occurnces per channel) and `buf = 32` and `128` (control window size, referred to as $N$ within our paper). we keep test_size = `0.1` (to get `90%` training + validation), and `10%` testing. we keep `stride = 12` (overlap between I/Q samples).

For LTE, in "[generateLTEDataset.m]([[https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=4234](https://github.com/wineslab/deepsense-spectrum-sensing-datasets/blob/main/sdr_wifi_code/bin2hdf5.py](https://github.com/wineslab/deepsense-spectrum-sensing-datasets/blob/main/sim_lte_code/generateLTEDataset.m))", we set `niq = 32` and `128` (control window size) and varry `snr_db` between `-20db` and `20db`. we keep Cross validation (train: `90%`, test: `10%`) of the generated data. rest of simulation stettings remain as provided by authors. 

## Training models
Models are trained (with modifications) according to the original DeepSense[^1] and ParallelCNN[^2] architectures. 

[^1]:https://ieeexplore.ieee.org/document/9488764
[^2]:https://ieeexplore.ieee.org/document/10236565


## Hardware Evaluation
The best-performing models from each configuration, selected based on validation F1-score, were deployed on the Sony Spresense using TensorFlow Lite Micro (TFLM). 

Deployment on Sony Spresense involves converting the `.tflite` model to a byte array and integrating it into embedded C code using TensorFlow Lite for Microcontrollers (TFLM). Detailed steps are provided in our [Sony Spresense TFLite Guide](https://github.com/7abushahla/Sony-Spresense-TFLite-Guide).

Details on how deployment was done can be found here: https://github.com/7abushahla/Sony-Spresense-TFLite-Guide. Trained models in .tflite format were converted to .h header files and then using the (embedded C code is an Arduino sketch (`.ino` file)) in [`inference_scripts`](inference_scripts) the models are flashed to the devices memory (Flash the compiled code onto the Sony Spresense device's memory using the Arduino IDE) and 1000 inferences are carried to get the average inference time (Each script runs the model for **1,000 inferences** and reports the mean and standard deviation of the inference times in **milliseconds (ms)**.). Datasets were converted into csv format and uploaded on an SD card, which is read by the Sony through the extenstention board. 

Power consumption measurements are performed using the **[Yocto-Amp](https://www.yoctopuce.com/EN/products/usb-electrical-sensors/yocto-amp)** current sensor, connected in series with an external 5V source.

## Citation & Reaching out
If you use our work for your own research, please cite us with the below: 

```
@Article{abushahla2025cognitive,
AUTHOR = {Abushahla, Hamza A. and Varam, Dara and AlHajri, Mohamed I.},
TITLE = {Cognitive Radio Spectrum Sensing on the Edge: A Quantization-Aware Deep Learning Approach},
JOURNAL = { },
VOLUME = {},
YEAR = {},
NUMBER = {},
ARTICLE-NUMBER = {},
URL = {},
ISSN = {},
ABSTRACT = {},
DOI = {}
}
```

You can also reach out through email to: 
- Hamza Abushahla - b00090279@alumni.aus.edu
- Dara Varam - b00081313@alumni.aus.edu
- Dr. Mohamed AlHajri - mialhajri@aus.edu
