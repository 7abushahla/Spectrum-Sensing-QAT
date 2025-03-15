# Cognitive Radio Spectrum Sensing on the Edge: A Quantization-Aware Deep Learning Approach
_Hamza A. Abushahla, Dara Varam, and Dr. Mohamed I. AlHajri_

This repository contains code and resources for the paper: "[Cognitive Radio Spectrum Sensing on the Edge: A Quantization-Aware Deep Learning Approach](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=4234)"

## Introduction

## Datasets

We evaluate our models using the publicly available SDR and LTE datasets obtained from [here](https://github.com/wineslab/deepsense-spectrum-sensing-datasets).

For SDR, in [bin2hdf5.py](https://github.com/wineslab/deepsense-spectrum-sensing-datasets/blob/main/sdr_wifi_code/bin2hdf5.py), we set `nsamples_per_file = 50000` (to match reported # of occurnces per channel) and `buf = 32` and `128` (control window size, referred to as $N$ within our paper). we keep test_size = `0.1` (to get `90%` training + validation), and `10%` testing. we keep `stride = 12` (overlap between I/Q samples).

For LTE, in [generateLTEDataset.m](https://github.com/wineslab/deepsense-spectrum-sensing-datasets/blob/main/sim_lte_code/generateLTEDataset.m), we set `niq = 32` and `128` (control window size) and varry `snr_db` between `-20db` and `20db`. We keep Cross validation (train: `90%`, test: `10%`) of the generated data. rest of simulation stettings remain as provided by authors. 

## Training models
Models are trained (with modifications) according to the original DeepSense[^1] and ParallelCNN[^2] architectures. To train the standard and QAT versions of the model, navigate to [`/training_scripts`](training_scripts) and look at the different architectures and datasets available. Note that the full model training details (including training parameters, such as `batch_size`, `epochs`, `learning_rate`, etc... can be found in the respective `.py` files corresponding to the configuration. 

For example: 
```css
Spectrum-Sensing/
├──training_scripts/
│   ├──DeepSense/
│   │   ├──LTE
│   │   └──SDR/
│   │       ├──Deepsense128QAT_SDR.py
│   │       └──Deepsense128_SDR.py
```

This structure shows how to access the DeepSense architecture trained on the SDR dataset with a window size of $N=128$. To adjust the window size, keep the dataset and model structure unchanged while modifying the input layer accordingly. Sample scripts are available for $N=128$ (SDR and LTE) and LTE (SNR $=10dB$). For $N=32$ and different SNR values, update the dataset loading to use the corresponding training and testing files.

A sample of trained models in `.tflite` and `.h` formats is available in the [`/trained_models`](trained_models) directory. The folder structure is as follows:


```css
Spectrum-Sensing/
├──trained_models/
│   ├──DeepSense/
│   │   ├──LTE/  
│   │   └──SDR/
│   │       ├──DeepSense_128_normal_SDR_best_overall_model.tflite
│   │       ├──DeepSense_128_normal_SDR_best_overall_model.h
│   │       ├──DeepSense_128_QAT_SDR_best_overall_model_INT8.tflite
│   │       └──DeepSense_128_QAT_SDR_best_overall_model_INT8.h
```

[^1]:https://ieeexplore.ieee.org/document/9488764
[^2]:https://ieeexplore.ieee.org/document/10236565


## Hardware Evaluation
The best-performing models from each configuration, selected based on validation F1-score, were deployed on the Sony Spresense using TensorFlow Lite for Microcontrollers (TFLM).

### Deployment Process  
Deployment on the Sony Spresense involves converting the `.tflite` model into a byte array and integrating it into embedded C code. The steps are outlined in our [Sony Spresense TFLite Guide](https://github.com/7abushahla/Sony-Spresense-TFLite-Guide). Specifically:

1. **Model Conversion**: The trained `.tflite` models were converted into `.h` header files.  
2. **Integration & Flashing**: The models were integrated into the Arduino sketchs (`.ino` files) located in [`inference_scripts`](inference_scripts) and flashed onto the device using the Arduino IDE.  
3. **Inference Testing**: Each script runs the model for **1,000 inferences**, reporting the mean and standard deviation of inference times in milliseconds (ms).

### Data Handling  
Datasets were converted into CSV format and uploaded to an SD card, which was read by the Sony Spresense through the extension board.

### Power Consumption Measurement  
Power consumption was measured using the [Yocto-Amp](https://www.yoctopuce.com/EN/products/usb-electrical-sensors/yocto-amp) current sensor, connected in series with an external 5V source.

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
