# Wavenet pytorch
Easy to read wavenet pytorch implementation, focusing on high gain amps.

Heavily inspired by https://github.com/sdatkinson/neural-amp-modeler

## About
Simple implementation of wavenet, and a standard convnet for audio amp capture.
This project focuses on high gain amps.

The training data consists of
* Input: DI guitar
    * Low threshold noise gate used to remove noise
* Output: Full amp rig
    * Full rig consists of gate, compressor, drive, amp, cab and EQ
    * Made using the AxeFX 2

## Audio examples
Input
* [Validation audio](audio/validation_DI.wav)

Output
* [Wavenet output](audio/model_output/wavenet.wav)
* [Convnet output](audio/model_output/convnet.wav)

## Test
```
python test.py
```
## Train
```
python train.py
```
![Training waveform](images/training_waveform.png)
_Wavenet result from 100 epochs of training_

