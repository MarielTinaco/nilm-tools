# Sensor Emulator

Emulates sensor via UART. Should work with time-series data. Multi-dimensional data should also work with minor preprocessing at the firmware head. 

## ToDos
- [x] integrate with `cnn_run` or `inference_run`
- [x] validate with other datatypes (floats, uint16, uint8, int8)
- [ ] add statistics to validate model on-device
- [x] arg parser
- [ ] Zephyr version for firmware

## Prereqs
- MaximSDK
- Python 
    - Pyserial `pip install pyserial`

## Run
- To run, flash first the firmware to MAX78000FTHR. (Via vscode, press Ctrl+Shift+B, then flash or Drag and Drop latest hex file in the release directory)
- Restart device
- Run `python sensor_emulator.py --port COM7 --liveplots`

Expected result below:
```
[HOS] Testing comms with device...
[HOS] Connected
[HOS] Sending input row 0
[HOS] Starting inference.
[HOS] Getting predictions.
[DEV] Prediction: 0, 0, 1, 1, 1, 1, 0, 127, 3, 112, 
[HOS] Sending input row 1
[HOS] Starting inference.
[HOS] Getting predictions.
[DEV] Prediction: 0, 0, 1, 1, 1, 1, 0, 127, 5, 112,
```
