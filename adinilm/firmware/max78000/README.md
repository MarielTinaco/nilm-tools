# Sensor Emulator

Emulates sensor via UART. Should work with time-series data. Multi-dimensional data should also work with minor preprocessing at the firmware head. 

## ToDos
- [x] integrate with `cnn_run` or `inference_run`
- [x] validate with other datatypes (floats, uint16, uint8, int8)
- [ ] add statistics to validate model on-device
- [ ] arg parser
- [ ] Zephyr version for firmware

## Prereqs
- MaximSDK
- Python 
    - Pyserial `pip install pyserial`

## Run
- To run, flash first the firmware to MAX78000FTHR. (Via vscode, press Ctrl+Shift+B, then flash or Drag and Drop latest hex file in the release directory)
- Restart device
- Run `python sensor_emulator.py`

Expected result below:
```
[HOS] Testing comms with device...
[DEV] ACK'ed
[HOS] Sending input window 1
[DEV] ACK'ed
[DEV] Prediction: TEST_PRED. Last values are 5 28 3 3 3
[HOS] Sending input window 2
[DEV] ACK'ed
[DEV] Prediction: TEST_PRED. Last values are 1 7 0 2 3
```