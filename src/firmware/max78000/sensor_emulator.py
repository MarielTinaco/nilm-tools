import datetime
import time
import serial
import csv
import numpy as np
import matplotlib.pyplot as plt

# TODO 
# 1. Add streaming mode.
# 2. Fix plot. Speed up pipeline.

# Packet format is <Command 2B> <Len 2B> <Data Len-B> <Delimeter 2B>
# AA01 - Init
# AA02 - Send packet
# DD01 - Delimeter

class SensorEmulator():
    def __init__(self,
                 serial_port,
                 serial_baudrate,
                 aggregated_power_csv,
                 buffered = True,
                 streaming_mode = True):
        self.serial = serial.Serial(
            port=serial_port,
            baudrate=serial_baudrate,
        )

        self.row_idx = 0
        self.col_idx = 0
        self.num_rows = 0
        self.buffered = buffered
        self.raw_data = []
        self.data_bytes = []
        self.read_data = bytes()
        if self.buffered:
            self.states = np.zeros((5, 1000))
            self.rms = np.zeros((5, 1000))

            self.fig, self.axes = plt.subplots(nrows=5, ncols=2)

            plt.sca(self.axes[0,0])
            self.axes[0,0].set_ylabel("State")
            self.axes[0,0].set_xlabel("Time")
            self.axes[0,0].set_title('Appliance 0') 

            plt.sca(self.axes[1,0])
            self.axes[1,0].set_ylabel("State")
            self.axes[1,0].set_xlabel("Time")
            self.axes[1,0].set_title('Appliance 1') 

            plt.sca(self.axes[2,0])
            self.axes[2,0].set_ylabel("State")
            self.axes[2,0].set_xlabel("Time")
            self.axes[2,0].set_title('Appliance 2') 

            plt.sca(self.axes[3,0])
            self.axes[3,0].set_ylabel("State")
            self.axes[3,0].set_xlabel("Time")
            self.axes[3,0].set_title('Appliance 3') 

            plt.sca(self.axes[4,0])
            self.axes[4,0].set_ylabel("State")
            self.axes[4,0].set_xlabel("Time")
            self.axes[4,0].set_title('Appliance 4') 

            plt.sca(self.axes[0,1])
            self.axes[0,1].set_ylabel("RMS")
            self.axes[0,1].set_xlabel("Time")
            self.axes[0,1].set_title('Appliance 0') 

            plt.sca(self.axes[1,1])
            self.axes[1,1].set_ylabel("RMS")
            self.axes[1,1].set_xlabel("Time")
            self.axes[1,1].set_title('Appliance 1') 

            plt.sca(self.axes[2,1])
            self.axes[2,1].set_ylabel("RMS")
            self.axes[2,1].set_xlabel("Time")
            self.axes[2,1].set_title('Appliance 2') 

            plt.sca(self.axes[3,1])
            self.axes[3,1].set_ylabel("RMS")
            self.axes[3,1].set_xlabel("Time")
            self.axes[3,1].set_title('Appliance 3') 

            plt.sca(self.axes[4,1])
            self.axes[4,1].set_ylabel("RMS")
            self.axes[4,1].set_xlabel("Time")
            self.axes[4,1].set_title('Appliance 4') 

        with open(aggregated_power_csv, 'r') as f:
            csv_reader = csv.reader(f)

            for i in csv_reader:
                self.raw_data.append(list(i))
                self.data_bytes.append("".join(str(j) + "," for j in i))
                self.num_rows += 1
    
    def build_packet(self, packet_id = None):
        if packet_id == 'init':
            packet = bytes()
            packet += (bytes.fromhex("AA01"))
            packet += (bytes.fromhex("{:04x}".format(0)))
            packet += (bytes.fromhex("DD01"))
        elif packet_id == 'infer':
            packet = bytes()
            packet += (bytes.fromhex("AA03"))
            packet += (bytes.fromhex("{:04x}".format(0)))
            packet += (bytes.fromhex("DD01"))
        elif packet_id == 'get_output':
            packet = bytes()
            packet += (bytes.fromhex("AA04"))
            packet += (bytes.fromhex("{:04x}".format(0)))
            packet += (bytes.fromhex("DD01"))
        else: 
            packet = bytes()
            packet += (bytes.fromhex("AA02"))
            packet += (bytes.fromhex("{:04x}".format(len(self.data_bytes[self.row_idx]))))
            packet += (self.data_bytes[self.row_idx].encode())
            packet += (bytes.fromhex("DD01"))

        return packet
    
    def send(self, packet_id = None, text = None):
        if text is not None:
            return self.serial.write(text.encode())
        if self.row_idx < self.num_rows:
            return self.serial.write(self.build_packet(packet_id=packet_id))
    
    def recv(self, num_bytes = 1):
        self.read_data = self.serial.read(num_bytes)
        return self.read_data

    def wait(self, expected_bytes, num_bytes = 4, timeout = 10, serial_rx_timeout = 1):
        assert timeout > serial_rx_timeout
        assert timeout % serial_rx_timeout == 0

        self.serial.timeout = serial_rx_timeout
        expected_bytes = [i.encode() for i in expected_bytes]

        start = datetime.datetime.now()
        elapsed = datetime.datetime.now() - start

        while (elapsed < datetime.timedelta(seconds=timeout)):
            read_data = self.serial.read(num_bytes)
            if (read_data in expected_bytes):
                self.read_data = read_data
                return 0
            else:
                elapsed = datetime.datetime.now() - start

        return -2
    
    def update_and_show(self, output_tensor: list):
        states = np.array([output_tensor[:5]])
        rms = np.array([output_tensor[5:]])

        self.states = np.concatenate((self.states, states.T), axis = 1)
        self.rms = np.concatenate((self.rms, rms.T), axis = 1)
        self.states = np.delete(self.states, 0, 1)
        self.rms = np.delete(self.rms, 0, 1)

        tvec = np.linspace(0, 999, 1000)

        self.axes[0,0].plot(tvec, self.states[0])
        self.axes[1,0].plot(tvec, self.states[1])
        self.axes[2,0].plot(tvec, self.states[2])
        self.axes[3,0].plot(tvec, self.states[3])
        self.axes[4,0].plot(tvec, self.states[4])
        self.axes[0,1].plot(tvec, self.rms[0])
        self.axes[1,1].plot(tvec, self.rms[1])
        self.axes[2,1].plot(tvec, self.rms[2])
        self.axes[3,1].plot(tvec, self.rms[3])
        self.axes[4,1].plot(tvec, self.rms[4])

        plt.pause(0.1)
    
    def log(self, tag, data):
        print("[{0}]".format(tag), end=' ')
        print(data)
    
    def emulate(self, debug_acks=False):
        ret = 0
        count = 1

        self.log("HOS","Testing comms with device...")
        self.send(packet_id='init')
        ret = self.wait(["ACK", "NAK"], num_bytes=3)
        if ret:
            self.log("HOS", "Timed out")
            return ret
        
        if debug_acks:
            self.log("DEV","{0}'ed".format(self.read_data.decode()))
        else:
            self.log("HOS","Connected")

        while (self.row_idx < self.num_rows):
            self.log("HOS", "Sending input data {0}".format(count))
            time.sleep(1)
            self.send()
            time.sleep(1)
            ret = self.wait(["ACK", "NAK"], num_bytes=3)
            if ret:
                self.log("HOS", "Timed out")
                return ret
            if debug_acks:
                self.log("DEV","{0}'ed".format(self.read_data.decode()))

            time.sleep(1)
            self.log("HOS", "Starting inference.")
            time.sleep(1)
            self.send('infer')
            time.sleep(0.5)
            ret = self.wait(["ACK", "NAK"], num_bytes=3)
            if ret:
                self.log("HOS", "Timed out")
                return ret
            if debug_acks:
                self.log("DEV","{0}'ed".format(self.read_data.decode()))

            time.sleep(0.5)
            self.log("HOS", "Getting predictions.")
            time.sleep(0.5)
            self.send('get_output')
            time.sleep(1)
            self.recv(1000)
            time.sleep(1)
            self.log("DEV", "Prediction: {0}".format(self.read_data.decode()))

            output = [int(i) for i in self.read_data.decode().replace(' ','').split(',') if i]
            self.update_and_show(output)

            self.row_idx += 1
            count += 1
        return 0

if __name__=="__main__":
    emulator = SensorEmulator("COM7", 115200, "./test2.csv")
    emulator.emulate()

