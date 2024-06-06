import datetime
import time
import serial
import csv
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser(description="Sensor emulator")
    parser.add_argument("--port", default="COM7", type=str)
    parser.add_argument("--baudrate", default=115200, type=int, metavar="N")
    parser.add_argument("--test-set", default="./dataset/test_set.csv", type=str)
    parser.add_argument("--test-set-rms", default="./dataset/test_set_rms.csv", type=str)
    parser.add_argument("--test-set-states", default="./dataset/test_set_states.csv", type=str)
    parser.add_argument("--test-start-idx", default=0, type=int, metavar="N")
    parser.add_argument("--test-limit", default=1000, type=int, metavar="N")
    parser.add_argument("--liveplots", action="store_true")
    parser.add_argument("--plot-width", default=10, type=int, metavar="N")
    parser.add_argument("--plot-height", default=6, type=int, metavar="N")
    parser.add_argument("--plot-font-size", default=6, type=int, metavar="N")
    parser.add_argument("--debug-acks", action="store_true")
    return parser

class SensorEmulator():
    def __init__(self,
                 serial_port,
                 serial_baudrate,
                 aggregated_power_csv,
                 disaggregated_state_csv,
                 disaggregated_power_csv,
                 row_start = 0,
                 limit = 100,
                 buffer_size = 100,
                 liveplots = False,
                 plot_width = 10,
                 plot_height = 6,
                 plot_font_size = 6):
        self.serial = serial.Serial(
            port=serial_port,
            baudrate=serial_baudrate,
        )
        self.col_idx = 0
        self.num_rows = 0
        self.row_idx = row_start
        self.limit = limit
        self.buffer_size = buffer_size
        self.liveplots = liveplots
        self.plot_config = [plot_width, plot_height, plot_font_size]
        self.raw_data = []
        self.data_bytes = []
        self.disaggregated_states = []
        self.disaggregated_power = []
        self.read_data = bytes()
        self.output = [0 for _ in range(10)]

        self.states = np.zeros((5, self.buffer_size))
        self.rms = np.zeros((5, self.buffer_size))
        self.true_states = np.zeros((5, self.buffer_size))
        self.true_rms = np.zeros((5, self.buffer_size))

        if self.buffer_size > 0:
           self.plot_init()

        with open(aggregated_power_csv, 'r') as f:
            csv_reader = csv.reader(f)

            for i in csv_reader:
                self.raw_data.append(list(i))
                self.data_bytes.append("".join(str(j) + "," for j in i))
                self.num_rows += 1

        with open(disaggregated_state_csv, 'r') as f:
            csv_reader = csv.reader(f)

            for i in csv_reader:
                self.disaggregated_states.append([int(j) for j in list(i)])

        with open(disaggregated_power_csv, 'r') as f:
            csv_reader = csv.reader(f)

            for i in csv_reader:
                self.disaggregated_power.append([int(j) for j in list(i)])
            
        assert self.row_idx < self.num_rows

    def plot_init(self):
        if not self.liveplots:
            return
        matplotlib.rc('font', size=self.plot_config[2])
        matplotlib.rc('axes', titlesize=self.plot_config[2])

        self.fig, self.axes = plt.subplots(figsize=(self.plot_config[0], self.plot_config[1]), nrows=5, ncols=2)
        plt.sca(self.axes[0,0])
        self.axes[0,0].set_ylabel("State")
        self.axes[0,0].set_xlabel("Time")
        self.axes[0,0].set_title('Appliance 0') 
        self.axes[0,0].set_ylim([0,1.2])

        plt.sca(self.axes[1,0])
        self.axes[1,0].set_ylabel("State")
        self.axes[1,0].set_xlabel("Time")
        self.axes[1,0].set_title('Appliance 1')
        self.axes[1,0].set_ylim([0,1.2])

        plt.sca(self.axes[2,0])
        self.axes[2,0].set_ylabel("State")
        self.axes[2,0].set_xlabel("Time")
        self.axes[2,0].set_title('Appliance 2')
        self.axes[2,0].set_ylim([0,1.2])

        plt.sca(self.axes[3,0])
        self.axes[3,0].set_ylabel("State")
        self.axes[3,0].set_xlabel("Time")
        self.axes[3,0].set_title('Appliance 3')
        self.axes[3,0].set_ylim([0,1.2])

        plt.sca(self.axes[4,0])
        self.axes[4,0].set_ylabel("State")
        self.axes[4,0].set_xlabel("Time")
        self.axes[4,0].set_title('Appliance 4')
        self.axes[4,0].set_ylim([0,1.2])

        plt.sca(self.axes[0,1])
        self.axes[0,1].set_ylabel("RMS")
        self.axes[0,1].set_xlabel("Time")
        self.axes[0,1].set_title('Appliance 0')
        self.axes[0,1].set_ylim([0,150])

        plt.sca(self.axes[1,1])
        self.axes[1,1].set_ylabel("RMS")
        self.axes[1,1].set_xlabel("Time")
        self.axes[1,1].set_title('Appliance 1')
        self.axes[1,1].set_ylim([0,150])

        plt.sca(self.axes[2,1])
        self.axes[2,1].set_ylabel("RMS")
        self.axes[2,1].set_xlabel("Time")
        self.axes[2,1].set_title('Appliance 2')
        self.axes[2,1].set_ylim([0,150])

        plt.sca(self.axes[3,1])
        self.axes[3,1].set_ylabel("RMS")
        self.axes[3,1].set_xlabel("Time")
        self.axes[3,1].set_title('Appliance 3')
        self.axes[3,1].set_ylim([0,150])

        plt.sca(self.axes[4,1])
        self.axes[4,1].set_ylabel("RMS")
        self.axes[4,1].set_xlabel("Time")
        self.axes[4,1].set_title('Appliance 4')
        self.axes[4,1].set_ylim([0,150])

        self.fig.tight_layout()

        tvec = np.linspace(0, self.buffer_size - 1, self.buffer_size)

        self.line00, = self.axes[0,0].plot(tvec, self.states[0], c='b', label='Prediction')
        self.line10, = self.axes[1,0].plot(tvec, self.states[1], c='b')
        self.line20, = self.axes[2,0].plot(tvec, self.states[2], c='b')
        self.line30, = self.axes[3,0].plot(tvec, self.states[3], c='b')
        self.line40, = self.axes[4,0].plot(tvec, self.states[4], c='b')
        self.line01, = self.axes[0,1].plot(tvec, self.rms[0], c='b')
        self.line11, = self.axes[1,1].plot(tvec, self.rms[1], c='b')
        self.line21, = self.axes[2,1].plot(tvec, self.rms[2], c='b')
        self.line31, = self.axes[3,1].plot(tvec, self.rms[3], c='b')
        self.line41, = self.axes[4,1].plot(tvec, self.rms[4], c='b')

        self.line00_true, = self.axes[0,0].plot(tvec, self.true_states[0], c='r', label='Actual')
        self.line10_true, = self.axes[1,0].plot(tvec, self.true_states[1], c='r')
        self.line20_true, = self.axes[2,0].plot(tvec, self.true_states[2], c='r')
        self.line30_true, = self.axes[3,0].plot(tvec, self.true_states[3], c='r')
        self.line40_true, = self.axes[4,0].plot(tvec, self.true_states[4], c='r')
        self.line01_true, = self.axes[0,1].plot(tvec, self.true_rms[0], c='r')
        self.line11_true, = self.axes[1,1].plot(tvec, self.true_rms[1], c='r')
        self.line21_true, = self.axes[2,1].plot(tvec, self.true_rms[2], c='r')
        self.line31_true, = self.axes[3,1].plot(tvec, self.true_rms[3], c='r')
        self.line41_true, = self.axes[4,1].plot(tvec, self.true_rms[4], c='r')

        self.fig.legend(loc="upper left")
        plt.show(block=False)

    def plot_update_and_show(self, show=False):
        assert len(self.output) == 10

        states = np.array([self.output[:5]])
        rms = np.array([self.output[5:]])
        true_states = np.array([self.disaggregated_states[self.row_idx]])
        true_rms = np.array([self.disaggregated_power[self.row_idx]])

        self.states = np.concatenate((self.states, states.T), axis = 1)
        self.rms = np.concatenate((self.rms, rms.T), axis = 1)
        self.true_states = np.concatenate((self.true_states, true_states.T), axis = 1)
        self.true_rms = np.concatenate((self.true_rms, true_rms.T), axis = 1)
        self.states = np.delete(self.states, 0, 1)
        self.rms = np.delete(self.rms, 0, 1)
        self.true_states = np.delete(self.true_states, 0, 1)
        self.true_rms = np.delete(self.true_rms, 0, 1)

        self.line00.set_ydata(self.states[0])
        self.line10.set_ydata(self.states[1])
        self.line20.set_ydata(self.states[2])
        self.line30.set_ydata(self.states[3])
        self.line40.set_ydata(self.states[4])
        self.line01.set_ydata(self.rms[0])
        self.line11.set_ydata(self.rms[1])
        self.line21.set_ydata(self.rms[2])
        self.line31.set_ydata(self.rms[3])
        self.line41.set_ydata(self.rms[4])
        self.line00_true.set_ydata(self.true_states[0])
        self.line10_true.set_ydata(self.true_states[1])
        self.line20_true.set_ydata(self.true_states[2])
        self.line30_true.set_ydata(self.true_states[3])
        self.line40_true.set_ydata(self.true_states[4])
        self.line01_true.set_ydata(self.true_rms[0])
        self.line11_true.set_ydata(self.true_rms[1])
        self.line21_true.set_ydata(self.true_rms[2])
        self.line31_true.set_ydata(self.true_rms[3])
        self.line41_true.set_ydata(self.true_rms[4])

        self.axes[0,0].draw_artist(self.axes[0,0].patch)
        self.axes[1,0].draw_artist(self.axes[1,0].patch)
        self.axes[2,0].draw_artist(self.axes[2,0].patch)
        self.axes[3,0].draw_artist(self.axes[3,0].patch)
        self.axes[4,0].draw_artist(self.axes[4,0].patch)
        self.axes[0,1].draw_artist(self.axes[0,1].patch)
        self.axes[1,1].draw_artist(self.axes[1,1].patch)
        self.axes[2,1].draw_artist(self.axes[2,1].patch)
        self.axes[3,1].draw_artist(self.axes[3,1].patch)
        self.axes[4,1].draw_artist(self.axes[4,1].patch)

        self.axes[0,0].draw_artist(self.line00)
        self.axes[1,0].draw_artist(self.line10)
        self.axes[2,0].draw_artist(self.line20)
        self.axes[3,0].draw_artist(self.line30)
        self.axes[4,0].draw_artist(self.line40)
        self.axes[0,1].draw_artist(self.line01)
        self.axes[1,1].draw_artist(self.line11)
        self.axes[2,1].draw_artist(self.line21)
        self.axes[3,1].draw_artist(self.line31)
        self.axes[4,1].draw_artist(self.line41)
        self.axes[0,0].draw_artist(self.line00_true)
        self.axes[1,0].draw_artist(self.line10_true)
        self.axes[2,0].draw_artist(self.line20_true)
        self.axes[3,0].draw_artist(self.line30_true)
        self.axes[4,0].draw_artist(self.line40_true)
        self.axes[0,1].draw_artist(self.line01_true)
        self.axes[1,1].draw_artist(self.line11_true)
        self.axes[2,1].draw_artist(self.line21_true)
        self.axes[3,1].draw_artist(self.line31_true)
        self.axes[4,1].draw_artist(self.line41_true)

        self.fig.canvas.update()
        self.fig.canvas.flush_events()
        if show:
            plt.show()
        # else:
        #     plt.pause(0.1)

    def plot_reshow(self):
        plt.clf()
        self.plot_init()
        self.plot_update_and_show(True)
    
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
    
    def send(self, packet_id = None, text = None, delay = 0.2):
        if text is not None:
            self.serial.write(text.encode())
        if self.row_idx < self.num_rows:
            self.serial.write(self.build_packet(packet_id=packet_id))
        time.sleep(delay)
    
    def recv(self, num_bytes = 1, delay = 0.2):
        self.read_data = self.serial.read(num_bytes)
        time.sleep(delay)
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
    
    def log(self, tag, data):
        print("[{0}]".format(tag), end=' ')
        print(data)
    
    def emulate(self, debug_acks=False):
        ret = 0

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

        while (self.row_idx < self.num_rows and self.limit):
            self.log("HOS", "Sending input row {0}".format(self.row_idx))
            self.send(delay=0.5)
            ret = self.wait(["ACK", "NAK"], num_bytes=3)
            if ret:
                self.log("HOS", "Timed out")
                return ret
            if debug_acks:
                self.log("DEV","{0}'ed".format(self.read_data.decode()))

            self.log("HOS", "Starting inference.")
            self.send('infer', delay=0)
            ret = self.wait(["ACK", "NAK"], num_bytes=3)
            if ret:
                self.log("HOS", "Timed out")
                return ret
            if debug_acks:
                self.log("DEV","{0}'ed".format(self.read_data.decode()))

            self.log("HOS", "Getting predictions.")
            self.send('get_output', delay=0)
            self.recv(1000)
            self.log("DEV", "Prediction: {0}".format(self.read_data.decode()))

            self.output = [int(i) for i in self.read_data.decode().replace(' ','').split(',') if i]
            
            if self.liveplots:
                self.plot_update_and_show()

            self.row_idx += 1
            self.limit -= 1

        return 0

if __name__=="__main__":
    args = get_args().parse_args()

    emulator = SensorEmulator(serial_port=args.port, 
                              serial_baudrate=args.baudrate,
                              aggregated_power_csv=args.test_set,
                              disaggregated_state_csv=args.test_set_states,
                              disaggregated_power_csv=args.test_set_rms,
                              row_start=args.test_start_idx,
                              limit=args.test_limit,
                              liveplots=args.liveplots,
                              plot_font_size=args.plot_font_size,
                              plot_height=args.plot_height,
                              plot_width=args.plot_width)

    emulator.emulate(debug_acks=args.debug_acks)
    emulator.plot_reshow()