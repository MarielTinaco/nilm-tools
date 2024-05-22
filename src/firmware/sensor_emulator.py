import datetime
import time
import serial
import csv

# Packet format is <Command 2B> <Len 2B> <Data Len-B> <Delimeter 2B>
# AA01 - Init
# AA02 - Send packet
# DD01 - Delimeter

class SensorEmulator():
    def __init__(self,
                 serial_port,
                 serial_baudrate,
                 aggregated_power_csv):
        self.serial = serial.Serial(
            port=serial_port,
            baudrate=serial_baudrate,
        )

        self.row_idx = 0
        self.col_idx = 0
        self.num_rows = 0
        self.raw_data = []
        self.data_bytes = []
        self.read_data = bytes()

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
            self.row_idx += 1
            count += 1

        return 0

if __name__=="__main__":
    emulator = SensorEmulator("COM7", 115200, "./test2.csv")
    emulator.emulate()

