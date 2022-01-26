from typing import Optional, List, Union
from random import random, shuffle, randint
from time import perf_counter, sleep
import serial

from kivy.properties import ObjectProperty

from pymoa.device import Device
from pymoa_remote.client import apply_executor, apply_generator_executor


class StratuscentBase(Device):

    _config_props_ = ('com_port', )

    com_port: str = ''

    precision_resistor: Union[int, float] = 0

    sensors_data: List[float] = []

    temp: float = ObjectProperty(0.)

    humidity: float = ObjectProperty(0.)

    device_id: str = ObjectProperty('')

    def __init__(self, com_port: str = '', **kwargs):
        self.com_port = com_port
        super().__init__(**kwargs)

    async def __aenter__(self):
        await self.open_device()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_device()

    def open_device(self):
        raise NotImplementedError

    def close_device(self):
        raise NotImplementedError

    def is_open(self):
        raise NotImplementedError

    def update_data(self, result):
        values, t = result

        self.timestamp = t
        self.precision_resistor = values[0]
        self.sensors_data = values[1:33]
        self.temp = values[33]
        self.humidity = values[34]
        self.device_id = values[35]

        self.dispatch('on_data_update', self)

    def read_sensor_values(self):
        raise NotImplementedError

    @staticmethod
    def get_data_header():
        return ['t'] + [f'{i}' for i in range(32)] + [
            'precision_resistor', 'temp', 'humidity', 'id']

    def get_data_row(self):
        return [self.timestamp] + self.sensors_data + [
            self.precision_resistor, self.temp, self.humidity, self.device_id]

    @staticmethod
    def get_time():
        return perf_counter()


class StratuscentSensor(StratuscentBase):

    device: Optional[serial.Serial] = None

    @apply_executor
    def open_device(self):
        ser = self.device = serial.Serial()
        ser.port = self.com_port
        ser.baudrate = 9600
        ser.bytesize = serial.EIGHTBITS  # number of bits per bytes
        ser.parity = serial.PARITY_NONE  # set parity check: no parity
        ser.stopbits = serial.STOPBITS_ONE  # number of stop bits
        # ser.timeout = None          #block read
        ser.timeout = 10  # non-block read
        # ser.timeout = 2              #timeout block read
        ser.xonxoff = False  # disable software flow control
        ser.rtscts = False  # disable hardware (RTS/CTS) flow control
        ser.dsrdtr = False  # disable hardware (DSR/DTR) flow control
        ser.writeTimeout = 10  # timeout for write
        ser.open()

        # discard first sample as it may be partial
        ser.readline()

    @apply_executor
    def close_device(self):
        if self.device is not None:
            self.device.close()
            self.device = None

    @apply_executor
    def is_open(self):
        return self.device.isOpen()

    @apply_generator_executor(callback='update_data')
    def read_sensor_values(self):
        dev = self.device
        while True:
            line = dev.readline()
            decoded_line = line.decode('utf-8').strip()
            items = decoded_line.split("\t")
            if len(items) != 36:
                raise ValueError('Did not read all 36 values')

            int_vals = list(map(int, items[:-3]))
            float_vals = [float(items[-3]), float(items[-2])]
            yield int_vals + float_vals + items[-1:], self.get_time()


class VirtualStratuscentSensor(StratuscentBase):

    @apply_executor
    def open_device(self):
        pass

    @apply_executor
    def close_device(self):
        pass

    @apply_executor
    def is_open(self):
        return True

    def random_walk(self, last_val, change, min_val, max_val):
        val = last_val + (change if random() < 0.5 else -change)
        return max(min(val, max_val), min_val)

    @apply_generator_executor(callback='update_data')
    def read_sensor_values(self):
        walk = self.random_walk
        rate = 1
        # how often we shuffle saturated sensors
        saturated_rate = 30
        p_resistor = 10000
        temp = random() * 15 + 15
        humidity = random() * 100
        dev_id = '012345_6789_0123456'
        sensors = [random() * 1_000_000 for _ in range(27)] + [1_000_000.] * 5
        shuffle(sensors)

        ts = self.get_time()
        n_samples = 0
        while True:
            yield [p_resistor] + sensors + [temp, humidity, dev_id], \
                  self.get_time()
            n_samples += 1

            # wait for new samples
            while (self.get_time() - ts) * rate < n_samples:
                sleep(.1)

            temp = walk(temp, .1, 15., 30.)
            humidity = walk(humidity, .05, 0., 100.)

            for i, val in enumerate(sensors):
                if val != 1_000_000.:
                    sensors[i] = walk(val, 10, 0, 1_000_000.)

            # shuffle the saturated samples at saturated_rate
            if n_samples % saturated_rate:
                continue
            for i, val in enumerate(sensors):
                if val == 1_000_000.:
                    i_switch = randint(0, 31)
                    temp_val = sensors[i_switch]
                    sensors[i_switch] = sensors[i]
                    sensors[i] = temp_val
                    # do only one shuffle
                    break
