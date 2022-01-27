from typing import Optional, List, Union, Iterable, Dict
from random import random, shuffle, randint
from time import perf_counter, sleep
import serial
from smbus2 import SMBus, i2c_msg

from kivy.properties import ObjectProperty

from pymoa.device import Device
from pymoa.device.digital import DigitalPort
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


class MODIOBase(DigitalPort):

    _config_props_ = ('dev_address', )

    dev_address: int = 0

    relay_0: bool = False

    relay_1: bool = False

    relay_2: bool = False

    relay_3: bool = False

    opto_0: bool = False

    opto_1: bool = False

    opto_2: bool = False

    opto_3: bool = False

    analog_0: float = 0

    analog_1: float = 0

    analog_2: float = 0

    analog_3: float = 0

    channel_names: List[str] = [
        f'relay_{i}' for i in range(4)] + [
        f'opto_{i}' for i in range(4)] + [
        f'analog_{i}' for i in range(4)]

    _relay_map: Dict[str, int] = {f'relay_{i}': i for i in range(4)}

    _analog_map: Dict[str, int] = {f'analog_{i}': i for i in range(4)}

    def __init__(self, dev_address: int = 0, **kwargs):
        self.dev_address = dev_address
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

    def _combine_write_args(
            self, high: Iterable[str], low: Iterable[str],
            kwargs: Dict[str, bool]):
        relay_map = self._relay_map
        value = 0

        for item in high:
            kwargs[item] = True
        for item in low:
            kwargs[item] = False

        for name, val in kwargs.items():
            if val:
                value |= 1 << relay_map[name]

        return value

    def update_write_data(self, result):
        value, t = result

        self.timestamp = t
        for i in range(4):
            setattr(self, f'relay_{i}', bool((1 << i) & value))

        self.dispatch('on_data_update', self)

    def update_read_data(self, result):
        opto_val, analog_vals, t = result

        self.timestamp = t
        if opto_val is not None:
            for i in range(4):
                setattr(self, f'opto_{i}', bool((1 << i) & opto_val))
        for name, value in analog_vals.items():
            setattr(self, name, value)

        self.dispatch('on_data_update', self)

    @staticmethod
    def get_time():
        return perf_counter()

    def write_states(
            self, high: Iterable[str] = (), low: Iterable[str] = (), **kwargs):
        raise NotImplementedError

    def _read_state(self, opto=True, analog_channels: Iterable[str] = ()):
        raise NotImplementedError

    @apply_executor(callback='update_read_data')
    def read_state(self, opto=True, analog_channels: Iterable[str] = ()):
        return self._read_state(opto, analog_channels)

    @apply_generator_executor(callback='update_read_data')
    def pump_state(self, opto=True, analog_channels: Iterable[str] = ()):
        while True:
            yield self._read_state(opto, analog_channels)


class MODIOBoard(MODIOBase):

    bus: Optional[SMBus] = None

    @apply_executor
    def open_device(self):
        self.bus = SMBus(1)

    @apply_executor
    def close_device(self):
        if self.bus is not None:
            self.bus.close()
            self.bus = None

    @apply_executor(callback='update_write_data')
    def write_states(
            self, high: Iterable[str] = (), low: Iterable[str] = (),
            **kwargs: bool):
        value = self._combine_write_args(high, low, kwargs)
        self.bus.write_byte_data(self.dev_address, 0x10, value)
        return value, self.get_time()

    def _read_state(self, opto=True, analog_channels: Iterable[str] = ()):
        if not opto and not analog_channels:
            raise ValueError('No channels specified to read')

        opto_val = None
        if opto:
            self.bus.write_byte(self.dev_address, 0x20)
            opto_val = self.bus.read_byte(self.dev_address)

        analog_vals = {}
        a_map = self._analog_map
        for chan in analog_channels:
            self.bus.write_byte(self.dev_address, 0x30 | (1 << a_map[chan]))
            msg = i2c_msg.read(self.dev_address, 2)
            self.bus.i2c_rdwr(msg)
            data = list(msg)
            assert len(data) == 2

            low, high = data
            val = 0
            for i in range(8):
                if (1 << (7 - i)) & low:
                    val |= 1 << i
            if high & 0x02:
                val |= 1 << 8
            if high & 0x01:
                val |= 1 << 9

            analog_vals[chan] = val * 3.3 / 1024

        return opto_val, analog_vals, self.get_time()


class VirtualMODIOBoard(MODIOBase):

    @apply_executor
    def open_device(self):
        pass

    @apply_executor
    def close_device(self):
        pass

    @apply_executor(callback='update_write_data')
    def write_states(
            self, high: Iterable[str] = (), low: Iterable[str] = (),
            **kwargs: bool):
        value = self._combine_write_args(high, low, kwargs)
        return value, self.get_time()

    def _read_state(self, opto=True, analog_channels: Iterable[str] = ()):
        if not opto and not analog_channels:
            raise ValueError('No channels specified to read')

        opto_val = None
        if opto:
            opto_val = randint(0, 0b1111)

        analog_vals = {}
        for chan in analog_channels:
            analog_vals[chan] = random() * 3.3

        return opto_val, analog_vals, self.get_time()
