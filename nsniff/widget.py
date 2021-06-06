import numpy as np
from typing import List, Dict
from matplotlib import cm
from kivy_trio.to_trio import kivy_run_in_async, mark, KivyEventCancelled
from pymoa_remote.threading import ThreadExecutor
from base_kivy_app.app import app_error
from kivy_garden.graph import Graph, ContourPlot

from kivy.properties import ObjectProperty, StringProperty, BooleanProperty, \
    NumericProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics.texture import Texture
from kivy.uix.widget import Widget

from nsniff.device import StratuscentSensor, VirtualStratuscentSensor, \
    StratuscentBase

__all__ = ('DeviceDisplay', )


class DeviceDisplay(BoxLayout):

    _config_props_ = ('com_port', 'virtual', 'log_z', 'auto_scale')

    com_port: str = StringProperty('')

    device: StratuscentBase = ObjectProperty(None, allownone=True, rebind=True)

    virtual = BooleanProperty(False)

    notes = StringProperty('')

    t = NumericProperty(0)

    done = False

    graph: Graph = None

    plot: ContourPlot = None

    _data: np.ndarray = None

    num_points: int = NumericProperty(0)

    _ts = None

    log_z = BooleanProperty(False)

    auto_scale = BooleanProperty(True)

    scale_tex = ObjectProperty(None, allownone=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fbind('log_z', self.recompute_bar)
        self.fbind('log_z', self.draw_data)
        self.fbind('auto_scale', self.draw_data)

    def create_plot(self, graph):
        self.graph = graph
        self.plot = plot = ContourPlot()
        plot.mag_filter = 'nearest'
        plot.min_filter = 'nearest'
        graph.add_plot(plot)

        self.recompute_bar()

    def recompute_bar(self, *args):
        tex = self.scale_tex = Texture.create(size=(250, 1), colorfmt='rgb')
        tex.mag_filter = tex.min_filter = 'linear'

        if self.log_z:
            points = (np.logspace(0, 1, 250, endpoint=True) - 1) / 9
        else:
            points = np.linspace(0, 1, 250, endpoint=True)

        data = cm.get_cmap()(points, bytes=True)[:, :3]
        tex.blit_buffer(data.tobytes(), colorfmt='rgb', bufferfmt='ubyte')

    def process_data(self, device: StratuscentBase):
        if self._ts is None:
            self._ts = device.timestamp
            self._data = np.empty(
                (len(device.sensors_data), 10), dtype=np.float)

        data = self._data
        self.t = device.timestamp - self._ts
        self.graph.xmax = max(device.timestamp - self._ts, 100)
        data[:, self.num_points] = device.sensors_data
        self.num_points += 1

        s = data.shape[1]
        if self.num_points == s:
            self._data = np.concatenate(
                (data, np.empty((len(device.sensors_data), s), dtype=np.float)),
                axis=1
            )

        self.draw_data()

    def draw_data(self, *args):
        data = self._data
        if data is None:
            return
        n = self.num_points

        if self.log_z:
            # min val will be 1
            min_val = np.min(data[:, :n], axis=1, keepdims=True) - 1
            scaled_data = data[:, :n] - min_val

            max_val = np.max(scaled_data[:, :n], axis=1, keepdims=True)
            max_val = np.log(max_val)
            scaled_data = np.log(scaled_data[:, :n])

            # min is now zero (log 1 == 0)
            zero_range = max_val[:, 0] == 0
            scaled_data[zero_range] = 0
            not_zero = np.logical_not(zero_range)

            scaled_data[not_zero] /= max_val[not_zero]
        else:
            min_val = np.min(data[:, :n], axis=1, keepdims=True)
            max_val = np.max(data[:, :n], axis=1, keepdims=True)
            scaled_data = data[:, :n] - min_val

            zero_range = min_val[:, 0] == max_val[:, 0]
            scaled_data[zero_range] = 0
            not_zero = np.logical_not(zero_range)

            scaled_data[not_zero] /= max_val[not_zero] - min_val[not_zero]

        np_data = cm.get_cmap()(scaled_data, bytes=True)
        self.plot.rgb_data = np_data[:, :, :3]

    async def run_device(self):
        async with ThreadExecutor() as executor:
            async with executor.remote_instance(self.device, 'sensor'):
                async with self.device as device:
                    async with device.read_sensor_values() as aiter:
                        async for _ in aiter:
                            if self.done:
                                break
                            self.process_data(device)

    @app_error
    @kivy_run_in_async
    def start(self):
        self._data = None
        self.num_points = 0
        self._ts = None
        self.done = False

        if self.virtual:
            cls = VirtualStratuscentSensor
        else:
            cls = StratuscentSensor

        self.device = cls(com_port=self.com_port)

        try:
            yield mark(self.run_device)
        except KivyEventCancelled:
            pass
        finally:
            self.device = None

    @app_error
    def stop(self):
        self.done = True
