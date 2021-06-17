import numpy as np
from typing import List, Dict
from matplotlib import cm
from kivy_trio.to_trio import kivy_run_in_async, mark, KivyEventCancelled
from pymoa_remote.threading import ThreadExecutor
from base_kivy_app.app import app_error
from kivy_garden.graph import Graph, ContourPlot

from kivy.properties import ObjectProperty, StringProperty, BooleanProperty, \
    NumericProperty, ListProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics.texture import Texture
from kivy.factory import Factory
from kivy.uix.widget import Widget
from kivy_garden.graph import Graph

from nsniff.device import StratuscentSensor, VirtualStratuscentSensor, \
    StratuscentBase

__all__ = ('DeviceDisplay', )


class SniffGraph(Graph):

    dev_display: 'DeviceDisplay' = None

    def _scale_percent_pos(self, touch):
        w, h = self.view_size

        x, y = touch.pos
        x -= self.x + self.view_pos[0]
        y -= self.y + self.view_pos[1]

        x = x / w if w else 0
        y = y / h if h else 0

        return x, y

    def on_touch_down(self, touch):
        if super().on_touch_down(touch):
            return True
        if not self.collide_point(*touch.pos):
            return False

        x, y = self._scale_percent_pos(touch)
        if x > 1 or x < 0 or y > 1 or y < 0:
            return False
        touch.ud[f'sniff_graph.{self.uid}'] = x, y
        touch.grab(self)
        return True

    def on_touch_up(self, touch):
        if super().on_touch_up(touch):
            return True

        opos = touch.ud.get(f'sniff_graph.{self.uid}', None)
        if opos is not None:
            touch.ungrab(self)

        cpos = None
        if self.collide_point(*touch.pos):
            x, y = self._scale_percent_pos(touch)
            if x > 1 or x < 0 or y > 1 or y < 0:
                cpos = None
            else:
                cpos = x, y

        if opos or cpos:
            self.dev_display.set_range_from_pos(opos, cpos)
            return True
        return False


class DeviceDisplay(BoxLayout):

    _config_props_ = (
        'com_port', 'virtual', 'log_z', 'auto_range', 'global_range',
        'range_chan')

    com_port: str = StringProperty('')

    device: StratuscentBase = ObjectProperty(None, allownone=True, rebind=True)

    virtual = BooleanProperty(False)

    notes = StringProperty('')

    t = NumericProperty(0)

    t_start = NumericProperty(None, allownone=True)

    t_end = NumericProperty(None, allownone=True)

    done = False

    graph: Graph = None

    plot: ContourPlot = None

    _data: np.ndarray = None

    num_points: int = NumericProperty(0)

    _ts = None

    log_z = BooleanProperty(False)

    auto_range = BooleanProperty(True)

    scale_tex = ObjectProperty(None, allownone=True)

    global_range = BooleanProperty(False)

    min_val: np.ndarray = None

    max_val: np.ndarray = None

    range_chan: str = StringProperty('mouse')

    active_channels = ListProperty([True, ] * 32)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fbind('log_z', self.recompute_bar)
        self.fbind('log_z', self.draw_data)
        self.fbind('auto_range', self.draw_data)
        self.fbind('global_range', self.draw_data)
        self.fbind('t_start', self.draw_data)
        self.fbind('t_end', self.draw_data)
        self.fbind('active_channels', self.draw_data)

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
        data[:, self.num_points] = device.sensors_data
        self.num_points += 1

        s = data.shape[1]
        if self.num_points == s:
            self._data = np.concatenate(
                (data, np.empty((len(device.sensors_data), s), dtype=np.float)),
                axis=1
            )

        self.draw_data()

    def time_to_index(self, t):
        if not self.t:
            return 0
        return min(int(self.num_points * t / self.t), self.num_points - 1)

    def get_data_indices_range(self):
        s = 0
        if self.t_start:
            s = self.time_to_index(self.t_start)

        e = self.num_points
        if self.t_end:
            e = self.time_to_index(self.t_end) + 1

        return s, e

    def draw_data(self, *args):
        data = self._data
        if data is None:
            return
        inactive_channels = np.logical_not(
            np.asarray(self.active_channels, dtype=np.bool))
        s, e = self.get_data_indices_range()
        data = data[:, s:e]

        if self.auto_range or self.min_val is None or self.max_val is None:
            min_val = self.min_val = np.min(data, axis=1, keepdims=True)
            max_val = self.max_val = np.max(data, axis=1, keepdims=True)
        else:
            min_val = self.min_val
            max_val = self.max_val

        if self.global_range:
            # reduce to scalar
            min_val = np.min(min_val)
            max_val = np.max(max_val)
            zero_range = min_val == max_val
            print(min_val, max_val, zero_range)
        else:
            zero_range = min_val[:, 0] == max_val[:, 0]

        scaled_data = np.clip(data, min_val, max_val) - min_val
        max_val = max_val - min_val

        if self.log_z:
            # min val will be 1 (log 1 == 0)
            max_val = np.log(max_val + 1)
            scaled_data = np.log(scaled_data + 1)

        scaled_data[zero_range] = 0
        not_zero = np.logical_not(zero_range)

        scaled_data[not_zero] /= max_val[not_zero]

        scaled_data[inactive_channels, :] = 0

        np_data = cm.get_cmap()(scaled_data, bytes=True)
        self.plot.rgb_data = np_data[:, :, :3]

    def set_range_from_pos(self, open_pos, close_pos):
        data = self._data
        if data is None or self.min_val is None or self.max_val is None:
            return
        data_s, data_e = self.get_data_indices_range()
        data = data[:, data_s:data_e]

        chan = self.range_chan
        n = data.shape[1]
        s = 0
        e = n - 1
        if open_pos is not None:
            x, y = open_pos
            s = min(int(x * n), n - 1)

        if close_pos is not None:
            x, y = close_pos
            e = min(int(x * n), n - 1)

        if s > e:
            s, e = e, s
        e += 1

        if chan == 'all':
            self.min_val = np.min(data[:, s:e], axis=1, keepdims=True)
            self.max_val = np.max(data[:, s:e], axis=1, keepdims=True)
        else:
            if chan == 'mouse':
                _, y = open_pos or close_pos
                i = min(int(y * 32), 31)
            else:
                i = int(chan) - 1
            self.min_val[i, 0] = np.min(data[i, s:e])
            self.max_val[i, 0] = np.max(data[i, s:e])

        self.draw_data()

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
        self.min_val = self.max_val = None
        self.t_start = None
        self.t_end = None
        self.t = 0

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

    def add_channel_selection(self, container):
        ChannelActiveButton = Factory.ChannelActiveButton
        for i in range(32):
            widget = ChannelActiveButton()
            widget.dev = self
            widget.channel = i
            widget.text = str(i)
            container.add_widget(widget)
