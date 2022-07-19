import csv
import numpy as np
from threading import Thread
import trio
from math import ceil
import pathlib
from time import perf_counter, sleep
import os.path
from typing import List, Dict, Optional, Tuple
from matplotlib import cm
from kivy_trio.to_trio import kivy_run_in_async, mark, KivyEventCancelled
from kivy_trio.to_kivy import AsyncKivyEventQueue
from pymoa_remote.threading import ThreadExecutor
from pymoa_remote.socket.websocket_client import WebSocketExecutor
from pymoa_remote.client import Executor
from base_kivy_app.app import app_error
from base_kivy_app.utils import pretty_time, yaml_dumps
from tree_config import read_config_from_object
import nixio as nix

from kivy.metrics import dp
from kivy.properties import ObjectProperty, StringProperty, BooleanProperty, \
    NumericProperty, ListProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.core.audio import SoundLoader
from kivy.graphics.texture import Texture
from kivy.factory import Factory
from kivy.uix.widget import Widget
from kivy.event import EventDispatcher
from kivy_garden.graph import Graph, ContourPlot, LinePlot

from nsniff.device import StratuscentSensor, VirtualStratuscentSensor, \
    StratuscentBase, MODIOBase, MODIOBoard, VirtualMODIOBoard, MFCBase, MFC, \
    VirtualMFC

__all__ = ('DeviceDisplay', 'ValveBoardWidget', 'MFCWidget', 'ExperimentStages')


ProtocolItem = Tuple[float, List[Optional[bool]], List[Optional[float]]]
FlatProtocolItem = Tuple[
    float, List[Tuple['ValveBoardWidget', Dict[str, bool]]],
    List[Tuple['MFCWidget', float]]
]


class SniffGraph(Graph):

    dev_display: 'DeviceDisplay' = None

    pos_label: Widget = None

    visible = BooleanProperty(False)

    is_3d = True

    def _scale_percent_pos(self, pos):
        w, h = self.view_size

        x, y = pos
        x -= self.x + self.view_pos[0]
        y -= self.y + self.view_pos[1]

        x = x / w if w else 0
        y = y / h if h else 0

        return x, y

    def show_pos_label(self):
        label = self.pos_label
        if label is None:
            label = self.pos_label = Factory.GraphPosLabel()
        if label.parent is None:
            from kivy.core.window import Window
            Window.add_widget(label)

    def hide_pos_label(self):
        label = self.pos_label
        if label is not None and label.parent is not None:
            from kivy.core.window import Window
            Window.remove_widget(label)

    def on_kv_post(self, base_widget):
        from kivy.core.window import Window
        Window.fbind('mouse_pos', self._set_hover_label)

    def _set_hover_label(self, *args):
        from kivy.core.window import Window
        pos = self.to_parent(*self.to_widget(*Window.mouse_pos))

        if not self.visible or \
                len(Window.children) > 1 and \
                Window.children[0] is not self.pos_label or \
                not self.collide_point(*pos):
            self.hide_pos_label()
            return

        x, y = self._scale_percent_pos(pos)
        if x > 1 or x < 0 or y > 1 or y < 0:
            self.hide_pos_label()
            return

        self.show_pos_label()
        text = self.dev_display.get_data_from_graph_pos(x, y, self.is_3d)
        if text:
            self.pos_label.text = text
            x_pos, y_pos = Window.mouse_pos
            self.pos_label.pos = min(
                x_pos + dp(20), Window.width - dp(200)), y_pos + dp(20)
        else:
            self.hide_pos_label()

    def on_touch_down(self, touch):
        if super().on_touch_down(touch):
            return True
        if not self.collide_point(*touch.pos):
            return False

        x, y = self._scale_percent_pos(touch.pos)
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
            x, y = self._scale_percent_pos(touch.pos)
            if x > 1 or x < 0 or y > 1 or y < 0:
                cpos = None
            else:
                cpos = x, y

        if opos or cpos:
            self.dev_display.set_range_from_pos(opos, cpos, self.is_3d)
            return True
        return False


class ExecuteDevice:

    __events__ = ('on_data_update', )

    _config_props_ = (
        'virtual', 'remote_server', 'remote_port', 'unique_dev_id',
    )

    remote_server: str = StringProperty('')

    remote_port: int = NumericProperty(0)

    device = None

    virtual = BooleanProperty(False)

    unique_dev_id: str = StringProperty('')

    log_file: Optional[nix.File] = None

    n_time_log_sec: int = 1

    log_time_arr: Optional[nix.DataArray] = None

    is_running: bool = BooleanProperty(False)

    def on_data_update(self, *args):
        pass

    async def _run_device(self, executor):
        raise NotImplementedError

    async def run_device(self):
        if self.remote_server and not self.virtual:
            async with trio.open_nursery() as nursery:
                async with WebSocketExecutor(
                        nursery=nursery, server=self.remote_server,
                        port=self.remote_port) as executor:
                    async with executor.remote_instance(
                            self.device, self.unique_dev_id):
                        async with self.device:
                            await self._run_device(executor)
        else:
            async with ThreadExecutor() as executor:
                async with executor.remote_instance(
                        self.device, self.unique_dev_id):
                    async with self.device:
                        await self._run_device(executor)

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    @staticmethod
    def _get_next_array_num(block: nix.Block, prefix: str) -> int:
        n = len(
            [arr for arr in block.data_arrays if arr.name.startswith(prefix)])
        return n

    def _get_create_dev_block(self, dev_type: str) -> nix.Block:
        f = self.log_file
        name = f'{dev_type}_dev_{self.unique_dev_id}'
        if name not in f.blocks:
            f.create_block(name, dev_type)
        return f.blocks[name]

    def start_data_logging(self):
        raise NotImplementedError

    def stop_data_logging(self):
        raise NotImplementedError

    def set_logging_file(self, log_file: Optional[nix.File] = None):
        self.log_file = log_file
        if self.device is not None:
            self.start_data_logging()

    def clear_logging_file(self):
        self.stop_data_logging()
        self.log_file = None

    def _create_time_log_array(
            self, block: nix.Block, dev_type: str, arr_n: int) -> nix.DataArray:
        return block.create_data_array(
            f'{dev_type}_times_{arr_n}', 'times', dtype=np.float64,
            shape=(0, 3))

    async def _log_times(self, executor: Executor):
        if self.log_time_arr is None:
            return
        times = [t / 1e9 for t in await executor.get_echo_clock()]
        self.log_time_arr.append([times], 0)

    def _set_array_metadata(
            self, block: nix.Block, array: nix.DataArray) -> nix.Section:
        sec = self.log_file.create_section(
            f'{block.name}.{array.name}', 'metadata')

        sec['config'] = yaml_dumps(read_config_from_object(self))
        array.metadata = sec
        return sec


class DeviceDisplay(BoxLayout, ExecuteDevice):

    _config_props_ = (
        'com_port', 'log_z', 'auto_range', 'global_range',
        'range_chan', 'n_channels')

    com_port: str = StringProperty('')

    device: Optional[StratuscentBase] = ObjectProperty(
        None, allownone=True, rebind=True)

    n_channels = 32

    t0 = NumericProperty(0)

    t = NumericProperty(0)

    t_start = NumericProperty(None, allownone=True)

    t_end = NumericProperty(None, allownone=True)

    t_last = NumericProperty(None, allownone=True)

    done = False

    graph_3d: Graph = None

    plot_3d: ContourPlot = None

    graph_2d: Graph = None

    plots_2d: List[LinePlot] = []

    _data: Optional[np.ndarray] = None

    num_points: int = NumericProperty(0)

    log_z = BooleanProperty(False)

    auto_range = BooleanProperty(True)

    scale_tex = ObjectProperty(None, allownone=True)

    global_range = BooleanProperty(False)

    min_val: Optional[np.ndarray] = None

    max_val: Optional[np.ndarray] = None

    range_chan: str = StringProperty('mouse')

    active_channels = ListProperty([True, ] * n_channels)

    channels_stats = []

    _draw_trigger = None

    _t_trigger = None

    _plot_colors = []

    _event_plots: Tuple[List[LinePlot], List[LinePlot]] = ([], [])

    _event_plots_trigger = None

    data_arr: Optional[nix.DataArray] = None

    _item_dtype: np.dtype = np.dtype([
        ('timestamp', np.float64), ('data', np.float64, 32),
        ('temp', np.float64), ('humidity', np.float64),
        ('local_time', np.float64)]
    )

    def __init__(self, **kwargs):
        self._plot_colors = cm.get_cmap('tab20').colors + \
                            cm.get_cmap('tab20b').colors
        super().__init__(**kwargs)
        self._event_plots = [], []
        self._event_plots_trigger = Clock.create_trigger(
            self._move_events_to_top)

        self._draw_trigger = Clock.create_trigger(self.draw_data)
        self.fbind('log_z', self.recompute_bar)
        self.fbind('log_z', self._draw_trigger)
        self.fbind('auto_range', self._draw_trigger)
        self.fbind('global_range', self._draw_trigger)
        self.fbind('t_start', self._draw_trigger)
        self.fbind('t_end', self._draw_trigger)
        self.fbind('t_last', self._draw_trigger)
        self.fbind('active_channels', self._draw_trigger)

        self._t_trigger = Clock.create_trigger(self._set_graph_t_axis)
        self.fbind('t_start', self._t_trigger)
        self.fbind('t_end', self._t_trigger)
        self.fbind('t_last', self._t_trigger)
        self.fbind('t0', self._t_trigger)
        self.fbind('t', self._t_trigger)

    def _set_graph_t_axis(self, *args):
        xmax = self.t_end if self.t_end is not None else self.t
        if self.t_start is not None:
            xmin = self.t_start
        elif self.t_last is not None:
            xmin = xmax - self.t_last
        else:
            xmin = self.t0

        if xmin > xmax:
            xmin = xmax

        self.graph_2d.xmin = xmin
        self.graph_2d.xmax = xmax
        self.graph_3d.xmin = max(min(xmin, self.t), self.t0)
        self.graph_3d.xmax = max(min(xmax, self.t), self.t0)

    def _move_events_to_top(self, *args):
        plots2, plots3 = self._event_plots
        graph2 = self.graph_2d
        graph3 = self.graph_3d

        for plot in plots2:
            graph2.remove_plot(plot)
            graph2.add_plot(plot)
        for plot in plots3:
            graph3.remove_plot(plot)
            graph3.add_plot(plot)

    def create_plot(self, graph_3d, graph_2d):
        self.graph_3d = graph_3d
        self.plot_3d = plot = ContourPlot()
        plot.mag_filter = 'nearest'
        plot.min_filter = 'nearest'
        graph_3d.add_plot(plot)

        self.recompute_bar()

        self.graph_2d = graph_2d
        self.plots_2d = plots = []
        for i in range(self.n_channels):
            plot = LinePlot(color=self._plot_colors[i], line_width=dp(2))
            graph_2d.add_plot(plot)
            plots.append(plot)

    def show_hide_channel(self, channel, visible):
        self.active_channels[channel] = visible
        if visible:
            self.graph_2d.add_plot(self.plots_2d[channel])
            self._event_plots_trigger()
        else:
            self.graph_2d.remove_plot(self.plots_2d[channel])

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
        arr = self.data_arr
        if arr is not None:
            rec = np.rec.array([
                device.timestamp, device.sensors_data, device.temp,
                device.humidity, device.local_time], dtype=self._item_dtype)
            arr.append(rec)

        if self._data is None:
            self.t0 = device.timestamp
            self._data = np.empty(
                (len(device.sensors_data) + 1, 10), dtype=np.float)

        data = self._data
        self.t = device.timestamp
        data[:self.n_channels, self.num_points] = device.sensors_data
        data[self.n_channels, self.num_points] = device.timestamp
        self.num_points += 1

        s = data.shape[1]
        if self.num_points == s:
            self._data = np.concatenate(
                (data,
                 np.empty((len(device.sensors_data) + 1, s), dtype=np.float)),
                axis=1
            )

        self._draw_trigger()

    def time_to_index(self, t):
        if self._data is None:
            return 0
        n = self.num_points
        t0 = self.t0
        total_t = self._data[self.n_channels, n - 1] - t0
        if not total_t:
            return 0

        return max(min(int(n * (t - t0) / total_t), n - 1), 0)

    def get_data_from_graph_pos(self, x_frac, y_frac, plot_3d):
        data = self.get_visible_data()
        if data is None:
            return

        n = data.shape[1]
        i = min(int(x_frac * n), n - 1)
        t = (self.graph_3d.xmax - self.graph_3d.xmin) * x_frac + \
            self.graph_3d.xmin
        if plot_3d:
            channel = min(int(y_frac * self.n_channels), self.n_channels - 1)
            value = data[channel, i]
            return f'{t:0.1f}, {channel + 1}, {value:0.1f}'

        if self.range_chan in ('mouse', 'all'):
            if self.log_z:
                y_frac = (np.power(10, y_frac) - 1) / 9
            y = (self.graph_2d.ymax - self.graph_2d.ymin) * y_frac + \
                self.graph_2d.ymin
            return f'{t:0.1f}, {y:0.3f}'

        channel = int(self.range_chan)
        value = data[channel - 1, i]
        return f'{t:0.1f}, {channel}, {value:0.1f}'

    def get_data_indices_range(self):
        s = 0
        if self.t_start:
            s = self.time_to_index(self.t_start)

        e = self.num_points
        if self.t_end:
            e = self.time_to_index(self.t_end) + 1

        if not self.t_start and self.t_last:
            if self.t_end:
                s = self.time_to_index(self.t_end - self.t_last)
            else:
                s = self.time_to_index(self.t - self.t_last)

        return s, e

    def get_visible_data(self):
        data = self._data
        if data is None:
            return None
        s, e = self.get_data_indices_range()
        return data[:, s:e]

    def draw_data(self, *args):
        data = self.get_visible_data()
        if data is None:
            return
        n_channels = self.n_channels
        inactive_channels = np.logical_not(
            np.asarray(self.active_channels, dtype=np.bool))

        if self.auto_range or self.min_val is None or self.max_val is None:
            min_val = self.min_val = np.min(
                data[:n_channels, :], axis=1, keepdims=True)
            max_val = self.max_val = np.max(
                data[:n_channels, :], axis=1, keepdims=True)
            for widget, mn, mx in zip(
                    self.channels_stats, min_val[:, 0], max_val[:, 0]):
                widget.min_val = mn.item()
                widget.max_val = mx.item()
        else:
            min_val = self.min_val
            max_val = self.max_val

        if self.global_range:
            # reduce to scalar
            min_val[:, 0] = np.min(min_val)
            max_val[:, 0] = np.max(max_val)
        zero_range = min_val[:, 0] == max_val[:, 0]

        scaled_data = np.clip(data[:n_channels, :], min_val, max_val) - min_val
        max_val = max_val - min_val

        scaled_data[inactive_channels, :] = 0
        scaled_data[zero_range, :] = 0
        not_zero = np.logical_not(np.logical_or(zero_range, inactive_channels))

        times = data[n_channels, :].tolist()
        log_z = self.log_z
        for i, plot in enumerate(self.plots_2d):
            if not_zero[i]:
                d = scaled_data[i, :] / max_val[i, 0]
                if log_z:
                    d = d * .9 + .1
                plot.points = list(zip(times, d.tolist()))
            else:
                plot.points = []

        if np.any(not_zero):
            if log_z:
                # min val will be 1 (log 1 == 0)
                max_val = np.log10(max_val + 1)
                scaled_data[not_zero] = np.log10(scaled_data[not_zero] + 1)

            scaled_data[not_zero] /= max_val[not_zero]

        np_data = cm.get_cmap()(scaled_data, bytes=True)
        self.plot_3d.rgb_data = np_data[:, :, :3]

    def set_range_from_pos(self, open_pos, close_pos, plot_3d):
        data = self.get_visible_data()
        if data is None or self.min_val is None or self.max_val is None:
            return

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

        if chan == 'all' or chan == 'mouse' and not plot_3d:
            self.min_val = np.min(
                data[:self.n_channels, s:e], axis=1, keepdims=True)
            self.max_val = np.max(
                data[:self.n_channels, s:e], axis=1, keepdims=True)
            for widget, mn, mx in zip(
                    self.channels_stats, self.min_val[:, 0],
                    self.max_val[:, 0]):
                widget.min_val = mn.item()
                widget.max_val = mx.item()
        else:
            if chan == 'mouse':
                _, y = open_pos or close_pos
                i = min(int(y * self.n_channels), self.n_channels - 1)
            else:
                i = int(chan) - 1
            self.min_val[i, 0] = np.min(data[i, s:e])
            self.max_val[i, 0] = np.max(data[i, s:e])
            widget = self.channels_stats[i]
            widget.min_val = self.min_val[i, 0].item()
            widget.max_val = self.max_val[i, 0].item()

        self._draw_trigger()

    async def _run_device(self, executor: Executor):
        device = self.device
        rate = 1 / self.n_time_log_sec
        count = 0

        async with device.read_sensor_values() as aiter:
            ts = perf_counter()
            async for _ in aiter:
                if self.done:
                    break
                self.process_data(device)

                if perf_counter() - ts > rate * count:
                    await self._log_times(executor)
                    count += 1

    @app_error
    @kivy_run_in_async
    def start(self):
        for graph, plots in zip(
                (self.graph_2d, self.graph_3d), self._event_plots):
            for plot in plots:
                graph.remove_plot(plot)
        self._event_plots = [], []

        self._data = None
        self.num_points = 0
        self.t0 = 0
        self.done = False
        self.min_val = self.max_val = None
        self.t_start = None
        self.t_end = None
        self.t = 0
        self.is_running = True

        if self.virtual:
            cls = VirtualStratuscentSensor
        else:
            cls = StratuscentSensor

        self.device = cls(com_port=self.com_port)
        self.device.fbind('on_data_update', self.dispatch, 'on_data_update')
        self.start_data_logging()

        try:
            yield mark(self.run_device)
        except KivyEventCancelled:
            pass
        finally:
            self.device = None
            self.stop_data_logging()
            self.is_running = False

    @app_error
    def stop(self):
        self.done = True

    def add_channel_selection(self, container):
        ChannelControl = Factory.ChannelControl
        channels = self.channels_stats = []
        for i in range(self.n_channels):
            widget = ChannelControl()
            widget.dev = self
            widget.channel = i
            widget.plot_color = self._plot_colors[i]
            container.add_widget(widget)
            channels.append(widget)

    def set_channel_min_val(self, channel, value):
        if self.min_val is None:
            return

        value = float(value)
        self.min_val[channel, 0] = value
        self._draw_trigger()

    def set_channel_max_val(self, channel, value):
        if self.max_val is None:
            return

        value = float(value)
        self.max_val[channel, 0] = value
        self._draw_trigger()

    def add_event(self, t: Optional[float], name: str):
        if t is None:
            t = self.t
        p = LinePlot(color=(0, 0, 0), line_width=dp(3))
        p.points = [(t, .1), (t, 1)]
        self.graph_2d.add_plot(p)
        self._event_plots[0].append(p)

        p = LinePlot(color=(0, 0, 0), line_width=dp(3))
        p.points = [(t, 0), (t, self.graph_3d.ymax)]
        self.graph_3d.add_plot(p)
        self._event_plots[1].append(p)

    def start_data_logging(self):
        if self.log_file is None:
            return

        block = self._get_create_dev_block('stratuscent')
        n = self._get_next_array_num(block, 'sensor')
        self.data_arr = block.create_data_array(
            f'sensor_{n}', 'stratuscent', dtype=self._item_dtype, data=[])
        self.log_time_arr = self._create_time_log_array(block, 'sensor', n)
        self._set_array_metadata(block, self.data_arr)

    def stop_data_logging(self):
        self.data_arr = None
        self.log_time_arr = None


class ValveBoardWidget(BoxLayout, ExecuteDevice):

    _config_props_ = ('dev_address', 'com_port')

    dev_address: int = NumericProperty(0)

    com_port: str = StringProperty('')

    device: Optional[MODIOBase] = ObjectProperty(
        None, allownone=True, rebind=True)

    _event_queue: Optional[AsyncKivyEventQueue] = None

    data_arr: Optional[nix.DataArray] = None

    _item_dtype: np.dtype = np.dtype([
        ('timestamp', np.float64), ('data', np.uint8, 4),
        ('local_time', np.float64)]
    )

    async def _run_device(self, executor: Executor):
        device = self.device
        rate = 1 / self.n_time_log_sec
        async with self._event_queue as queue:
            while True:
                with trio.move_on_after(rate) as cancel_scope:
                    async for low, high, kwargs in queue:
                        cancel_scope.shield = True
                        await device.write_states(high, low, **kwargs)
                        arr = self.data_arr
                        if arr is not None:
                            rec = np.rec.array([
                                device.timestamp, [
                                    device.relay_0, device.relay_1,
                                    device.relay_2, device.relay_3],
                                device.local_time
                            ], dtype=self._item_dtype)
                            arr.append(rec)
                        cancel_scope.shield = False

                if cancel_scope.cancelled_caught:
                    await self._log_times(executor)
                else:
                    break

    @app_error
    @kivy_run_in_async
    def start(self):
        if self.virtual:
            cls = VirtualMODIOBoard
        else:
            cls = MODIOBoard

        self.is_running = True
        self.device = cls(dev_address=self.dev_address, com_port=self.com_port)
        self.device.fbind('on_data_update', self.dispatch, 'on_data_update')
        self._event_queue = AsyncKivyEventQueue()
        self.start_data_logging()

        try:
            yield mark(self.run_device)
        except KivyEventCancelled:
            pass
        finally:
            self._event_queue.stop()
            self._event_queue = None
            self.device = None
            self.stop_data_logging()
            self.is_running = False

    @app_error
    def stop(self):
        if self._event_queue is not None:
            self._event_queue.stop()

    def set_valves(self, low=(), high=(), **kwargs: bool):
        self._event_queue.add_item(low, high, kwargs)

    def start_data_logging(self):
        if self.log_file is None:
            return

        block = self._get_create_dev_block('mod-io-relays')
        n = self._get_next_array_num(block, 'relays')
        self.data_arr = block.create_data_array(
            f'relays_{n}', 'mod-io-relays', dtype=self._item_dtype, data=[])
        self.log_time_arr = self._create_time_log_array(block, 'relays', n)
        self._set_array_metadata(block, self.data_arr)

    def stop_data_logging(self):
        self.data_arr = None
        self.log_time_arr = None


class MFCWidget(BoxLayout, ExecuteDevice):

    _config_props_ = ('dev_address', 'com_port')

    dev_address: int = NumericProperty(0)

    com_port: str = StringProperty('')

    device: Optional[MFCBase] = ObjectProperty(
        None, allownone=True, rebind=True)

    _event_queue: List[float] = []

    _done = False

    data_arr: Optional[nix.DataArray] = None

    _item_dtype: np.dtype = np.dtype([
        ('timestamp', np.float64), ('data', np.float64),
        ('local_time', np.float64)]
    )

    async def _run_device(self, executor: Executor):
        device = self.device
        queue = self._event_queue
        rate = 1 / self.n_time_log_sec
        count = 0
        ts = perf_counter()

        while not self._done:
            i = len(queue)
            if i:
                await device.write_state(queue[i - 1])
                del queue[:i]
            await device.read_state()

            arr = self.data_arr
            if arr is not None:
                rec = np.rec.array([
                    device.timestamp, device.state, device.local_time
                ], dtype=self._item_dtype)
                arr.append(rec)

            if perf_counter() - ts > rate * count:
                await self._log_times(executor)
                count += 1

    @app_error
    @kivy_run_in_async
    def start(self):
        if self.virtual:
            cls = VirtualMFC
        else:
            cls = MFC

        self.is_running = True
        self.device = cls(dev_address=self.dev_address, com_port=self.com_port)
        self.device.fbind('on_data_update', self.dispatch, 'on_data_update')
        self._event_queue = []
        self._done = False
        self.start_data_logging()

        try:
            yield mark(self.run_device)
        except KivyEventCancelled:
            pass
        finally:
            self._done = True
            self._event_queue = []
            self.device = None
            self.stop_data_logging()
            self.is_running = False

    @app_error
    def stop(self):
        self._done = True

    def set_value(self, value):
        self._event_queue.append(value)

    def start_data_logging(self):
        if self.log_file is None:
            return

        block = self._get_create_dev_block('mfc')
        n = self._get_next_array_num(block, 'mfc')
        self.data_arr = block.create_data_array(
            f'mfc_{n}', 'mfc', dtype=self._item_dtype, data=[])
        self.log_time_arr = self._create_time_log_array(block, 'mfc', n)
        self._set_array_metadata(block, self.data_arr)

    def stop_data_logging(self):
        self.data_arr = None
        self.log_time_arr = None


class ExperimentStages(EventDispatcher):

    _config_props_ = ('last_directory', )

    playing: bool = BooleanProperty(False)

    remaining_time: str = StringProperty('00:00:00.0')

    stage_remaining_time: str = StringProperty('00:00:00.0')

    last_directory: str = StringProperty('')

    _timer_ts: float = 0

    _total_time: float = 0

    _total_stage_time: float = 0

    _sound_thread: Thread = None

    _clock_event = None

    _protocol_clock_event = None

    _app = None

    _log_file: Optional[nix.File] = None

    _exp_protocol_text: str = ''

    filename: str = StringProperty('')

    protocol: Dict[str, List[ProtocolItem]] = {}

    n_stages: int = NumericProperty(0)

    stage_i: int = NumericProperty(0)

    _col_names: Tuple[List[str], List[str]] = ()

    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self._app = app
        self.protocol = {}

    def set_logging_file(self, log_file: nix.File):
        assert not self.playing
        self._log_file = log_file

    def clear_logging_file(self):
        assert not self.playing
        self._log_file = None

    def _run_protocol(
            self, protocol: List[FlatProtocolItem], alarm: bool,
            log_array: Optional[nix.DataArray],
            sensors: List[DeviceDisplay]):
        self._total_time = sum(item[0] for item in protocol)
        self._total_stage_time = 0
        rem_time = 0.
        i = -1
        self.n_stages = n = len(protocol)
        ts = 0
        self.stage_i = 0

        def protocol_callback(*args):
            nonlocal i, rem_time, ts, n

            t = perf_counter()
            if t - ts < rem_time:
                return

            i += 1
            if i == n:
                if alarm:
                    thread = self._sound_thread = Thread(
                        target=self._make_sound)
                    thread.start()

                self.stop()
                self._update_clock()
                return
            # update only if not stopping
            self.stage_i = i + 1

            if log_array is not None:
                log_array.append([perf_counter()])

            dur, valves, mfcs = protocol[i]
            rem_time += dur
            for board, values in valves:
                board.set_valves(**values)
            for board, value in mfcs:
                board.set_value(value)

            self._total_stage_time = rem_time

            for sensor in sensors:
                if sensor.is_running:
                    sensor.add_event(None, str(i))

        self._protocol_clock_event = Clock.schedule_interval(
            protocol_callback, 0)
        self._clock_event = Clock.schedule_interval(self._update_clock, .25)

        self._timer_ts = ts = perf_counter()
        # this can stop above events
        protocol_callback()

        self._update_clock()

    def _flatten_protocol(
            self, protocol: List[ProtocolItem]) -> List[FlatProtocolItem]:
        valves: List[ValveBoardWidget] = self._app.valve_boards
        mfcs: List[MFCWidget] = self._app.mfcs

        stages = []
        for dur, valve_states, mfc_vals in protocol:
            valve_groups = [
                valve_states[i * 4: (i + 1) * 4]
                for i in range(int(ceil(len(valve_states) / 4)))
            ]
            if len(valve_groups) != len(valves):
                raise ValueError(
                    'The number of valve columns is not the same as valves')
            if len(mfc_vals) != len(mfcs):
                raise ValueError(
                    'The number of MFC columns is not the same as MFCs')

            prepped_valves = []
            for valve_board, states in zip(valves, valve_groups):
                relays = {
                    f'relay_{i}': val for i, val in enumerate(states)
                    if val is not None
                }
                if relays:
                    prepped_valves.append((valve_board, relays))

            prepped_mfcs = [
                (mfc, val) for mfc, val in zip(mfcs, mfc_vals)
                if val is not None
            ]
            stages.append((dur, prepped_valves, prepped_mfcs))

        return stages

    @app_error
    def start(self, key: str, alarm: bool):
        from nsniff.main import NSniffApp
        self.playing = True
        app: NSniffApp = self._app

        try:
            dev: ExecuteDevice
            for dev in app.valve_boards + app.mfcs:
                if not dev.is_running:
                    raise TypeError('Not all valves/MFCs have been started')

            app.dump_app_settings_to_file()
            config = pathlib.Path(
                os.path.join(app.data_path, app.yaml_config_path)).read_text()

            if key not in self.protocol:
                raise ValueError('Protocol not available')
            protocol = self._flatten_protocol(self.protocol[key])
            if not len(protocol):
                raise ValueError('No protocol available')
        except Exception:
            self.stop()
            raise

        f = self._log_file
        log_array = None
        if f is not None:
            if 'experiment' not in f.blocks:
                f.create_block('experiment', 'experiment')
            block = f.blocks['experiment']
            n = len(block.data_arrays)

            log_array = block.create_data_array(
                f'experiment_{n}', 'experiment', dtype=np.float64, data=[])
            sec = f.create_section(f'experiment_{n}_metadata', 'metadata')
            log_array.metadata = sec

            valve_names, mfc_names = self._col_names
            sec['protocol'] = self._exp_protocol_text
            sec['valve_names'] = ','.join(valve_names)
            sec['mfc_names'] = ','.join(mfc_names)
            sec['protocol_key'] = key
            sec['app_config'] = config

        self._run_protocol(protocol, alarm, log_array, app.devices)

    def stop(self):
        if self._clock_event is not None:
            self._clock_event.cancel()
            self._clock_event = None

        if self._protocol_clock_event is not None:
            self._protocol_clock_event.cancel()
            self._protocol_clock_event = None

        self.playing = False

    @app_error
    def load_protocol(self, paths):
        """Called by the GUI when user browses for a file.
        """
        self.protocol = {}
        self.filename = ''

        if not paths:
            return
        fname = pathlib.Path(paths[0])

        self.last_directory = str(fname.parent)

        data = fname.read_text(encoding='utf-8-sig')
        self._exp_protocol_text = data

        reader = csv.reader(data.splitlines(False))
        header = next(reader)
        if len(header) < 2:
            raise ValueError(
                'The csv file must have at least a duration and key column')

        if header[0].lower() != 'duration':
            raise ValueError('First column must be named duration')
        if header[-1].lower() != 'key':
            raise ValueError('Last column must be named key')

        i = valve_s = 1
        while i < len(header) - 1 and header[i].lower().startswith('valve_'):
            i += 1
        mfc_s = valve_e = i
        while i < len(header) - 1 and header[i].lower().startswith('mfc_'):
            i += 1
        mfc_e = i
        if i != len(header) - 1:
            raise ValueError(
                f'Reached column "{header[i]}" that does not start with valve_ '
                f'or mfc_')

        valve_names = [header[k][6:] for k in range(valve_s, valve_e)]
        mfc_names = [header[k][4:] for k in range(mfc_s, mfc_e)]

        protocols = {}
        for row in reader:
            dur = float(row[0])
            key = row[-1]
            valves = [
                bool(int(row[k])) if row[k] else None
                for k in range(valve_s, valve_e)
            ]
            mfcs = [
                float(row[k]) if row[k] else None for k in range(mfc_s, mfc_e)]
            item = dur, valves, mfcs

            if key not in protocols:
                protocols[key] = []
            protocols[key].append(item)

        self.protocol = protocols
        self._col_names = valve_names, mfc_names
        self.filename = str(fname)

    def _update_clock(self, *args):
        ts = self._timer_ts
        if not ts:
            self.remaining_time = self.stage_remaining_time = '00:00:00.0'
            return

        elapsed = perf_counter() - ts
        self.remaining_time = pretty_time(
            max(self._total_time - elapsed, 0), pad=True)
        self.stage_remaining_time = pretty_time(
            max(self._total_stage_time - elapsed, 0), pad=True)

    def _make_sound(self):
        sound = SoundLoader.load(
            os.path.join(
                os.path.dirname(__file__), 'data', 'Electronic_Chime.wav'))
        sound.loop = True
        sound.play()
        sleep(4)
        sound.stop()
