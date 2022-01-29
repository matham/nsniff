import numpy as np
import trio
from typing import List, Dict, Optional, Tuple
from matplotlib import cm
from kivy_trio.to_trio import kivy_run_in_async, mark, KivyEventCancelled
from pymoa_remote.threading import ThreadExecutor
from pymoa_remote.socket.websocket_client import WebSocketExecutor
from base_kivy_app.app import app_error
from kivy_garden.graph import Graph, ContourPlot, LinePlot

from kivy.metrics import dp
from kivy.properties import ObjectProperty, StringProperty, BooleanProperty, \
    NumericProperty, ListProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.factory import Factory
from kivy.uix.widget import Widget
from kivy_garden.graph import Graph

from nsniff.device import StratuscentSensor, VirtualStratuscentSensor, \
    StratuscentBase

__all__ = ('DeviceDisplay', )


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


class DeviceDisplay(BoxLayout):

    __events__ = ('on_data_update', )

    _config_props_ = (
        'com_port', 'virtual', 'log_z', 'auto_range', 'global_range',
        'range_chan', 'n_channels', 'remote_server', 'remote_port')

    com_port: str = StringProperty('')

    remote_server: str = StringProperty('')

    remote_port: int = NumericProperty(0)

    device: Optional[StratuscentBase] = ObjectProperty(
        None, allownone=True, rebind=True)

    virtual = BooleanProperty(False)

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

    def on_data_update(self, instance):
        pass

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
        self.dispatch('on_data_update', self)

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

    async def _run_device(self, executor):
        async with executor.remote_instance(self.device, self.com_port):
            async with self.device as device:
                async with device.read_sensor_values() as aiter:
                    async for _ in aiter:
                        if self.done:
                            break
                        self.process_data(device)

    async def run_device(self):
        if self.remote_server:
            async with trio.open_nursery() as nursery:
                async with WebSocketExecutor(
                        nursery=nursery, server=self.remote_server,
                        port=self.remote_port) as executor:
                    await self._run_device(executor)
        else:
            async with ThreadExecutor() as executor:
                await self._run_device(executor)

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

    @staticmethod
    def get_data_header():
        return StratuscentBase.get_data_header()

    def add_event(self, t, name):
        p = LinePlot(color=(0, 0, 0), line_width=dp(3))
        p.points = [(t, .1), (t, 1)]
        self.graph_2d.add_plot(p)
        self._event_plots[0].append(p)

        p = LinePlot(color=(0, 0, 0), line_width=dp(3))
        p.points = [(t, 0), (t, self.graph_3d.ymax)]
        self.graph_3d.add_plot(p)
        self._event_plots[1].append(p)
