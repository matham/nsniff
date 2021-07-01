from os.path import join, dirname, exists, 
import time
import trio
from base_kivy_app.app import BaseKivyApp, run_app_async as run_app_async_base
from base_kivy_app.graphics import HighightButtonBehavior
from typing import List, IO, Dict, Set, Tuple
from tree_config import apply_config
from string import ascii_letters, digits
from threading import Thread
from ffpyplayer.player import MediaPlayer

from kivy.lang import Builder
from kivy.factory import Factory
from kivy.properties import ObjectProperty, StringProperty, BooleanProperty, \
    ListProperty, NumericProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.behaviors.focus import FocusBehavior
from kivy.clock import Clock

from kivy_trio.context import kivy_trio_context_manager

import nsniff
from nsniff.widget import DeviceDisplay
from nsniff.device import StratuscentBase

__all__ = ('NSniffApp', 'run_app')


class MainView(FocusBehavior, BoxLayout):
    """The root widget displayed in the GUI.
    """

    app: 'NSniffApp' = None

    keyboard_chars: Set[str] = set()

    valid_chars = set(ascii_letters + digits)

    def __init__(self, app, **kwargs):
        super(MainView, self).__init__(**kwargs)
        self.ctrl_chars = set()
        self.app = app

    def on_focus(self, *args):
        self.app.global_focus = self.focus

    def keyboard_on_key_down(self, window, keycode, text, modifiers):
        if super(MainView, self).keyboard_on_key_down(
                window, keycode, text, modifiers):
            return True

        if keycode[1] not in self.valid_chars:
            return False

        if self.app.filename:
            self.keyboard_chars.add(keycode[1])
            self.app.log_event(text, display=True)
            return True
        return False

    def keyboard_on_key_up(self, window, keycode):
        if super(MainView, self).keyboard_on_key_up(window, keycode):
            return True

        if keycode[1] in self.keyboard_chars:
            self.keyboard_chars.remove(keycode[1])
            return True
        return False


class NSniffApp(BaseKivyApp):
    """The app which runs the main Ceed GUI.
    """

    _config_props_ = (
        'last_directory', 'event_times_countdown',
        'event_times_countdown_default'
    )

    _config_children_ = {'devices': 'devices'}

    last_directory = StringProperty('~')
    """The last directory opened in the GUI.
    """

    kv_loaded = False
    """For tests, we don't want to load kv multiple times so we only load kv if
    it wasn't loaded before.
    """

    yesno_prompt = ObjectProperty(None, allownone=True)
    '''Stores a instance of :class:`YesNoPrompt` that is automatically created
    by this app class. That class is described in ``base_kivy_app/graphics.kv``
    and shows a prompt with yes/no options and callback.
    '''

    filename: str = StringProperty('')

    devices: List[DeviceDisplay] = []

    _log_file: IO = None

    _devices_data = []

    _devices_n_cols: List[int] = []

    global_focus = BooleanProperty(False)

    _timer_ts = 0

    _last_countdown = 0

    remaining_time = StringProperty('00:00.0')

    event_times_countdown_default = NumericProperty(180)

    event_times_countdown: List[Tuple[str, float]] = ListProperty(
        [('', 300)])

    _event_times_countdown_dict: Dict[str, float] = {}

    _sound_thread = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.devices = []
        self._event_times_countdown_dict = {}
        self.fbind('filename', self.set_tittle)
        self.fbind('event_times_countdown', self._parse_event_times)

    def load_app_kv(self):
        """Loads the app's kv files, if not yet loaded.
        """
        if NSniffApp.kv_loaded:
            return
        NSniffApp.kv_loaded = True

        base = dirname(__file__)
        Builder.load_file(join(base, 'nsniff_style.kv'))

    def build(self):
        self.load_app_kv()
        self.yesno_prompt = Factory.FlatYesNoPrompt()

        root = MainView(app=self)
        return super().build(root)

    def on_start(self):
        self.set_tittle()
        HighightButtonBehavior.init_class()

        self.add_device()

        self.load_app_settings_from_file()
        self.apply_app_settings()
        Clock.schedule_interval(self._update_clock, .25)
        self._parse_event_times()

    def apply_config_child(self, name, prop, obj, config):
        if prop == 'devices':
            if len(config) >= len(self.devices):
                for _ in range(len(config) - len(self.devices)):
                    self.add_device()
            else:
                for _ in range(len(self.devices) - len(config)):
                    self.remove_device(self.devices[-1])

            for dev, conf in zip(self.devices, config):
                apply_config(dev, conf)
        else:
            apply_config(obj, config)

    def set_tittle(self, *largs):
        """Periodically called by the Kivy Clock to update the title.
        """
        from kivy.core.window import Window
        filename = ''
        if self.filename:
            filename = ' - {}'.format(self.filename)

        Window.set_title(f'NSniff v{nsniff.__version__}, CPL lab{filename}')

    def check_close(self):
        if False:
            self._close_message = 'Cannot close during an experiment.'
            return False
        return True

    def handle_exception(self, *largs, **kwargs):
        val = super().handle_exception(*largs, **kwargs)
        return val

    def clean_up(self):
        super().clean_up()
        HighightButtonBehavior.uninit_class()
        self.dump_app_settings_to_file()
        for dev in self.devices:
            dev.stop()

        self.close_file()

    async def async_run(self, *args, **kwargs):
        with kivy_trio_context_manager():
            await super().async_run(*args, **kwargs)

    def add_device(self) -> None:
        if self.filename:
            raise TypeError('Cannot add device while saving data')

        dev = DeviceDisplay()
        self.root.ids.dev_container.add_widget(dev)
        self.devices.append(dev)

    def remove_device(self, device: DeviceDisplay) -> None:
        if self.filename:
            raise TypeError('Cannot remove device while saving data')

        device.stop()
        self.root.ids.dev_container.remove_widget(device)
        self.devices.remove(device)

    def log_event(self, name, display=False):
        t = StratuscentBase.get_time()
        if display:
            self.root.ids.event_name.text = name

        self.log_device([t, name], len(self.devices))

        for dev in self.devices:
            dev.add_event(t, name)

        self._timer_ts = t
        self._last_countdown = self._event_times_countdown_dict.get(
            name, self.event_times_countdown_default)
        self._update_clock()

    def save_file_callback(self, paths):
        """Called by the GUI when user browses for a file.
        """
        if self.filename:
            raise TypeError('Log file already open')

        if not paths:
            return
        fname = paths[0]
        if not fname.endswith('.csv'):
            fname += '.csv'

        self.last_directory = dirname(fname)

        self.filename = fname
        self._log_file = open(fname, 'w')
        self._devices_n_cols = n_cols = []

        header = []
        for dev in self.devices:
            item = dev.get_data_header()
            header.extend(item)
            n_cols.append(len(item))

            dev.fbind(
                'on_data_update', self._get_dev_data, index=len(n_cols) - 1)

        header.extend(['Event time', 'event'])
        n_cols.append(2)
        self.write_line(header)
        self._devices_data = [None, ] * len(n_cols)

    def _get_dev_data(self, dev, *args, index=0):
        self.log_device(dev.device.get_data_row(), index)

    def log_device(self, data, index):
        if self._devices_data[index] is None:
            self._devices_data[index] = data
            return

        line = []
        for count, item in zip(self._devices_n_cols, self._devices_data):
            if item is not None:
                line.extend(item)
            else:
                line.extend(['', ] * count)
        self.write_line(line)

        for i in range(len(self._devices_data)):
            self._devices_data[i] = None

        self._devices_data[index] = data

    def write_line(self, values):
        self._log_file.write(
            ','.join(
                [f'"{val}"' if isinstance(val, str) and val else str(val)
                 for val in values]
            ))
        self._log_file.write('\n')
        self._log_file.flush()

    def close_file(self):
        if not self.filename:
            return

        if any(data is not None for data in self._devices_data):
            line = []
            for count, item in zip(self._devices_n_cols, self._devices_data):
                if item is not None:
                    line.extend(item)
                else:
                    line.extend(['', ] * count)
            self.write_line(line)

        self._log_file.close()
        self._log_file = None
        self.filename = ''

        for i, dev in enumerate(self.devices):
            dev.funbind('on_data_update', self._get_dev_data, index=i)

        self._timer_ts = 0
        self._update_clock()

    def _update_clock(self, *args):
        ts = self._timer_ts
        if not ts:
            self.remaining_time = '00:00.0'
            return

        was_zero = self.remaining_time == '00:00.0'
        elapsed = StratuscentBase.get_time() - ts
        remaining = max(self._last_countdown - elapsed, 0)
        ms = round(remaining * 10) % 10
        sec = int(remaining) % 60
        minute = int(remaining / 60)
        self.remaining_time = f'{minute:0>2}:{sec:0>2}.{ms}'

        if not remaining and was_zero:
            thread = self._sound_thread = Thread(target=self._make_sound)
            thread.start()

    def _make_sound(self):
        player = MediaPlayer('video=Logitech HD Webcam C525:audio=Microphone (HD Webcam C525)',
                     ff_opts=ff_opts, lib_opts=lib_opts)

    def _parse_event_times(self, *args):
        items = self._event_times_countdown_dict = {}
        for key, value in self.event_times_countdown:
            for char in key.split(','):
                char = char.strip()
                if not char:
                    continue

                items[char] = value


def run_app():
    """The function that starts the main Ceed GUI and the entry point for
    the main script.
    """
    return trio.run(run_app_async_base, NSniffApp, 'trio')


if __name__ == '__main__':
    run_app()
