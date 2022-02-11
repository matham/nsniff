from os.path import join, dirname
import trio
from base_kivy_app.app import BaseKivyApp, run_app_async as run_app_async_base
from base_kivy_app.graphics import HighightButtonBehavior
from typing import List, Set, Optional
from string import ascii_letters, digits
import nixio as nix

from kivy.lang import Builder
from kivy.factory import Factory
from kivy.properties import ObjectProperty, StringProperty, BooleanProperty, \
    NumericProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.behaviors.focus import FocusBehavior

from kivy_trio.context import kivy_trio_context_manager

import nsniff
from nsniff.widget import DeviceDisplay, ValveBoardWidget, MFCWidget, \
    ExperimentStages
from nsniff.model import SapinetModel

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

        if self.app.root.ids.stage_key.state == 'down' and \
                not self.app.stage.playing and text in self.app.stage.protocol:
            self.keyboard_chars.add(keycode[1])
            self.app.root.ids.stage_key_text.text = text
            self.app.root.ids.play_stage.trigger_action(0)
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
        'last_directory', 'pixel_height', 'n_valve_boards',
        'n_mfc', 'n_sensors', 'compression'
    )

    _config_children_ = {
        'devices': 'devices', 'valve_boards': 'valve_boards', 'mfcs': 'mfcs',
        'stage': 'stage'
    }

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

    _dev_container = None

    n_sensors: int = NumericProperty(1)

    devices: List[DeviceDisplay] = []

    _nix_file: Optional[nix.File] = None

    global_focus = BooleanProperty(False)

    pixel_height: int = NumericProperty(4)

    _valve_container = None

    n_valve_boards: int = NumericProperty(1)

    valve_boards: List[ValveBoardWidget] = []

    _mfc_container = None

    n_mfc: int = NumericProperty(1)

    mfcs: List[MFCWidget] = []

    compression = StringProperty('Auto')
    """Whether the h5 data file should internally compress the data that it
    writes.

    This is handled internally by the H5 library, with no external difference
    in how the file is loaded/saved/accessed, except that the file size may be
    smaller when compressed. Additionally, it may take a little more CPU when
    saving experiment data if compressions is enabled.

    Valid values are ``"ZIP"``, ``"None"``, or ``"Auto"``.
    """

    stage: ExperimentStages = ObjectProperty(None)

    model: SapinetModel = ObjectProperty(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.devices = []
        self.valve_boards = []
        self.mfcs = []

        self.stage = ExperimentStages(app=self)
        self.stage.fbind('filename', self.set_tittle)

        self.model = SapinetModel()

        self.fbind('filename', self.set_tittle)
        self.fbind(
            'n_sensors', self._update_num_io, 'devices', 'n_sensors',
            '_dev_container', DeviceDisplay)
        self.fbind(
            'n_valve_boards', self._update_num_io, 'valve_boards',
            'n_valve_boards', '_valve_container', ValveBoardWidget)
        self.fbind(
            'n_mfc', self._update_num_io, 'mfcs', 'n_mfc',
            '_mfc_container', MFCWidget)

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

        self.load_app_settings_from_file()
        self.apply_app_settings()

    def apply_config_property(self, name, value):
        setattr(self, name, value)
        if name in {'n_sensors', 'n_valve_boards', 'n_mfc'}:
            self.property(name).dispatch(self)

    def set_tittle(self, *largs):
        """Periodically called by the Kivy Clock to update the title.
        """
        from kivy.core.window import Window

        filename = ''
        if self.filename:
            filename = ' - {}'.format(self.filename)
        protocol = ''
        if self.stage.filename:
            protocol = ' - {}'.format(self.stage.filename)

        Window.set_title(
            f'NSniff v{nsniff.__version__}, CPL lab{filename}{protocol}')

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

        for dev in self.devices[:]:
            dev.stop()
        for dev in self.valve_boards[:]:
            dev.stop()
        for dev in self.mfcs[:]:
            dev.stop()

        self.stage.stop()

        self.close_file()

    async def async_run(self, *args, **kwargs):
        with kivy_trio_context_manager():
            await super().async_run(*args, **kwargs)

    def _update_num_io(
            self, widgets_name, n_items_name, widgets_container_name,
            widget_cls, *args):
        n_items = getattr(self, n_items_name)
        widgets = getattr(self, widgets_name)
        widgets_container = getattr(self, widgets_container_name)
        nix_file = self._nix_file

        if n_items < len(widgets):
            for dev in widgets[n_items:]:
                if widgets_name == 'devices':
                    dev.funbind('on_data_update', self.sensor_update)

                dev.stop()
                dev.clear_logging_file()

                widgets_container.remove_widget(dev)
                widgets.remove(dev)
        else:
            for _ in range(n_items - len(widgets)):
                dev = widget_cls()
                widgets_container.add_widget(dev)
                widgets.append(dev)

                if nix_file is not None:
                    dev.set_logging_file(nix_file)

                if widgets_name == 'devices':
                    dev.fbind('on_data_update', self.sensor_update)

    def save_file_callback(self, paths):
        """Called by the GUI when user browses for a file.
        """
        if self.filename:
            raise TypeError('Log file already open')

        if not paths:
            return
        fname = paths[0]
        if not fname.endswith('.h5'):
            fname += '.h5'

        self.last_directory = dirname(fname)

        if self.compression == 'ZIP':
            c = nix.Compression.DeflateNormal
        elif self.compression == 'None':
            c = nix.Compression.No
        else:
            c = nix.Compression.Auto
        self.filename = fname

        f = self._nix_file = nix.File.open(
            fname, nix.FileMode.Overwrite, compression=c)
        sec = f.create_section('app_config', 'configuration')
        sec['app'] = 'nsniff'
        sec['version'] = nsniff.__version__

        for dev in self.devices + self.valve_boards + self.mfcs:
            dev.set_logging_file(f)

        self.stage.set_logging_file(f)

    def close_file(self):
        if not self.filename:
            return

        for dev in self.devices + self.valve_boards + self.mfcs:
            dev.clear_logging_file()
        self.stage.clear_logging_file()

        self._nix_file.close()
        self._nix_file = None
        self.filename = ''

    def sensor_update(self, widget: DeviceDisplay):
        self.model.update_data(widget.unique_dev_id, widget.device.sensors_data)


def run_app():
    """The function that starts the main Ceed GUI and the entry point for
    the main script.
    """
    return trio.run(run_app_async_base, NSniffApp, 'trio')


if __name__ == '__main__':
    run_app()
