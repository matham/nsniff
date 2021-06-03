from os.path import join, dirname
import trio
from base_kivy_app.app import BaseKivyApp, run_app_async as run_app_async_base
from base_kivy_app.graphics import HighightButtonBehavior

from kivy.lang import Builder
from kivy.factory import Factory
from kivy.properties import ObjectProperty, StringProperty

from kivy_trio.context import kivy_trio_context_manager

import nsniff
from nsniff.widget import DeviceDisplay


__all__ = ('NSniffApp', 'run_app')


class NSniffApp(BaseKivyApp):
    """The app which runs the main Ceed GUI.
    """

    _config_props_ = (
        'last_directory',
    )

    _config_children_ = {'device': 'device'}

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

    filename: str = ''

    device: DeviceDisplay = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

        root = Factory.get('MainView')()
        return super().build(root)

    def on_start(self):
        self.set_tittle()
        HighightButtonBehavior.init_class()

        self.load_app_settings_from_file()
        self.apply_app_settings()

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
        if self.device is not None:
            self.device.stop()
            self.device = None

    async def async_run(self, *args, **kwargs):
        with kivy_trio_context_manager():
            await super().async_run(*args, **kwargs)


def run_app():
    """The function that starts the main Ceed GUI and the entry point for
    the main script.
    """
    return trio.run(run_app_async_base, NSniffApp, 'trio')


if __name__ == '__main__':
    run_app()
