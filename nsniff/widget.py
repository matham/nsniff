
from kivy_trio.to_trio import kivy_run_in_async, mark, KivyEventCancelled
from pymoa_remote.threading import ThreadExecutor
from base_kivy_app.app import app_error

from kivy.properties import ObjectProperty, StringProperty, BooleanProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget

from nsniff.device import StratuscentSensor, VirtualStratuscentSensor, \
    StratuscentBase

__all__ = ('DeviceDisplay', )


class DeviceDisplay(BoxLayout):

    _config_props_ = ('com_port', 'virtual')

    com_port: str = StringProperty('')

    device: StratuscentBase = ObjectProperty(None, allownone=True, rebind=True)

    virtual = BooleanProperty(False)

    notes = StringProperty('')

    done = False

    async def run_device(self):
        async with ThreadExecutor() as executor:
            async with executor.remote_instance(self.device, 'sensor'):
                async with self.device as device:
                    async with device.read_sensor_values() as aiter:
                        async for _ in aiter:
                            if self.done:
                                break
                            print(device.timestamp, device.sensors_data)

    @app_error
    @kivy_run_in_async
    def start(self):
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


class SensorGraph(Widget):

    pass
