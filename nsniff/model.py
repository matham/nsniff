from typing import List

from kivy.event import EventDispatcher
from kivy.properties import StringProperty


class SapinetModel(EventDispatcher):

    class_pred: str = StringProperty('unknown')

    def update_data(self, sensor, data: List[float]):
        pass
