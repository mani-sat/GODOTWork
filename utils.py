from datetime import datetime
from pytz import UTC
from godot.core import tempo

def convert_to_datetime(timestamp: tempo.Epoch):
    dts = datetime.strptime(timestamp.calStr('TT'), '%Y-%m-%dT%H:%M:%S.%f TT').astimezone(UTC)
    return dts

def get_date_string(timestamp: tempo.Epoch):
    return convert_to_datetime(timestamp).strftime('%Y-%m-%d')


class EventGrid():
    """
    A simple structure to hold and return the EpochRange grid.

    *This class has been created to keep Event Grids in scopes*
    *and not stored in memory*
    """

    def __init__(self, t1: tempo.Epoch, t2: tempo.Epoch, res: float):
        self.t1 = t1
        self.t2 = t2
        self.resolution = res
        
    def get_event_grid(self):
        """
        Create and fetch the event grid

        Returns
        -------
        list[godot.core.tempo.Epoch]
            An grid of timestamps
        """
        return tempo.EpochRange(self.t1, self.t2).createGrid(self.resolution)
    
    