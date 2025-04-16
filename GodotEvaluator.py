import numpy as np
from multiprocessing import Pool
from VisibilityModel import VisibilityModel
from StateEvaluator import SEEnum, StateEvaluator
from utils import EventGrid
import pandas as pd
from numba import njit

import godot.core.util as util
util.suppressLogger()
from tqdm import tqdm
from godot import cosmos, core

from godot.core import tempo

class GodotHandler():
    def __init__(self,t1, t2, res, universe_file):
        self.event_grid = EventGrid(t1, t2, res)
        self.universe_file = universe_file

    def get_event_grid(self):
        return self.event_grid.get_event_grid()

    def calculate_visibility(self, chunksize:int = 1000):
        event_grid = self.get_event_grid()
        # RUN WORK
        params = self.create_chunks(self.universe_file, chunksize, event_grid)
        eval = self._evaluate_chuncks_multiprocessed(params)
        results_df = self._move_to_StateEvaluator(eval, event_grid)
        return results_df

    def _evaluate_chuncks_multiprocessed(self, params) -> list[dict]:
        with Pool() as pool:
            results = pool.map(self._evaluate_timestamps, params)
            pool.close()
            pool.join()
        
        eval = []
        for val in tqdm(results):
            eval.extend(val)
        return eval
    
    @staticmethod
    def create_chunks(universe_file, chksize, event_grid) -> list[tempo.Epoch]:
        params = []
        for i in range(0, len(event_grid), chksize):
            end = min(i + chksize, len(event_grid))
            params.append((universe_file, event_grid[i:end]))
        return params
    
    @staticmethod
    def _move_to_StateEvaluator(eval: list[dict], event_grid) -> StateEvaluator:
        result_df = pd.DataFrame(eval)
        result_df.insert(0,"time",event_grid)
        result_df = StateEvaluator(result_df)
        return result_df
    
    @staticmethod
    def _evaluate_timestamps(args):
        uni_file, t_list = args
        uni = fetch_universe(uni_file)
        vismod = VisibilityModel()

        stations = {
            'NN11' : SEEnum.CLEAR_MOON_NN,
            'CB11' : SEEnum.CLEAR_MOON_CB,
            'MG11' : SEEnum.CLEAR_MOON_MG
            }
        
        my_eval = []

        for t in t_list:
            #Setup a 8 bit holder for flags
            state:np.uint8= 0
            eval_entry = {}

            # Common vectors for all stations
            sun = uni.frames.vector3('Moon','Sun','ICRF', t)
            earth = uni.frames.vector3('Moon','Earth', 'ICRF', t)
            sc = uni.frames.vector3('Moon','SC', 'ICRF', t)

            # Calculate things that are not dependent on station
            slos = vismod.sun_light_on_spacecraft(sun, earth, sc)
            state = update_bit(state,SEEnum.SUN_ON_SPACECRAFT, slos)
            slom = vismod.sun_light_on_moon(sun, earth, sc)
            state = update_bit(state, SEEnum.SUN_ON_MOON, slom)

            # Run through each station, and get evaluations
            for station in stations.keys():
                # Get vector that are dependent on station
                GS = uni.frames.vector3('Moon',station, 'ICRF', t)
                gs_sc = uni.frames.vector3(station,'SC',station,t)
                lfgts = vismod.los_from_gs_to_sc(sc, GS)
                state = update_bit(state, stations[station], lfgts)
                elev = vismod.calcElevation(gs_sc)
                elev_deg = np.degrees(elev,dtype=np.float16)
                eval_entry[f'elv_{station}'] =  elev_deg
            #Update and append
            eval_entry[f'state'] = state
            my_eval.append(eval_entry)
        return my_eval
    
    def get_view_times_span(self, conditions) -> np.ndarray:
        times = self.get_event_grid()
        view_time_span = []
        visible = False
        start_time:str

        length = len(times)
        for i in range(length):
            if not visible and conditions[i]:
                t = times[i]
                start_time = t
                visible = True
            if visible and not conditions[i]:
                t = times[i]
                visible = False
                view_time_span.append((start_time, t))
        view_time_span = np.array(view_time_span)

        return view_time_span

    def get_view_time_lengths(self, view_time_span):
        arr = []
        for interval in view_time_span:
            start = interval[0]
            end = interval[1]
            arr.append(end-start)

        return np.array(arr)

@njit
def update_bit(state, setting: SEEnum, status:bool):
    """
    Updates the state of the bit matching setting

    Parameters
    ----------
    state : np.uint8
        The uint8 holding the flags
    setting : SEEnum
        The flag that is to be updated
    status : bool
        The value that the flag should be update to

    Returns
    -------
    np.uint8
        The uint8 holding the flags
    """
    if status:
        return state | setting
    else:
        return state & ~setting

def fetch_universe(universe_file):
    if not universe_file.endswith('.yml'):
        raise 'Universe file must be YAML!'
    # create the universe
    uni_config = cosmos.util.load_yaml(universe_file)
    return  cosmos.Universe(uni_config)

if __name__ == "__main__":
    universe_file = './universe2.yml'
    ep1 = core.tempo.Epoch('2026-06-02T00:00:00 TDB')
    ep2 = core.tempo.Epoch('2026-06-02T01:00:00 TDB')
    godotHandler = GodotHandler(ep1, ep2, 30.0, universe_file)

    res = godotHandler.calculate_visibility(200)

    flags = [SEEnum.CLEAR_MOON_NN, SEEnum.SUN_ON_MOON]
    condition2 = (res.above_elev('CB11', 10.0) & res.has(flags))
    [print(cond) for cond in condition2]