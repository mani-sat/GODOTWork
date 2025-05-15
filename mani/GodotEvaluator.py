from multiprocessing import Pool
import numpy as np
import pandas as pd
from numba import njit

from tqdm import tqdm
from godot import cosmos, core
from godot.core import tempo
import godot.core.util as util

from .VisibilityModel import VisibilityModel
from .StateEvaluator import SEEnum, StateEvaluator
from .utils import EventGrid, get_len
from .HaloOrbit.HaloOrbit import HaloOrbit

util.suppressLogger()

class GodotHandler:
    def __init__(self, start_time, end_time, resolution, universe_file):
        self.event_grid = EventGrid(start_time, end_time, resolution)
        self.universe_file = universe_file
        self.uni_config = cosmos.util.load_yaml(universe_file)
        self.Halo = HaloOrbit(self.event_grid.t1)

    def get_event_grid(self):
        return self.event_grid.get_event_grid()

    def calculate_visibility(self, chunksize:int = 1000):
        print("Initializing calculate visibility")
        event_grid = self.get_event_grid()
        print("Initializing Halo Orbit")
        self.initialize_halo_orbit(event_grid, 10000)

        # RUN WORK
        print("Creating chunks")
        params = self.create_chunks(chunksize, event_grid)
        print("Evaluating chunks")
        eval = self._evaluate_chuncks_multiprocessed(params)
        print("Moving chunks to StateEvaluator")
        results_df = self._move_to_state_evaluator(eval, event_grid)
        return results_df
    
    def initialize_halo_orbit(self, event_grid, n_points):
        self.Halo.load_Halo_Data("./mani/HaloOrbit/GateWayOrbit_prop.csv")

        uni = self.fetch_universe()
        delta = len(event_grid) // n_points
        eval_points = [event_grid[idx*delta] for idx in range(n_points)]
        moon_data = [uni.frames.vector3('Earth', 'Moon', 'ICRF', ep) for ep in eval_points]
        moon_data=np.asarray(moon_data)

        self.Halo.translate_to_orbit_plane(moon_data)

    def _evaluate_chuncks_multiprocessed(self, params) -> list[dict]:
        with Pool() as pool:
            pool_results = pool.map(self._evaluate_timestamps, params)
            pool.close()
            pool.join()
        
        gw_dists_list = []
        st_dists_list = []
        states_list = []
        elevations_list = []
        for gw_dist, elev, state, st_dist in tqdm(pool_results):
            st_dists_list.append(st_dist)
            gw_dists_list.append(gw_dist)
            elevations_list.append(elev)
            states_list.append(state)


        gw_dists_all = np.concatenate(gw_dists_list)
        st_dists_all = np.concatenate(st_dists_list)
        states_all = np.concatenate(states_list)
        elevs_all = np.concatenate(elevations_list)
        df = pd.DataFrame({
            'gw_dist': gw_dists_all,
            'NN11_elev': elevs_all[:, 0],
            'CB11_elev': elevs_all[:, 1],
            'MG11_elev': elevs_all[:, 2],
            'AAU_elev' : elevs_all[:, 3],
            'NN11_dist': st_dists_all[:, 0],
            'CB11_dist': st_dists_all[:, 1],
            'MG11_dist': st_dists_all[:, 2],
            'AAU_dist': st_dists_all[:, 3],
            'state': states_all
        })
        return df
    
    def create_chunks(self, chksize, event_grid) -> list[tempo.Epoch]:
        params = []
        for i in range(0, len(event_grid), chksize):
            end = min(i + chksize, len(event_grid))
            params.append(event_grid[i:end])
        return params

        
    def fetch_universe(self):
        return cosmos.Universe(self.uni_config)
    
    def _evaluate_timestamps(self, args):
        t_list = args
        vismod = VisibilityModel()
        uni = self.fetch_universe()

        list_length = len(t_list)
        gw_dists = np.empty(list_length, dtype=np.uint32)
        states = np.empty(list_length, dtype=np.uint8)
        elevations = np.empty((list_length, 4), dtype=np.float16)
        st_dists = np.empty((list_length, 4), dtype=np.float32)

        stations = {
            'NN11' : SEEnum.CLEAR_MOON_NN,
            'CB11' : SEEnum.CLEAR_MOON_CB,
            'MG11' : SEEnum.CLEAR_MOON_MG,
            'AAU' : SEEnum.CLEAR_MOON_AAU
            }
        
        for index1, t in enumerate(t_list):
            #Setup a 8 bit holder for flags
            state:np.uint8= 0
            # Common vectors for all stations
            sun = uni.frames.vector3('Moon', 'Sun', 'ICRF', t)
            earth = uni.frames.vector3('Moon','Earth', 'ICRF', t)
            sc = uni.frames.vector3('Moon','SC', 'ICRF', t)

            gw_pos = self._get_mooncentric_GW_pos(earth, t)
            gw_dists[index1] = np.linalg.norm(gw_pos - sc)
            gw_los = vismod.los_from_gs_to_sc(gw_pos, sc)
            state = self.update_bit(state, SEEnum.LOS_GW, gw_los)

            # Calculate things that are not dependent on station
            slos = vismod.sun_light_on_spacecraft(sun, earth, sc)
            state = self.update_bit(state,SEEnum.SUN_ON_SPACECRAFT, slos)
            slom = vismod.sun_light_on_moon(sun, earth, sc)
            state = self.update_bit(state, SEEnum.SUN_ON_MOON, slom)

            # Run through each station, and get evaluations
            for index2, (station, enum) in enumerate(stations.items()):
                # Get vector that are dependent on station
                ground_station = uni.frames.vector3('Moon',station, 'ICRF', t)
                gs_sc = uni.frames.vector3(station,'SC',station,t)
                st_dists[index1, index2] = get_len(gs_sc)
                lfgts = vismod.los_from_gs_to_sc(sc, ground_station)
                state = self.update_bit(state, enum, lfgts)
                elev = vismod.get_elevation(gs_sc)
                elev_deg = elev
                elevations[index1, index2] = elev_deg
            #Update and append
            states[index1] = state
        return (gw_dists, elevations, states, st_dists)
    
    def _get_mooncentric_GW_pos(self, earth, t):
        moon = - earth
        gw_pos_earth_centred = self.Halo.get_HaloGW_pos(t, moon)
        gw_pos = gw_pos_earth_centred + earth
        return gw_pos
    
    @staticmethod
    def _move_to_state_evaluator(result_df: pd.DataFrame, event_grid) -> StateEvaluator:
        result_df.insert(0,"time",event_grid)
        result_df = StateEvaluator(result_df)
        return result_df
    
    @staticmethod
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

if __name__ == "__main__":
    import sys
    universe_file = './universe2.yml'
    import time
    t1 = time.perf_counter()
    ep1 = core.tempo.Epoch('2026-06-02T00:00:00 TDB')
    ep2 = core.tempo.Epoch('2026-06-02T02:00:00 TDB')
    godotHandler = GodotHandler(ep1, ep2, 1.0, universe_file)

    res = godotHandler.calculate_visibility(200)

    flags = [SEEnum.CLEAR_MOON_NN, SEEnum.SUN_ON_MOON]
    condition2 = (res.above_elev('CB11', 10.0) & res.has(flags))
    print(time.perf_counter() - t1)

    # t1 = tempo.Epoch('2026-04-02T01:00:00 TDB')
    # t2 = tempo.Epoch('2026-04-03T01:00:00 TDB')
    # godotHandler = GodotHandler(t1, t2, 1.0, './universe2.yml')
    # eg = godotHandler.get_event_grid()
    # godotHandler.initialize_halo_orbit(eg,1000)
    # results = godotHandler._evaluate_timestamps(eg)
    # res = godotHandler._move_to_StateEvaluator(results, eg)