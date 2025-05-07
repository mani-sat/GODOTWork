import enum
import pandas as pd
import numpy as np
from functools import reduce
import operator

class SEEnum(enum.IntFlag):
    SUN_ON_SPACECRAFT = enum.auto()
    SUN_ON_MOON = enum.auto()
    CLEAR_MOON_NN = enum.auto()
    CLEAR_MOON_CB = enum.auto()
    CLEAR_MOON_MG = enum.auto()
    CLEAR_MOON_AAU = enum.auto()
    LOS_GW = enum.auto()

class SatState(enum.IntEnum):
    IDLE = enum.auto()
    LP_COMM = enum.auto()
    HP_COMM = enum.auto()
    SCIENCE = enum.auto()

class StateEvaluator:
    def __init__(self, df: pd.DataFrame):
        self.min_elevaion = 10.0
        self.df = df

    def set_internal_min_elevation(self, min_elevation:np.float16):
        """ 
        Set the minimum elevation used to evaluate states as los coloumns

        Parameters
        ----------
        min_elevation : np.float16
            The elevation at which LOS is determined
        """
        self.min_elevaion = min_elevation

    def get_length(self):
        """ 
        Get the length of the dataframe

        Returns
        -------
        int
            The length of the dataframe
        """
        return len(self.df)

    def elv(self, station: str) -> pd.Series:
        """
        Determine the elevation of the station.
        Equivalent to df[elv_station]

        Parameters
        -----------
        station : str
            The station name used during creation of state space
        
        Returns
        -------
        pd.Series
            A series of elevation values for all timestamps
        
        """
        return self.df[station]
    
    def above_elev(self, station:str, min_elevation:np.float16) -> pd.Series:
        """
        Determine whether elevation is above a minimum elevation.

        Parameters
        -----------
        station : str
            The station name used during creation of state space
        
        Returns
        -------
        pd.Series
            A series of boolean values for all timestamps
        """
        return self.elv(station) > min_elevation
    
    def has(self, flags: list[SEEnum]) -> pd.Series:
        """
        Checks if state has flag.

        Parameters
        -----------
        flags : list[SEEnum]
            The flags to check.
        
        Returns
        -------
        pd.Series
            A series of boolean values for all timestamps.
            Returns True is all flags are True
        """

        combined_flag = np.array(reduce(operator.or_, flags))
        # Vectorized check: each row should have all bits set
        return (self.df['state'] & combined_flag) == combined_flag
    
    def add_los_coloumns(self):
        """
        Add coloums containing the LOS states of the groundstations.
        """
        los_nn = self.above_elev('NN11_elev', self.min_elevaion) & self.has([SEEnum.CLEAR_MOON_NN])
        if not ('los_nn' in self.df.keys()):
            self.df.insert(len(self.df.keys()),'los_nn',los_nn)
        los_cb = self.above_elev('CB11_elev', self.min_elevaion) & self.has([SEEnum.CLEAR_MOON_CB])
        if not ('los_cb' in self.df.keys()):
            self.df.insert(len(self.df.keys()),'los_cb',los_cb)
        los_mg = self.above_elev('MG11_elev', self.min_elevaion) & self.has([SEEnum.CLEAR_MOON_MG])
        if not ('los_mg' in self.df.keys()):
            self.df.insert(len(self.df.keys()),'los_mg',los_mg)
        los_aau = self.above_elev('AAU_elev', self.min_elevaion) & self.has([SEEnum.CLEAR_MOON_AAU])
        if not ('los_aau' in self.df.keys()):
            self.df.insert(len(self.df.keys()),'los_aau',los_aau)
         
    
    def has_not(self, flags: list[SEEnum]) -> pd.Series:
        """
        Checks if state does not have flag.

        Parameters
        -----------
        flags : list[SEEnum]
            The flags to check.
        
        Returns
        -------
        pd.Series
            A series of boolean values for all timestamps.
            Returns True is all flags are False
        """

        combined_flag = reduce(operator.or_, flags)
        # Vectorized check: each row should have all bits set
        return (self.df['state'] & combined_flag) != combined_flag
    
    def __getitem__(self, key:int):
        """
        Fetch item from dataframe. Equivalent to df[key]
        
        Parameters
        ----------
        key : int
            The index to look for
        
        Returns
        -------
            The contents of the index
        
        """
        return self.df[key]
    
    def keys(self):
        """
        Get the keys of the pandas DataFrame

        Returns:
        Index
        
        """
        return self.df.keys()
    
    def get_state(self):
        """
        Fetches the various flags, and returns the equivalent states

        
        """
        los_nn = self.above_elev('NN11_elev', self.min_elevaion) & self.has([SEEnum.CLEAR_MOON_NN])
        los_cb = self.above_elev('CB11_elev', self.min_elevaion) & self.has([SEEnum.CLEAR_MOON_CB])
        los_mg = self.above_elev('MG11_elev', self.min_elevaion) & self.has([SEEnum.CLEAR_MOON_MG])
        los_aau = self.above_elev('AAU_elev', self.min_elevaion) & self.has([SEEnum.CLEAR_MOON_AAU])
        los = los_nn | los_cb | los_mg | los_aau
        som = self.has([SEEnum.SUN_ON_MOON])
        sos = self.has([SEEnum.SUN_ON_SPACECRAFT])

        # Start with all IDLE
        s = pd.Series(SatState.IDLE, index=self.df.index, dtype="int")

        # Set PURE_SCIENCE where som is True
        s[som] = SatState.SCIENCE

        # Set HP_COMM where los and sos are True, but not som
        s[los & sos & ~som] = SatState.HP_COMM

        # Set LP_COMM where los is True, sos is False, and not som
        s[los & ~sos & ~som] = SatState.LP_COMM

        # Cast to SatState enum type
        return s.map(SatState)