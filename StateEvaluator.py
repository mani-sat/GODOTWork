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
    LOS_GW = enum.auto()

class StateEvaluator:
    def __init__(self, df: pd.DataFrame):
        self.df = df

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
        return self.df[f'elv_{station}']
    
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

        combined_flag = reduce(operator.or_, flags)
        # Vectorized check: each row should have all bits set
        return (self.df['state'] & combined_flag) == combined_flag
    
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