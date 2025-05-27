import godot

import numpy as np
from godot.core import util
import matplotlib.pyplot as plt
import pickle


# optionally avoid verbose logging messages
import godot.core.util as util
util.suppressLogger()

import os

os.makedirs('./output/',exist_ok = True)

# create the universe
uni_config = godot.cosmos.util.load_yaml('universe2.yml')
uni = godot.cosmos.Universe(uni_config)

import mani_rain
from mani.StateEvaluator import SEEnum
from tqdm import tqdm

class data_generator:
    def __init__(self, year, station, filepath='./output/year_sim', bandwidth=50e6):
        lut = {'AAU':SEEnum.CLEAR_MOON_AAU, 'NN11':SEEnum.CLEAR_MOON_NN}
        self._load_data(filepath, year)

        self.station_name = station

        self.condition = self.res.above_elev(station+'_elev', 10.0) & self.res.has([lut[station]])
        self.bandwidth = bandwidth

        self.set_link()
    
    def _load_data(self, filepath, year):
        filename = filepath+'/one_year_' + str(year) + '.pickle'
        with open(filename, 'rb') as f:
            self.res = pickle.load(f)
        self.res.add_los_coloumns()

    def set_science_rate_day(self, sc_rate):
        sc = sc_rate / (60*60*24 * np.mean(self.res.has([SEEnum.SUN_ON_MOON])))
        self.science_rate = sc

    def set_target_data_rate(self, target_data_rate = 40e6):
        self.target_data_rate = target_data_rate
        self.full_data_rate = self.link.dvb_s2_fixed_rate(100, target_data_rate)

    def set_gateway_rate(self, gateway_rate = 16e6):
        self.gw_rate = gateway_rate
    
    def set_link(self):
        model_dict = {'AAU':mani_rain.rain.aau_ma_model,'NN11':mani_rain.rain.nn_ma_model}
        station_dict = {'AAU':mani_rain.aalborg, 'NN11':mani_rain.new_norcia}

        mrm = mani_rain.markov_rain(station_dict[self.station_name], model_dict[self.station_name])
        self.link = mani_rain.link_budget_markov(station_dict[self.station_name], mrm, self.bandwidth, link_margin = 0)

    def get_rates(self):
        return self.full_data_rate, self.science_rate, self.gw_rate
    
    def simulate(self):
        res = self.res
        stat = self.station_name

        st_snrs = np.empty(res.get_length())
        st_rates = np.empty(res.get_length())

        for i in tqdm(range(res.get_length())):
            dist = res.df.iloc[i][stat+'_dist'] * 1e3
            elev = res.df.iloc[i][stat+'_elev']
            if self.condition[i]:
                st_snrs[i] = self.link.snr_at_t(dist, elev)
                st_rates[i] = self.link.dvb_s2_fixed_rate(st_snrs[i], self.full_data_rate)
                #st_rates[i] = self.link.dvb_s2_cap(st_snrs[i])
            else:
                st_snrs[i] = np.nan
                st_rates[i] = 0

        self.gw_los = res.has([SEEnum.LOS_GW])
        self.comm_los = self.condition
        self.sun_on_moon = res.has([SEEnum.SUN_ON_MOON])
        self.st_rates = st_rates
        self.st_snrs = st_snrs

        return st_rates, st_snrs

    def get_outage_probability(self):
        #Following: Outage counter
        rate = self.link.dvb.modcod_at_rate(np.max(st_rates))
        esno = rate.esno
        los = self.condition
        outage_counter = np.sum((np.isnan(st_snrs) == False) & los & (st_snrs < esno))
        outage_prob = (outage_counter/len(los))
        print(f"Num samples: {len(self.condition)}, outages: {outage_counter}, outage prob: {outage_prob}")
        print(outage_prob)
        return outage_prob
    
    def plot_time(self):
        fig = plt.figure(figsize=(15, 4))
        ax0, ax1 = fig.subplots(1,2, sharex=True)
        ax0.plot(self.st_snrs)
        ax1.step(range(len(self.st_rates)), self.st_rates / 1e6, label = 'gs')
        ax1.step(range(len(self.gw_los)), self.gw_los * self.gw_rate / 1e6, label = 'gw')
        mean_rate = np.mean(self.st_rates[self.condition == True])
        ax1.axhline(mean_rate/1e6, color='tab:green', label='mean rate')
        ax1.set_ylabel('Rate Mbit')
        plt.show()

station = 'NN11'
science_rate = 25 * 1e9
year = 2027
repeats = 10
repeats_outer = 20
outprops = np.zeros(repeats*repeats_outer)
for ro in range(repeats_outer):
    for r in range(repeats):
        print(f"year {year} repeat {r+ro*repeats}")
        target_data_rates = {'AAU':60 * 1e6, 'NN11':404*1e6}
        bandwidths = {'AAU':100e6, 'NN11':100e6}
        dg = data_generator(year=year, station=station, bandwidth=bandwidths[station])
        dg.set_science_rate_day(science_rate)
        dg.set_target_data_rate(target_data_rates[station])
        dg.set_gateway_rate()
        st_rates, st_snrs = dg.simulate()
        outprops[r+ro*repeats] = dg.get_outage_probability()
    np.save(f'./output/outages_{station}', outprops)