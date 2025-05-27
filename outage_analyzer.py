import numpy as np

stations = ['AAU', 'NN11']
full_rate = [75211954, 404820636]

for i, station in enumerate(stations):
    outages = np.load(f'./output/outages_{station}_old.npy')
    outages = outages[outages > 0]
    print(len(outages))
    print(sum(outages > 0))
    print(f"station: {station} mean: {np.mean(outages)}")
    print(f"station: {station} std: {np.std(outages)}")
    print(f"Full rate: {full_rate[i]}, reduced rate: {full_rate[i]*(1-np.mean(outages))}")
    