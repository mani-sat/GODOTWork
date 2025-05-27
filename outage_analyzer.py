import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import pickle


font_path = '/home/dyb/.fonts/palatinolinotype_roman.ttf'
fm.fontManager.addfont(font_path)

mpl.rcParams['font.family'] = 'Palatino Linotype'
mpl.rcParams['font.sans-serif'] = ['Palatino Linotype']
mpl.rcParams['font.serif'] = ['Palatino Linotype']
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12

stations = ['AAU', 'NN11']
filename = ["outages_AAU.npy", "outages_NN11.npy"]
full_rate = [75211954, 404820636]

for i, station in enumerate(stations):
    with open(f'./output/{station}_data/outage_los.pickle', 'rb') as f:
        outlos = pickle.load(f)
    
    with open(f'./output/{station}_data/los.pickle', 'rb') as f:
        los = pickle.load(f)
    
    print(f"{station}: Outages : {(np.mean(los) - np.mean(outlos)) * 100:.2f}%")
    
    print()

    outages_with_zeros = np.load(f"./output/{filename[i]}")
    outages = outages_with_zeros[outages_with_zeros > 0]
    N = len(outages)
    mean = np.mean(outages)
    std = np.std(outages, ddof=1)
    print(f"station: {station}, N: {N}, mean: {np.mean(outages)},  std: {np.std(outages)}")
    print(f"Full rate: {full_rate[i]}, reduced rate: {full_rate[i]*(1-np.mean(outages))}")
    print(f"Full rate: {full_rate[i]}, reduced rate used: {full_rate[i]*(1-(np.mean(los) - np.mean(outlos)))}")

    from scipy import stats
    #Studnt, n=999, p<0.05, 2-tail
    #equivalent to Excel TINV(0.05,999)
    T =stats.t.ppf(0.95, N)
    print(T)

    neg = mean-T*std*np.sqrt(1+(1/N))
    pos = mean+T*std*np.sqrt(1+(1/N))
    print(neg)
    print(pos)
    print(f"Full rate: {full_rate[i]}, reduced rate: {full_rate[i]*(1-pos)}")

    plt.figure()
    plt.hist(outages, bins=25, color='steelblue', edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(outages), label='Mean outage probability', color = 'tab:orange')
    plt.axvline(neg, label='95% Prediction interval', color = 'tab:orange', linestyle='--')
    plt.axvline(pos, color = 'tab:orange', linestyle='--')
    plt.title(f"{station}, N = {N}")
    plt.xlabel("Outage Probability")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()