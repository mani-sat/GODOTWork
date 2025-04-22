from HaloOrbit import HaloOrbit
from utils_luke import to_standard_units, formatter, rodrigues
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from godot.core import tempo, astro, events
from godot.core.autodif import bridge as br
from godot import cosmos
from godot.model import eventgen, common, prop
import godot.core.util as util
from sklearn.decomposition import PCA
util.suppressLogger()

class HaloOrbitTest(HaloOrbit):
    def __init__(self, epoch):
        HaloOrbit.__init__(self, epoch)

    def test_min_max_mean_diff(self, moonData, grid):
        HaloOrbit.load_Halo_Data(self, HaloDataFile="HaloOrbit/GateWayOrbit_prop.csv")
        self.x_moon
        distances=np.zeros(len(self.HaloData))
        print("min\t", "max\t", "mean")
        for i, point in enumerate(self.HaloData):
              distances[i]=np.linalg.norm(point[1:]-self.x_moon)
        print(np.min(distances), np.max(distances), np.mean(distances))

        HaloOrbit.translate_to_orbit_plane(self, moonData)

        for i, point in enumerate(self.new_HaloData):
              distances[i]=np.linalg.norm(point[1:]-self.init_moon_point)
        print(np.min(distances), np.max(distances), np.mean(distances))

        rand_index=np.random.randint(0, len(grid), 10)
        emph=np.zeros((len(grid), 3))
        distances=np.zeros(len(grid))
        for index in rand_index:
            point_moon=moonData[index]
            for i, ep in enumerate(grid):
                pos=HaloOrbit.get_HaloGW_pos(self, ep, point_moon)
                emph[i]=pos
            for x, point in enumerate(emph):
                 distances[x]=np.linalg.norm(point-self.init_moon_point)
            print(np.min(distances), np.max(distances), np.mean(distances))


if __name__=="__main__":
    import time
    t1 = time.perf_counter()
    # os.chdir("../")
    uni_config=cosmos.util.load_yaml("./universe2.yml")
    uni = cosmos.Universe(uni_config)

    ep1 = tempo.Epoch('2026-01-01T00:00:00 TT')
    ep2 = tempo.Epoch('2026-02-01T00:00:00 TT')
    ran = tempo.EpochRange( ep1, ep2 )
    timestep = 1.0
    grid = ran.createGrid(timestep) # 60 seconds stepsize
    frames = uni.frames
    icrf = frames.axesId('ICRF')
    moonPoint=frames.pointId('Moon')
    earthPoint=frames.pointId('Earth')

    moonData=np.asarray([frames.vector3(earthPoint, moonPoint, icrf, ep) for ep in grid])
    
    Halo_test=HaloOrbitTest(ep1)
    Halo_test.test_min_max_mean_diff(moonData, grid)
    