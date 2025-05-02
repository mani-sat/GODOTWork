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

class HaloOrbit:
    def __init__(self, epoch0):
        self.x_moon=np.array([to_standard_units(1, "LU"), 0, 0])
        self.epoch0=epoch0

    def load_Halo_Data(self, HaloDataFile):
        """
        Load in data points from HaloDataFile

        Parameters
        ----------
        HaloDataFile: csv file
            file containing the data points of the orbit around a static moon

        Returns
        -------
        HaloData: np.ndarray
            a 4 dimensional list containing time|x_value|y_value|z_value
        HaloOrbitTime: float
            orbit duration
        """
        f=open(HaloDataFile,"r")
        lines=f.readlines()
        f.close()
        print(lines[0])
        lines = lines[1:]
        lines = [[float(element) for element in line.strip().split(",")] for line in lines]
        self.HaloData = np.array([[to_standard_units(line[0], "TU"), 
                                   to_standard_units(line[1], "LU"), 
                                   to_standard_units(line[2], "LU"), 
                                   to_standard_units(line[3], "LU")] for line in lines])
        self.HaloOrbitTime = self.HaloData[-1][0]

    def find_rotation_axis(self, moonData):
        """
        find mean rotation axis for lunar orbit
        """
        pca=PCA(n_components=3)
        pca.fit(moonData)
        rot_axis=pca.components_[2]
        rot_axis /= np.linalg.norm(rot_axis)
        return rot_axis

    def get_initial_point_on_plane(self, moonData):
        """
        find nearest point on the lunar orbit and project it onto the lunar plane
        """
        moon_index=np.argmin([np.linalg.norm(moonPos-self.x_moon) for moonPos in moonData])
        init_moonPoint=moonData[moon_index]
        #project to plane
        Proj=np.eye(3)-np.outer(self.rot_axis,self.rot_axis)
        init_moonPoint=Proj@init_moonPoint
        return init_moonPoint

    def rotation_matrix(self, rot_axis, angle):
        """
        Create rotation axis
        """
        rot_axis/=np.linalg.norm(rot_axis)
        R=np.array([[rot_axis[0]**2*(1-np.cos(angle))+np.cos(angle),                        rot_axis[0]*rot_axis[1]*(1-np.cos(angle))-rot_axis[2]*np.sin(angle),    rot_axis[0]*rot_axis[2]*(1-np.cos(angle))+rot_axis[1]*np.sin(angle)],
                    [rot_axis[0]*rot_axis[1]*(1-np.cos(angle))+rot_axis[2]*np.sin(angle),   rot_axis[1]**2*(1-np.cos(angle))+np.cos(angle),                         rot_axis[1]*rot_axis[2]*(1-np.cos(angle))-rot_axis[0]*np.sin(angle)],
                    [rot_axis[0]*rot_axis[2]*(1-np.cos(angle))-rot_axis[1]*np.sin(angle),   rot_axis[1]*rot_axis[2]*(1-np.cos(angle))+rot_axis[0]*np.sin(angle),    rot_axis[2]**2*(1-np.cos(angle))+np.cos(angle)]])
        return R
    def rotate_plane(self, plane1, plane2):
        # xy_plane=np.array([0,0,1])
        #rot_axis
        plane_axis=np.cross(plane1, plane2)
        plane_angle=np.dot(plane1, plane2)/(np.linalg.norm(plane1)*np.linalg.norm(plane2))
        plane_angle=np.arccos(plane_angle)

        new_HaloData=np.zeros((len(self.HaloData), 3))
        Rot_mat=self.rotation_matrix(plane_axis, plane_angle)
        self.x_moon=self.x_moon @ Rot_mat
        for i, HaloPos in enumerate(self.HaloData):
            pos=Rot_mat@HaloPos[1:]
            new_HaloData[i]=pos

        new_HaloData=np.insert(new_HaloData.T, 0, self.HaloData.T[0], axis=0).T
        return new_HaloData
        
    def translate_HaloOrbit_to_plane(self, init_moonPoint, haloData):
        """
        move halo orbit from x-axis positioned moon, to point on lunar orbit
        """
        #rotate plane
        init_axis=np.cross(init_moonPoint, self.x_moon)
        init_axis/=np.linalg.norm(init_axis)
        init_angle=np.dot(self.x_moon, init_moonPoint)/(np.linalg.norm(init_moonPoint)*np.linalg.norm(self.x_moon))
        init_angle=-np.arccos(init_angle)
        moon_len=np.linalg.norm(init_moonPoint)

        new_HaloData=np.zeros((len(haloData), 3))
        Rot_mat=self.rotation_matrix(init_axis, init_angle)
        for i, HaloPos in enumerate(haloData):
            haloLen=np.linalg.norm(HaloPos)
            pos=Rot_mat@HaloPos[1:]
            new_HaloData[i]=pos
            # new_HaloData[i]=pos/np.linalg.norm(pos) *(moon_len+haloLen)
        #add time on data
        new_HaloData=np.insert(new_HaloData.T, 0, haloData.T[0], axis=0).T
        return new_HaloData

    def closest_time(self, time, orbits, HaloOrbitTime, HaloData):
        """
        find the closest time for the haslo position in terms of time in the grid
        """
        tempTime = time - orbits * HaloOrbitTime
        if tempTime < 0:
            raise ValueError("Time is less than 0")
        return np.abs(tempTime-HaloData[:,0]).argmin()
    
    def epoch_to_seconds(self, epoch):
        """
        get seconds since start of time grid
        """
        return epoch - self.epoch0

    def find_angle(self, init_moonPoint, moonPos, rot_axis):
        """
        find the angle between the initial moon position and the current moon position
        """
        angle = -np.arccos(np.dot(init_moonPoint, moonPos)/(np.linalg.norm(init_moonPoint)*np.linalg.norm(moonPos)))
        sign = np.sign(np.dot(np.cross(init_moonPoint, moonPos), rot_axis))
        return sign*angle




    def translate_to_orbit_plane(self, moonData):
        self.rot_axis=self.find_rotation_axis(moonData)
        self.init_moon_point=self.get_initial_point_on_plane(moonData)
        intermediate_data=self.rotate_plane(np.array([0,0,1]), self.rot_axis)
        self.new_HaloData=self.translate_HaloOrbit_to_plane(self.init_moon_point, intermediate_data)

    def get_HaloGW_pos(self, ep, moonPos):
        time = self.epoch_to_seconds(ep)
        orbits = np.floor(time / self.HaloOrbitTime)

        HaloOrbitPos = np.array(self.new_HaloData[self.closest_time(time, orbits, self.HaloOrbitTime, self.new_HaloData)])
        HaloOrbitPos = HaloOrbitPos[1:]
        HaloOrbitLen = np.linalg.norm(HaloOrbitPos)
        moonLen=np.linalg.norm(moonPos)

        Proj = np.eye(3)-np.outer(self.rot_axis, self.rot_axis)
        moonPos = Proj @ moonPos
        angle = self.find_angle(self.init_moon_point, moonPos, self.rot_axis)
        rotation_mat = self.rotation_matrix(self.rot_axis, angle)
        pos=HaloOrbitPos @ rotation_mat
        # return pos / np.linalg.norm(pos) * (moonLen + HaloOrbitLen)
        return pos
    
def Create_halo_point(moonData, epoch0, grid):
    Halo_orbit = HaloOrbit(epoch0)
    Halo_orbit.load_Halo_Data("mani/HaloOrbit/GateWayOrbit_prop.csv")
    Halo_orbit.translate_to_orbit_plane(moonData)
    emph=np.zeros((len(grid), 3))
    for i, ep in enumerate(grid):
        point=moonData[i]
        moonLen=np.linalg.norm(point)
        pos=Halo_orbit.get_HaloGW_pos(ep, point)
        emph[i]=pos/np.linalg.norm(pos)*moonLen
    return emph

def Create_halo_point_moon_index(moonData, epoch0, grid, index):
    Halo_orbit = HaloOrbit(epoch0)
    Halo_orbit.load_Halo_Data("mani/HaloOrbit/GateWayOrbit_prop.csv")
    Halo_orbit.translate_to_orbit_plane(moonData)
    emph=np.zeros((len(grid), 3))
    point=moonData[index]
    moonLen=np.linalg.norm(point)
    for i, ep in enumerate(grid):
        pos=Halo_orbit.get_HaloGW_pos(ep, point)
        emph[i]=pos/np.linalg.norm(pos)*moonLen
    return emph




if __name__=="__main__":
    
    import time
    t1 = time.perf_counter()
    # os.chdir("../")
    uni_config=cosmos.util.load_yaml("./universe2.yml")
    uni = cosmos.Universe(uni_config)

    ep1 = tempo.Epoch('2026-01-01T00:00:00 TT')
    ep2 = tempo.Epoch('2026-01-10T00:00:00 TT')
    ran = tempo.EpochRange( ep1, ep2 )
    timestep = 1.0
    grid = ran.createGrid(timestep) # 60 seconds stepsize
    frames = uni.frames
    icrf = frames.axesId('ICRF')
    moonPoint=frames.pointId('Moon')
    earthPoint=frames.pointId('Earth')

    moonData=np.asarray([frames.vector3(earthPoint, moonPoint, icrf, ep) for ep in grid])

    
    # emph = Create_halo_point(moonData, ep1, grid)
    rand_index=np.random.randint(0, len(grid), 10)
    fig=plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection="3d")
    for index in rand_index:
        emph=Create_halo_point_moon_index(moonData, ep1, grid, index)
        ax.plot(*emph.T, color="black")
        ax.scatter(*moonData[index].T, color="yellow")
    plt.savefig("test.png")

    print(time.perf_counter() - t1)
