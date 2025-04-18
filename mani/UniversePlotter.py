import matplotlib.pyplot as plt
import numpy as np
import godot
from .VisibilityModel import VisibilityModel

class Sphere():
    def __init__(self, coordinates, radius, name, points = 1000):
        """Helper function to plot a sphere."""
        self.centre = coordinates
        self.radius = radius
        u = np.linspace(0, 2 * np.pi, points)
        v = np.linspace(0, np.pi, points)
        self.x = coordinates[0] + radius * np.outer(np.cos(u), np.sin(v))
        self.y = coordinates[1] + radius * np.outer(np.sin(u), np.sin(v))
        self.z = coordinates[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        self.name = name

    def plot(self, ax, c, alpha = 1.0,):
        ax.plot_wireframe(self.x, self.y, self.z, alpha=alpha, label=self.name, color=c)

    def get_x_y_z(self):
        return self.x, self.y, self.z
    
    def set_2d(self, x, y):
        self.x2d = x
        self.y2d = y

class Plane():
    def __init__(self, centre, normal, size, density):
        self.centre = centre
        self.normal = normal

        d = -np.dot(centre, normal)
        # create x,y
        ran = np.linspace(-size //2, size // 2, density)
        xx, yy = np.meshgrid(ran,ran)
        self.xx = xx
        self.yy = yy

        # calculate corresponding z
        plane =  (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]
        self.plane = plane

    def plot(self, ax, vector_length):
        xx = self.xx
        yy = self.yy
        z = self.plane

        ax.plot_surface(xx, yy, z, alpha=0.2, color='gold')
        unit = self.normal/np.linalg.norm(self.normal) * vector_length
        ax.quiver(*self.centre, *unit,color='gold', label = 'Sunlight normal')

class UniversePlotter:
    def __init__(self, godot_handler, gs, t, target_elevation):
        self.timestamp = t
        self.groundstation_name = gs
        uni = godot_handler.fetch_universe()
        self.uni = uni
        self.moon = uni.frames.vector3('Moon', 'Moon', 'ICRF', t)
        self.sun = uni.frames.vector3('Moon', 'Sun', 'ICRF', t)
        self.groundstation = uni.frames.vector3('Moon', gs, 'ICRF', t)
        self.earth = uni.frames.vector3('Moon','Earth', 'ICRF', t)
        self.earth_radius = np.linalg.norm(self.groundstation - self.earth)
        self.elevation = uni.frames.vector3(gs,'SC',gs, t)
        self.spacecraft = uni.frames.vector3('Moon','SC', 'ICRF', t)
        self.gateway = godot_handler._get_mooncentric_GW_pos(self.earth,t)
        self.vismod = VisibilityModel()
        self.target_elevation = target_elevation

    def change_groundstation(self, gs):
        t = self.timestamp
        self.groundstation_name = gs
        self.groundstation = self.uni.frames.vector3('Moon', gs, 'ICRF', t)
        self.elevation = self.uni.frames.vector3(gs,'SC',gs, t)
    
    def plot_moon(self, ax):
        Sphere(self.moon, 1737.4, 'Moon').plot(ax, 'darkgray', 0.25)
    
    def plot_earth(self, ax):
        Sphere(self.earth, self.earth_radius, 'Earth').plot(ax, 'forestgreen', 0.25)
    
    def plot_spacecraft(self, ax):
        ax.scatter(*(self.spacecraft), color='blue', s = 50, label = 'Spacecraft')
        #Sphere(self.spacecraft, 100, 'Spacecraft').plot(ax, 'blue',1)
    
    def plot_gateway(self, ax):
        ax.scatter(*(self.gateway), color='cyan', s = 50, label = 'Gateway')
        #Sphere(self.gateway, 100, 'Gateway').plot(ax,'cyan',1)
    
    def plot_sun_plane(self, ax, size = 1000):
        Plane(self.moon, self.sun, size, 100).plot(ax, 1000)
    
    def plot_groundstation(self,ax):
        ax.scatter(*(self.groundstation), color = 'purple', s = 100, label = self.groundstation_name)
        #Sphere(self.groundstation, 500, self.groundstation_name).plot(ax,'purple',1)
    
    def plot_gs_vec(self,ax):
        elevation = self.vismod.get_elevation(self.elevation)
        moonVisible = self.vismod.los_from_gs_to_sc(self.spacecraft, self.groundstation)
        condition = moonVisible > 0 and elevation > self.target_elevation
        self.plot_vector(ax, self.spacecraft, self.groundstation,':', condition, f'SC to {self.groundstation_name}')
    
    def plot_gw_vec(self,ax):
        gw_vis = self.vismod.los_from_gs_to_sc(self.spacecraft, self.gateway)
        self.plot_vector(ax, self.spacecraft, self.gateway,'-.', gw_vis, 'SC to GW')

    
    @staticmethod
    def plot_vector(ax, start, end, style, condition, label):
        vec = np.linspace(start, end, 100)
        c = 'g' if condition else 'r'
        ax.plot(vec[:, 0], vec[:, 1], vec[:, 2], label=label, c=c, ls=style)

    def plot_universe(self, ax):
        self.plot_moon(ax)
        self.plot_earth(ax)
        self.plot_spacecraft(ax)
        self.plot_gateway(ax)
        self.plot_groundstation(ax)

        self.plot_sun_plane(ax)
        self.plot_gs_vec(ax)
        self.plot_gw_vec(ax)
        
        ax.legend()

    def set_view_earth_focuced(self, ax, view_distance = 5000):
        """Set axis limits based on the desired view."""
        earth = self.earth
        ax.set_xlim([earth[0]-view_distance, earth[0]+view_distance])
        ax.set_ylim([earth[1]-view_distance, earth[1]+view_distance])
        ax.set_zlim([earth[2]-view_distance, earth[2]+view_distance])

    def set_view_moon_focuced(self, ax, view_distance = 2000):
        """Set axis limits based on the desired view."""
        moon = self.moon
        ax.set_xlim([moon[0]-view_distance, moon[0]+view_distance])
        ax.set_ylim([moon[1]-view_distance, moon[1]+view_distance])
        ax.set_zlim([moon[2]-view_distance, moon[2]+view_distance])
    
    def get_gs_los_status(self):
        return self.vismod.los_from_gs_to_sc(self.spacecraft, self.groundstation)
    
    def get_gw_los_status(self):
        return self.vismod.los_from_gs_to_sc(self.spacecraft, self.gateway)
    
    def get_gs_elevation(self):
        return self.vismod.get_elevation(self.elevation)
    
    def get_sunlight_on_sc(self):
        return self.vismod.sun_light_on_spacecraft(self.sun, self.earth, self.spacecraft)
    
    def get_sunlight_on_moon(self):
        return self.vismod.sun_light_on_moon(self.sun, self.earth, self.spacecraft)
    
    def print_status(self):
        moon_visible = self.get_gs_los_status()

        gw_vis = self.get_gw_los_status()
        elevation = self.get_gs_elevation()


        slos = self.get_sunlight_on_sc()
        slon = self.get_sunlight_on_moon()

        s1 = f"Does moon block los from SC to GS: {'Yes' if moon_visible < 0 else 'No'}"
        s2 = f"Does moon block los from SC to GW: {'Yes' if gw_vis < 0 else 'No'}"
        s3 = f"Does earth block: {'Yes' if elevation < self.target_elevation else 'No'}"
        s4 = f"Is the spacecraft in sunlight? {'Yes' if slos else 'No'}"
        s5 = f"Is the moon in sunlight? {'Yes' if slon else 'No'}"
        
        print(f"Time: {self.timestamp}")
        print(f"\t{s1}")
        print(f"\t{s2}")
        print(f"\t{s3}")
        print(f"\t{s4}")
        print(f"\t{s5}")
