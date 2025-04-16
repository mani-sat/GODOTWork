from numba import njit
import numpy as np
from .utils import is_closer, resize_vector_to_radius, compute_projection_matrix, project_point

class VisibilityModel:
    def __init__(self, moon_radius:float=1737.4, earth_radius:float=6371):
        self.moon_radius:float = moon_radius
        self.earth_radius:float = earth_radius
        
    def los_from_gs_to_sc(self, spacecraft, groundstation) -> bool:
        """
        Determines if there is line-of-sight from one point to another.

        Parameters
        ----------
        spacecraft : (np.ndarray)
            A 3-element moon-centric vector to the spacecraft.
        groundstation : (np.ndarray)
            A 3-element moon-centric vector to the ground station.

        Returns
        -------
        bool
            True if there is line-of-sight, False otherwise.
        """
        if np.dot(groundstation, spacecraft) > 0:
            return True
        moon_centre = np.array([0, 0, 0], dtype = np.float64)
        pws = self.calculate_within(groundstation, spacecraft,
                                    moon_centre,
                                    self.moon_radius)
        return not pws
    
    def sun_light_on_spacecraft(self, moon_sun, moon_earth, moon_sc) -> bool:
        """
        Determines if there is sunlight on the spacecraft.

        This is evaluated by determining wether the sun or moon is 
        blocking the direct path of sunlight, by projecting the moon,
        sun and spacecraft onto a common plane.

        Parameters
        ----------
        moon_sun : (np.ndarray)
            A 3-element moon-centric vector to the sun.
        moon_earth : (np.ndarray)
            A 3-element moon-centric vector to the earth centre.
        moon_sc : (np.ndarray)
            A 3-element moon-centric vector to the spacecraft.

        Returns
        -------
        bool
            True if there is sunlight on spacecraft, False otherwise.
        """
        sun_earth = moon_sun - moon_earth
        sun_sc = moon_sun - moon_sc
        if is_closer(sun_earth, sun_sc):
            if self.calculate_within(moon_sun, 
                                     moon_sc, 
                                     moon_earth, 
                                     self.earth_radius):
                return False

        if is_closer(moon_sun, sun_sc):
            moon_centre = np.array([0, 0, 0], dtype=np.float64)
            if self.calculate_within(moon_sun,
                                          moon_sc, 
                                          moon_centre,
                                          self.moon_radius):
                return False
        return True
    
    def sun_light_on_moon(self, moon_sun, moon_earth, moon_sc) -> bool:
        moon_point = resize_vector_to_radius(moon_sc, self.moon_radius)

        sun_earth = moon_sun - moon_earth
        sun_point = moon_sun - moon_point

        if is_closer(sun_earth, sun_point):
            if self.calculate_within(moon_sun,
                                     moon_point, 
                                     moon_earth,
                                     self.earth_radius):
                return False
        return np.dot(moon_point, moon_sun) > 0
    
    def calculate_within(self, basis, point, sphere_centre, radius):
        """Checks if a projected point is within a projected sphere.

        **All vectors** (`basis`, `point`, `sphere_centre`)
        **must share the same origin.**

        Parameters
        ----------
        basis : (np.ndarray)
            A 3-element array representing the basis of the projection.
        point : (np.ndarray)
            A 3-element array representing the point to be projected.
        sphere_centre : (np.ndarray)
            A 3-element array representing the center of the sphere
            before projection.
        radius : (float):
            The radius of the sphere. Must be greater than 0.

        Returns
        -------
        bool
            True if the projected point is within the projected sphere,
            False otherwise.
        """
        P = compute_projection_matrix(basis)
        point_projected = project_point(P, point)
        centre_projected = project_point(P, sphere_centre)
        pws = self.point_within_sphere(point_projected, centre_projected, radius)
        return pws

    @staticmethod
    @njit
    def point_within_sphere(point, centre, radius):
        """ Calculates wether the point is located within a sphere
        Math
        ----
        (x0 - x)² + (y0 - y)² + (z0 - z)² = r² => x² + y² + z² = r

        Parameters
        ----------
        point : (np.ndarray)
            The point to be valuated, giving the x, y and z.
        centre : (np.ndarray)
            The centre of the sphere, giving x0, y0 and z0.
        radius : float
            The radius of the sphere, giving r
        
        """
        #np.sum(np.power(centre - point, 2)) < np.power(radius, 2)
        distance = centre - point
        length_sq = (distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2])
        radius_sq = radius * radius
        return length_sq < radius_sq

    @staticmethod
    @njit
    def get_elevation(vec) -> np.float16:
        """ Calculates the elevation of the spacecraft, as seen from a GS

        Parameters
        ----------
        GS : str
            The ground stations that the elevation is evaluated for
        timestamp : godot.core.tempo.Epoch
            Timestamp at which the function is evaluated
        uni : godot.cosmos.universe
            The universe to evaluate in. This contains ephemeresis,
            nutations, reference frames etc.
        
        Returns
        -------
        np.float16
            The elevation of the SC as seen from groundstation
        """
        rxy = np.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
        elev = np.arctan2(vec[2], rxy)
        return elev
    
if __name__ == "__main__":
    NN_moon_block = VisibilityModel()