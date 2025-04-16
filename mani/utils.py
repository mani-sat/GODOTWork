from datetime import datetime
from pytz import UTC
from godot.core import tempo
from numba import njit
import numpy as np

def convert_to_datetime(timestamp: tempo.Epoch):
    dts = datetime.strptime(timestamp.calStr('TT'), '%Y-%m-%dT%H:%M:%S.%f TT').astimezone(UTC)
    return dts

def get_date_string(timestamp: tempo.Epoch):
    return convert_to_datetime(timestamp).strftime('%Y-%m-%d')


class EventGrid():
    """
    A simple structure to hold and return the EpochRange grid.

    *This class has been created to keep Event Grids in scopes*
    *and not stored in memory*
    """

    def __init__(self, t1: tempo.Epoch, t2: tempo.Epoch, res: float):
        self.t1 = t1
        self.t2 = t2
        self.resolution = res
        
    def get_event_grid(self):
        """
        Create and fetch the event grid

        Returns
        -------
        list[godot.core.tempo.Epoch]
            An grid of timestamps
        """
        return tempo.EpochRange(self.t1, self.t2).createGrid(self.resolution)
    
    

@njit
def compute_projection_matrix(basis):
    """
    Computes the 3x3 projection matrix that projects a 3D
    point onto a plane with the given normal vector.

    Parameters
    ----------
    basis : (np.ndarray)
        An 3-element vector giving the basis for the projection.

    Returns
    -------
    (np.ndarray)
        The 3 x 3 projection matrix for the normal-plane
        of the basis.
    """
    if basis.shape != (3,):
        raise ValueError("The basis vector must be a 3-element vector.")
    basis_len = get_len(basis)
    if basis_len == 0:
        raise ValueError("The basis vector cannot be a zero vector.")
    normal = basis / basis_len  # Normalize the normal vector
    return get_eye() - calc_outer(normal) # Projection matrix
    
@njit
def resize_vector_to_radius(vector, radius):
    scale = radius / get_len(vector)
    return vector * scale


@njit
def project_point(P, point):
    projected_point = P @ point  # Apply projection
    return projected_point

@njit
def is_closer(vector1:float, vector2:float) -> bool:
    """Checks if `vector1` is shorter than `vector2`."""
    return get_len(vector1) < get_len(vector2)
    
@njit
def get_eye():
    return np.array([[1,0,0],[0,1,0],[0,0,1]],dtype=np.float64)

@njit
def get_len(basis):
    return np.sqrt(basis[0] * basis[0] + basis[1] * basis[1] + basis[2] * basis[2])

@njit
def calc_outer(v):
    return np.array([[v[0]*v[0], v[0] * v[1], v[0] * v[2]],
                     [v[1]*v[0], v[1] * v[1], v[1] * v[2]],
                     [v[2]*v[0], v[2] * v[1], v[2] * v[2]]])