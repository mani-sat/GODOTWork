import numpy as np

def to_standard_units(val, state):
    geo=6371+36000 #km
    LD=9.12 #GEO
    LU=389703 #km length unit
    TU=382981 #s time unit
    # LU and TU come from https://ssd.jpl.nasa.gov/tools/periodic_orbits.html
    if state=="GEO":
        return val*geo
    if state=="LD":
        return val*geo*LD
    if state=="LU":
        return val*LU
    if state=="LU/TU":
        return val*LU/TU
    if state=="TU":
        return val*TU
    else:
        raise Exception("not an implemented unit")

def formatter(a):
    z=np.array([float(e) for e in a.split("\t") if e!=""])
    pos=np.zeros(3)
    vel=np.zeros(3)
    for i in range(3):
        pos[i]=to_standard_units(z[i],"LU")
        vel[i]=to_standard_units(z[i+3],"LU/TU")
    return pos,vel

def rodrigues(pos, rot_axis, ang):
    v1=pos*np.cos(ang)
    v2=np.cross(rot_axis, pos)*np.sin(ang)
    v3=rot_axis*np.dot(rot_axis, pos)*(1-np.cos(ang))
    return v1+v2+v3