import numpy as np
import matplotlib.pyplot as plt
from utils_luke import to_standard_units


f=open("HaloOrbit/GateWayOrbit_prop.csv","r")
lines=f.readlines()
print(lines[0])
lines = lines[1:]
f.close()
lines = [[float(element) for element in line.strip().split(",")] for line in lines]
#change to seconds and km
HaloData = np.array([[to_standard_units(line[0], "TU"), to_standard_units(line[1], "LU"), to_standard_units(line[2], "LU"), to_standard_units(line[3], "LU")] for line in lines])

moon=np.array([to_standard_units(1, "LU"), 0, 0])
L1=np.array([to_standard_units(0.83691513, "LU"), 0, 0])
L2=np.array([to_standard_units(1.15568217, "LU"), 0, 0])

fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(111, projection="3d")
ax.plot(*HaloData.T[1:], color="blue", label="Halo Orbit")
ax.scatter(*L1, label="L1", color="green")
ax.scatter(*L2, label="L2", color="black")
ax.scatter(*moon, label="moon", color="gold", s=60)
ax.view_init(elev=20, azim=20)

ax.set_xlabel("X [km]")
ax.set_ylabel("Y [km]")
ax.set_zlabel("Z [km]")

ax.legend()
ax.axis("equal")
fig.savefig("HaloOrbit/OriginalHaloData.png", bbox_inches='tight')