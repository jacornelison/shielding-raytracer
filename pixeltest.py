import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import raytracer as rt
import time
import numpy as np

def fibonacci_sphere(samples=1,randomize=True):
    rnd = 1.
    if randomize:
        rnd = random.random() * samples

    points = []
    offset = 2./samples
    increment = np.pi * (3. - np.sqrt(5.));
    x,y,z = [],[],[]
    for i in range(samples):
        y.append(((i * offset) - 1) + (offset / 2));
        r = np.sqrt(1 - pow(y[-1],2))

        phi = ((i + rnd) % samples) * increment

        x.append(np.cos(phi) * r)
        z.append(np.sin(phi) * r)

        #points.append([x,y,z])

    return np.asarray(x),np.asarray(y),np.asarray(z)

t0 = time.time()
x,y,z = fibonacci_sphere(20*20,randomize=False)
Q = rt.vec3(x,y,z)
print("Took {0}".format(time.time() - t0))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(Q.x, Q.y, Q.z)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_xlim3d(-1,1)
ax.set_ylim3d(-1, 1)
ax.set_zlim3d(-1, 1)
plt.show()

