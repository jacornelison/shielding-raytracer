import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import geometry as rt
import numpy as np

# Define coordinate system
X = rt.vec3(1.,0.,0.)
Y = rt.vec3(0.,1.,0.)
Z = rt.vec3(0.,0.,1.)
ORG = rt.vec3(0.,0.,0.) # Origin


dir = Z
ort = X
ort2 = Y
crad = 2

O = rt.vec3(0,0,2)
D = rt.vec3(-0.3,0,-1).norm()
d = (crad-O.x)/D.x

def plotpnt(axis,pnt):
    axis.plot3D([pnt.x], [pnt.y], [pnt.z], '.')  # red
    return

def plotvec(axis,vec1,vec2):
    axis.plot3D([vec1.x, vec2.x], [vec1.y, vec2.y], [vec1.z, vec2.z])  # Blue
    return


def getnormcyl(ccen, cdir, vorg, vdir, vdist):
    P = vorg+(vdir*vdist)
    V1 = (cdir.cross(P - ccen).norm()).norm()
    N = (V1.cross(cdir)).norm()
    return N * np.sign(N.dot(vdir))*-1

def getnormdisc(ccen,cdir, vorg, vdir, vdist):
    return cdir*np.sign(cdir.dot(vdir))*-1

def intdisc(dcen,ddir,drad,vorg,vdir):
    Q = vorg-dcen
    h = -Q.dot(ddir)/vdir.dot(ddir)

    if (Q+D*h).dist()<drad:
        return h
    else:
        return 1e39


if __name__ == "__main__":
    d = intdisc(ORG,dir,crad,O,D)

    M = (O + D * d)  # intersection point
    N = getnormdisc(ORG, dir, O,D,d)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plotvec(ax,ORG,dir)  # Blue
    plotvec(ax,ORG,ort)  # orange
    plotvec(ax,ORG,ort2)  # green

    plotpnt(ax,O)
    plotvec(ax,O,M)
    plotvec(ax,O,O+D)
    plotvec(ax,M,M+N)

    # ax.scatter3D(x,z,Q.dist())
    #ax.scatter3D(Q.x, Q.y, Q.z)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim3d(-3, 3)
    ax.set_ylim3d(-3, 3)
    ax.set_zlim3d(-3, 3)
    plt.show()
