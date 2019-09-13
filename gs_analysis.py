import raytracer as rt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

# Define coordinate system
X = rt.vec3(1.,0.,0.)
Y = rt.vec3(0.,1.,0.)
Z = rt.vec3(0.,0.,1.)

def window(pos, rdir, rad, res, xang=0, yang=0):
    # Create ray bundle that represents a point on the window
    V = rt.vec3
    return pos, rdir.norm()

def camera(pos, dir, ort, Nxpix, Nypix, proj="flat"):
    dir.norm()
    ort.norm()
    r = float(Nxpix) / Nypix
    #E = Y * -1
    #E = E.rotate(Z, roll).rotate(X, tilt).rotate(Y, pan)
    if proj=="flat":
        # Screen coordinates: x0, y0, x1, y1.
        S = (-1., 1. / r , 1., -1. / r )
        print(S)
        x = np.tile(np.linspace(S[0], S[2], Nxpix), Nypix)
        y = np.repeat(np.linspace(S[1], S[3], Nypix), Nxpix)
        #plt.figure()
        #plt.plot(x, y, '.')
        #plt.show()
        az = -45
        el = 90
        dk = 0
        V = rt.vec3(x, 0, y).rotate(Z,dk*np.pi/180).rotate(Y,(el-90)*np.pi/180).rotate(Z,-az*np.pi/180)
        E = Y*-1 #rt.vec3(0,0,0)
        E = E.rotate(Z,dk*np.pi/180).rotate(Y,(el-90)*np.pi/180).rotate(Z,az*np.pi/180)
        return pos, (V-E).norm()
    elif proj=="sphere":
        fov = np.pi
        S = (-1.,1.,1.,-1.)
        S = tuple(i * fov for i in S)
        print(S)
        r = np.tile(np.linspace(S[0], S[2], Nxpix), Nypix)
        th = np.repeat(np.linspace(S[1], S[3], Nypix), Nxpix)
        print(np.size(r))
        print(np.size(th))
        x = r*np.cos(th)
        y = r*np.sin(th)
        #plt.figure()
        #plt.plot(x,y,'.')
        #plt.show()
        V = rt.vec3(x, 0, y)  # .rotate(Z, roll).rotate(X, tilt).rotate(Y, pan)
    elif proj=="azel":
        fov = np.pi/4
        S = (-1.,-1.,1.,1.)
        S = tuple(i * fov for i in S)

        r = np.tile(np.linspace(S[0], S[2], Nxpix), Nypix)
        th = np.repeat(np.linspace(S[1], S[3], Nypix), Nxpix)
        ort2 = dir.cross(ort)

        Q = dir.rotate(ort2,-th).rotate(ort,r)

        plt.figure()
        plt.plot(Q.norm().dist(), '.')
        plt.show()
        return pos, Q.norm()


    #Q = (V - E).norm()
    #plt.plot(Q.x,Q.y,'.')
    #plt.show()

    return pos, (V-E).norm()

def mountxform(pos, dir, ort, az, el, dk, mnt):
    # Position
    pos.x = pos.x + mnt["aptoffr"]
    pos = pos.rotate(Z,dk + mnt["drumangle"] - np.pi/2)
    pos.z = pos.z + mnt["aptoffz"]
    pos = pos.rotate(Y,np.pi/2-el).rotate(Z,-az)

    dir = dir.rotate(Z, dk + mnt["drumangle"] - np.pi / 2).rotate(Y, np.pi / 2 - el).rotate(Z, -az)
    ort = ort.rotate(Z, dk + mnt["drumangle"] - np.pi / 2).rotate(Y, np.pi / 2 - el).rotate(Z, -az)
    return pos, dir, ort


# Y-Z'-Y'' Rotation Matrix
def yzy(V,a,b,g):
    V = V.rotate(Y, g)
    XP = X.rotate(Y,g)
    YP = X.rotate(Y, g)
    ZP = X.rotate(Y, g)


    V = V.rotate(ZP, b)
    XPP = X.rotate(ZP, b)
    YPP = X.rotate(ZP, b)
    ZPP = X.rotate(ZP, b)

    return V.rotate(YPP,a)


t0 = time.time()

rgb = rt.vec3
defpos = rt.vec3(0.,0.,0.)
defdir = Z
defort = X

# Create Groundshield
gspos = defdir
gsdir = defdir
gsrad = 7. # in meters
gsheight = 7. # in meters

# Create Forebaffle
fbpos = defpos
fbdir = defdir
fbort = defort
fbrad = 1. # in meters
fbheight = 0.5 # in meters

# Translate/Orient Forebaffle as done in the pointing model
azdeg = 0 # in degrees
eldeg = 90 # in degrees
dkdeg = 0 # in degrees

az = azdeg*np.pi/180.
el = eldeg*np.pi/180.
dk = dkdeg*np.pi/180.


mountdict = {
    "aptoffr" : 1, # in meters
    "drumangle" : 0, # Not implementing this
    "aptoffz" : 1, # in meters
}

fbpos, fbdir, fbort = mountxform(fbpos, fbdir, fbort, az, el, dk, mountdict)

#fbpos = rt.vec3(0., 1., 1.0)

#gs = rt.Cylinder(gspos, gsrad, gsdir, cap=gsheight, diffuse=rgb(0.8, 0.1, 0.1), mirror=0.)
#fb = rt.Cylinder(fbpos, fbrad, fbdir, cap=fbheight, diffuse=rgb(0, .7, 0.3))
floor = rt.CheckeredSphere(rt.vec3(0, 0, -99999.5), 99999.4, rgb(.75, .75, .75), 0.25)
ds1 = rt.Sphere(rt.vec3(0, 3., 0.5), .6, rgb(0.5, 0., 0.), mirror=0.02)
ds2 = rt.Sphere(rt.vec3(0, -3., 0.5), .6, rgb(0., 0.7, 0.), mirror=0.02)
ds3 = rt.Sphere(rt.vec3(3., 0., 0.5), .6, rgb(0., 0., .5), mirror=0.02)
ds4 = rt.Sphere(rt.vec3(-3., 0., 0.5), .6, rgb(.5, .223, .5), mirror=0.02)
scene = [
    #gs,
    #fb,
    floor,
    ds1,
    ds2,
    ds3,
    ds4,
]

w, h = (800, 800)

camaz = -10
camel = 0
camdk = 0
camdir = Z.rotate(Z,camdk*np.pi/180).rotate(Y,(90-camel)*np.pi/180).rotate(Z,-camaz*np.pi/180)
camort = X.rotate(Z,camdk*np.pi/180).rotate(Y,(90-camel)*np.pi/180).rotate(Z,-camaz*np.pi/180)
O, D = camera(rt.vec3(0.,0.,1.),camdir,camort,w,h,proj="flat")
color = rt.raytrace(O, D, scene,hitmap=False)
print("Took {0}".format(time.time() - t0))

rgb = [Image.fromarray((255 * np.clip(c, 0, 1).reshape((h, w))).astype(np.uint8), "L") for c in color.components()]
Image.merge("RGB", rgb).save("fig.png")

