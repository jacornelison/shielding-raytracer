import raytracer as rt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time, random
from config import configDict
import argparse
import glob
import os.path as op

# Define coordinate system
X = rt.vec3(1.,0.,0.)
Y = rt.vec3(0.,1.,0.)
Z = rt.vec3(0.,0.,1.)

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


def camera(pos, dir, ort, Nxpix, Nypix, proj="flat", fov=np.pi/2, PLOT=False):
    dir.norm()
    ort.norm()
    r = float(Nxpix) / Nypix
    ort2 = dir.cross(ort)
    N = Nxpix*Nypix
    V3 = rt.vec3(dir.x * np.ones(np.size(N)), dir.y * np.ones(np.size(N)), dir.z * np.ones(np.size(N)))
    V1 = rt.vec3(ort.x * np.ones(np.size(N)), ort.y * np.ones(np.size(N)), ort.z * np.ones(np.size(N)))
    V2 = rt.vec3(ort2.x * np.ones(np.size(N)), ort2.y * np.ones(np.size(N)), ort2.z * np.ones(np.size(N)))
    #E = Y * -1
    #E = E.rotate(Z, roll).rotate(X, tilt).rotate(Y, pan)
    if proj=="flat":
        # Screen coordinates: x0, y0, x1, y1.
        t = np.tan(fov/2)
        S = (-t, -t/r , t, t / r )

        x = np.tile(np.linspace(S[0], S[2], Nxpix), Nypix)
        z = np.repeat(np.linspace(S[1], S[3], Nypix), Nxpix)
        r = np.sqrt(x**2+z**2)
        th = np.arctan2(z,x)
        #Q = rt.vec3(x,0,z)
        Q = (V1.rotate(V3, th)*r+V3).norm()
        #E = Y*-1 #rt.vec3(0,0,0)
        #E = E.rotate(Z,dk*np.pi/180).rotate(Y,(el-90)*np.pi/180).rotate(Z,az*np.pi/180)
        #Q = (V-E).norm()
        xpix = x
        ypix = z
        #return pos, (V-E).norm()

    elif proj=="sphere":
        fov = np.pi
        S = (-1./2,-1./2,1./2,1./2)
        #S = (0.,0.,1.,1.)
        S = tuple(i * fov for i in S)

        r = np.tile(np.linspace(S[0], S[2], Nxpix), Nypix)
        th = np.repeat(np.linspace(S[1], S[3], Nypix), Nxpix)


        Q = V3.rotate(V2,th).rotate(V1,r)
        xpix = r
        ypix = th

    elif proj=="sphere2":
        x,y,z = fibonacci_sphere(Nxpix*Nypix,randomize=False)

        Q = rt.vec3(x,y,z)

        xpix = x
        ypix = y
    if PLOT:

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot3D([0,dir.x], [0,dir.y], [0,dir.z]) # Blue
        ax.plot3D([0,ort.x], [0,ort.y], [0,ort.z]) # orange
        ax.plot3D([0,ort2.x], [0,ort2.y], [0,ort2.z]) # green
        #ax.scatter3D(x,z,Q.dist())
        ax.scatter3D(Q.x, Q.y, Q.z)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_xlim3d(-1,1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-1, 1)
        plt.show()

    return pos, Q.norm(), xpix, ypix

def mountxform(pos, dir, ort, az, el, dk, mnt):
    # Position

    pos = pos + rt.vec3(mnt["aptoffr"],0.,0.)
    pos = pos.rotate(Z,dk + mnt["drumangle"] - np.pi/2)
    pos = pos + rt.vec3(0.,0.,mnt["aptoffz"])
    pos = pos.rotate(Y,np.pi/2-el)
    pos = pos + rt.vec3(0.,0.,mnt["eloffz"])
    pos = pos.rotate(Z, -az)

    dir = dir.rotate(Z, dk + mnt["drumangle"] - np.pi / 2).rotate(Y, np.pi / 2 - el).rotate(Z, -az)
    ort = ort.rotate(Z, dk + mnt["drumangle"] - np.pi / 2).rotate(Y, np.pi / 2 - el).rotate(Z, -az)
    return pos, dir, ort


# Y-Z'-Y'' Rotation Matrix
def zyz(V,a,b,g):
    V = V.rotate(Z, g)
    XP = X.rotate(Z,g)
    YP = X.rotate(Z, g)
    ZP = X.rotate(Z, g)


    V = V.rotate(YP, b)
    XPP = X.rotate(YP, b)
    YPP = X.rotate(YP, b)
    ZPP = X.rotate(YP, b)

    return V.rotate(ZPP,a)

# Options for the arg-parser
###############################################
def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--title",
                        help="Choose a filename. Defaults to {0}_''fig''.png".format(def_filename),
                        default=def_filename)
    parser.add_argument("--dir",
                        help="dir in filename",
                        default=def_dir)
    parser.add_argument("--fbn",
                        help="Set number of forebaffles. Default=2",
                        default=2,
                        type=int)
    parser.add_argument("--cam",
                        help="Save a snapshot of the configuration (normally off)",
                        default=False,
                        action="store_true")
    parser.add_argument("--spherecam",
                        help="Save a snapshot of the typical view from the window (normally off)",
                        default=False,
                        action="store_true")
    parser.add_argument("--config",
                        help="Configuration file to use for dimensions, etc..., default ''BA''",
                        default=def_dict,
                        )
    parser.add_argument("--az",
                        help="Set telescope Azimuth",
                        default=0.,
                        type=float)
    parser.add_argument("--el",
                        help="Set telescope Azimuth",
                        default=90.,
                        type=float)
    parser.add_argument("--dk",
                        help="Set telescope Azimuth",
                        default=0.,
                        type=float)
    return parser, parser.parse_args()




if __name__ == "__main__":
    def_dir = op.join(op.expanduser("~"), "shielding-raytacer", "data")
    def_filename = "raytrace"
    def_dict = "BA"
    parser, args = get_args()

    # File handling stuff
    if args.title == "":
        lof = glob.glob(op.join(args.dir, "*.csv"))
        if len(lof) > 0:
            args.title = max(lof, key=op.getmtime)
            filename = '{0}'.format(args.title)
        else:
            raise NameError("No CSV files found in {0}".format(args.dir))
    else:
        if not args.title[-4::] == ".csv":
            args.title = args.title + ".csv"
        filename = op.join(args.dir, args.title)

    if op.isfile(filename) and not args.ow:
        run_num = 0
        print("File: {0} already exists!".format(args.title))
        while op.isfile(filename):
            run_num = run_num + 1
            filename = op.join(args.dir, args.title + "{0}.csv".format(run_num))
        print("Using file: " + filename)


    cf = configDict[args.config]
    t0 = time.time()

    rgb = rt.vec3
    defpos = rt.vec3(0.,0.,0.)
    defdir = Z
    defort = X

    # Create Groundshield
    gspos = defpos
    gsdir = defdir
    gsrad = cf["gsrad"] # in meters
    gsheight = cf["gsheight"] # in meters

    # Create Forebaffle
    fbpos = defpos
    fbdir = defdir
    fbort = defort
    fbrad = cf["fbrad"]  # in meters
    fbheight = cf["fbheight"] # in meters

    # Translate/Orient Forebaffle as done in the pointing model
    azdeg = args.az # in degrees
    eldeg = args.el # in degrees
    dkdeg = args.dk # in degrees

    az = azdeg*np.pi/180.
    el = eldeg*np.pi/180.
    dk = dkdeg*np.pi/180.

    mountdict = {
        "aptoffr" : 30*2.54/100, # in meters
        "drumangle" : 0, # RADIANS
        "aptoffz" : 1, # in meters
        "eloffz" : 1, # in meters
    }


    fbpos, fbdir, fbort = mountxform(defpos, defdir, defort, az, el, dk, cf)
    fb = rt.Cylinder(fbpos, fbrad, fbdir, cap=fbheight, diffuse=rgb(0, 1., 0.),mirror=0.02)

    mountdict["drumangle"] = np.pi
    fbpos, fbdir, fbort = mountxform(defpos, defdir, defort, az, el, dk, mountdict)
    fb2 = rt.Cylinder(fbpos, fbrad, fbdir, cap=fbheight, mirror=0.02)

    gs = rt.Cylinder(gspos, gsrad, gsdir, cap=gsheight, diffuse=rgb(5, 0., 0.), mirror=0.)
    floor = rt.CheckeredSphere(rt.vec3(0, 0, -99999.5), 99999.4, rgb(.75, .75, .75), 0.25)

    # ds2 = rt.Sphere(rt.vec3(0, -3., 0.5), .6, rgb(0., 0.7, 0.), mirror=0.02)
    # ds3 = rt.Sphere(rt.vec3(3., 0., 0.5), .6, rgb(0., 0., .5), mirror=0.02)
    # ds4 = rt.Sphere(rt.vec3(-3., 0., 0.5), .6, rgb(.5, .223, .5), mirror=0.02)
    scene = [
        gs,
        fb,
        fb2,
        floor,
    ]

    if args.config == "test":
        sppos = rt.vec3(0,0,10.)
        spr = 0.5
        scene.append(rt.Sphere(sppos, spr, rgb(5, 0., 0.), mirror=0.02))
        ang = 2*np.arcsin(spr/sppos.dist()) # angular diameter of the sphere.
        print(ang)
        sphcov = (ang/2)**2/4 # Area of the solid angle divided by 4pi
        print("Expecting fractional coverage of: {0}".format(sphcov))

    # Get Camera View
    if args.cam:
        w, h = (800, 400)
        camaz = 0
        camel = -30
        camdk = -90
        camdir = Z.rotate(Z, camdk * np.pi / 180).rotate(Y, (90 - camel) * np.pi / 180).rotate(Z, -camaz * np.pi / 180)
        camort = X.rotate(Z, camdk * np.pi / 180).rotate(Y, (90 - camel) * np.pi / 180).rotate(Z, -camaz * np.pi / 180)
        O, D, xp, yp = camera(rt.vec3(-10., 0., 8.), camdir, camort, w, h, proj="flat", fov=np.pi / 2.2)
        color = rt.raytrace(O, D, scene,hitmap=False)
        rgb = [Image.fromarray((255 * np.clip(c, 0, 1).reshape((h, w))).astype(np.uint8), "L") for c in color.components()]
        Image.merge("RGB", rgb).save("camview.png")

    # Grab Hit Map
    if args.spherecam:
        w, h = (400, 400)

        O, D, xp, yp = camera(fbpos,fbdir,fbort,w,h,proj="sphere", fov=2*np.pi)
        color = rt.raytrace(O, D, scene, hitmap=True)
        rgb = [Image.fromarray((255 * np.clip(c, 0, 1).reshape((h, w))).astype(np.uint8), "L") for c in color.components()]
        Image.merge("RGB", rgb).save("fbview.png")

    if False:
        fig = plt.figure()
        ax = Axes3D(fig)

        plt.subplot(projection="polar")
        print(np.shape(np.reshape(yp,(h,w))))
        cd = color.dist()
        x, y, c = np.reshape(xp,(h,w)),np.reshape(yp,(h,w)), np.reshape(cd,(h,w))
        ind =(cd>0.5)&(cd<1)
        r = np.size(np.where(ind))
        print((r/(w*h)))
        plt.pcolormesh(x, y, c)
        plt.grid()
        plt.colorbar()
        plt.show()

    if True:
        w, h = (400, 400)
        #w, h = (40, 40)
        O, D, xp, yp = camera(fbpos, fbdir, fbort, w, h, proj="sphere2")
        color = rt.raytrace(O, D, scene, hitmap=True)
        cd = color.dist()
        ind =(cd==5.)
        r = np.size(np.where(ind))
        print("Fractional coverage: {0}".format(r / (w * h)))
        #print(np.unique(cd))

    print("Took {0}".format(time.time() - t0))
    print("Complete!")
