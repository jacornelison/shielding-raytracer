import geometry as rt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import time, random
from config import configDict
import argparse
import glob
import os.path as op
import scipy.interpolate as intrp


# Define coordinate system
X = rt.vec3(1.,0.,0.)
Y = rt.vec3(0.,1.,0.)
Z = rt.vec3(0.,0.,1.)
ORG = rt.vec3(0.,0.,0.) # Origin
conv = 41252 # sq-degrees over the unit sphere


def fibonacci_sphere(samples=1):

    indices = np.arange(0, samples, dtype=float) + 0.5

    phi = np.arccos(1 - 2 * indices / samples)
    theta = np.pi * (1 + 5 ** 0.5) * indices

    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi);

    #return np.asarray(x),np.asarray(y),np.asarray(z)
    return x, y, z

def camera(pos, dir, ort, Nxpix, Nypix, proj="flat", fov=2*np.pi, PLOT=False):
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


    # Samples approximately evenly across the unit sphere using a fibonacci sphere.
    elif proj=="sphere2":
        x,y,z = fibonacci_sphere(Nxpix*Nypix)

        V0 = rt.vec3(x,y,z)
        r =  np.arccos(V0.dot(V3))
        theta = np.arctan2(V0.dot(V2),V0.dot(V1))

        ind = (r<fov/2)
        Q = V0.extract(ind)

        r = r[ind]
        theta = theta[ind]

        x = 2*np.sin(r/2)*np.cos(theta)
        y = 2*np.sin(r/2)*np.sin(theta)


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

def show_temp_map(xp,yp,temp,fname,fov=np.pi, log=False):
    if log:
        temp = 10*np.log10(temp+1e-10)
        vmn = -100+0*1.1*min(temp)
        if max(temp)>0:
            vmx = 10+0*max(temp)
        else:
            vmx = 10+0
        t = "dTb ( dB(K) )"
    else:
        vmx = 1.1*max(temp)
        vmn = 0
        t = "Tb ( K )"
    stepx = (max(xp) - min(xp)) / np.sqrt(len(xp)) * (2 * np.pi / fov)
    stepy = (max(yp) - min(yp)) / np.sqrt(len(yp)) * (2 * np.pi / fov)
    grid_x, grid_y = np.meshgrid(np.arange(min(xp), max(xp), stepx), np.arange(min(yp), max(yp), stepy))

    xy = np.zeros([2, len(xp)])
    xy[0, :], xy[1, :] = xp, yp
    c = intrp.griddata(xy.T, temp, (grid_x, grid_y), method="linear")

    fig = plt.figure()
    plt.subplot()
    plt.pcolormesh(grid_x * 180 / np.pi, grid_y * 180 / np.pi, c,vmin=vmn, vmax=vmx)
    plt.xlabel("x' (deg)")
    plt.ylabel("y' (deg)")
    plt.axis('equal')
    fov = fov * 180 / np.pi
    plt.xlim([-1 * fov / 2, fov / 2])
    plt.ylim([-1 * fov / 2, fov / 2])
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(t, rotation=270)
    plt.title("Test SAT: FB + GS")
    # plt.show()
    plt.savefig(fname+"_fbview_temp.png",dpi=600,edgecolor='black')

def show_hit_map(xp,yp,color,fname,pperd,fov=np.pi):
    stepx = (max(xp) - min(xp)) / np.sqrt(len(xp)) * (2 * np.pi / fov)
    stepy = (max(yp) - min(yp)) / np.sqrt(len(yp)) * (2 * np.pi / fov)
    grid_x, grid_y = np.meshgrid(np.arange(min(xp), max(xp), stepx), np.arange(min(yp), max(yp), stepy))

    xy = np.zeros([2, len(xp)])
    xy[0, :], xy[1, :] = xp, yp
    c = intrp.griddata(xy.T, color, (grid_x, grid_y), method="linear")
    #c = np.floor(c)
    keys = clrs.keys()
    cmap = cm.get_cmap("tab20b",len(keys))
    #fig = plt.figure(figsize=(4, 4))
    fig = plt.figure()
    plt.subplot()
    plt.pcolormesh(grid_x * 180 / np.pi, grid_y * 180 / np.pi, c, cmap=cmap,vmin=0, vmax=clrs["gnd"][0])
    plt.xlabel("x' (deg)")
    plt.ylabel("y' (deg)")
    plt.axis('equal')
    fov = fov * 180 / np.pi
    plt.xlim([-1 * fov / 2, fov / 2])
    plt.ylim([-1 * fov / 2, fov / 2])

    labs = []
    for k in keys:
        labs.append(k+": {0}".format(np.around(get_hit_frac(color,clrs[k],pperd),2)*2))

    cbar = plt.colorbar( ticks=range(0,len(keys)))
    cbar.ax.set_yticklabels(labs)  # vertically oriented colorbar
    cbar.ax.set_ylabel('Fraction of Solid Angle')#, rotation=270)
    plt.title("Test SAT: FB + GS")
    # plt.show()
    plt.savefig(fname+"_fbview_hit.png",dpi=600)

def mountxform(pos, dir, ort, az, el, dk, mnt):
    # Position
    pos = pos + rt.vec3(mnt["aptoffr"],0.,0.)
    pos = pos.rotate(Z,dk + mnt["drumangle"] - np.pi/2)
    pos = pos + rt.vec3(0.,0.,mnt["aptoffz"]) + rt.vec3(mnt["dkoffx"],0.,0.) + rt.vec3(0.,mnt["dkoffy"],0.)
    pos = pos.rotate(Y,np.pi/2-el)
    pos = pos + rt.vec3(0.,0.,mnt["eloffz"]) + rt.vec3(mnt["eloffx"],0.,0.)
    pos = pos.rotate(Z, -az)

    dir = dir.rotate(Z, dk + mnt["drumangle"] - np.pi / 2).rotate(Y, np.pi / 2 - el).rotate(Z, -az)
    ort = ort.rotate(Z, dk + mnt["drumangle"] - np.pi / 2).rotate(Y, np.pi / 2 - el).rotate(Z, -az)
    return pos, dir, ort

def make_forebaffles(scene, config, az, el, dk, fbclrs, mirr=0.001):

    nfb = config["nbaffles"]
    # Make the forebaffles slightly taller but slightly lower so the camera doesn't see out the back
    for i in range(0,nfb):
        if i==0:
            clr = rgb(0.,0.,fbclrs["fbm"])
        else:
            clr = rgb(0.,fbclrs["fba"],0.)
        fbpos, fbdir, fbort = mountxform(ORG, Z, X, az, el, dk, config)
        scene.append(rt.Cylinder(fbpos, fbdir, fbort, fbrad,cap=fbheight, diffuse=clr, mirror=mirr))
        config["drumangle"] = config["drumangle"] + 2 * np.pi / nfb

    return scene

def make_mirror(scene,config,mirr_config,az,el):
    mcon = mirr_config.copy()
    mcf = config.copy()

    # Account for mount and mirror offsets
    mcf["aptoffr"] = 0
    mcf["drumangle"] = 0
    mcf["aptoffz"] = mcf["aptoffz"] + mcon["height"]

    # Transform mirror position
    mpos = ORG.copy() + rt.vec3(0, mcon["offset"], 0)

    # Transform mirror dir and orts about roll
    mdir = Z.copy() * -1
    mort = X.copy()
    mort2 = mdir.cross(mort).norm()

    mdirP = zyz(mdir, np.pi / 2, mcon["roll"] * np.pi / 180, 0,
               e1=mort, e2=mort2, e3=mdir)
    mortP = zyz(mort, np.pi / 2, mcon["roll"] * np.pi / 180, 0,
               e1=mort, e2=mort2, e3=mdir)
    mort2P = zyz(mort2, np.pi / 2, mcon["roll"] * np.pi / 180, 0,
               e1=mort, e2=mort2, e3=mdir)

    # Transform mirror dir and orts about tilt
    mdirPP = zyz(mdirP, 0, mcon["tilt"] * np.pi / 180, 0,
               e1=mortP, e2=mort2P, e3=mdirP)
    mortPP = zyz(mortP, 0, mcon["tilt"] * np.pi / 180, 0,
               e1=mortP, e2=mort2P, e3=mdirP)

    mpos, mdir, mort = mountxform(mpos, mdirPP, mortPP, az, el, 0, mcf)
    vw = mcon["dims"][0] / 2
    vh = mcon["dims"][1] / 2
    verts = np.array([[-vw, vh], [vw, vh], [vw, -vh], [-vw, -vh]])
    mirror = rt.Quad(mpos, mdir, mort, verts, diffuse=rt.rgb(0., mcon["color"], 0.), mirror=0.3)
    scene.append(mirror)
    return scene

def get_hit_frac(cmap,c_id,pperd):
    npix = int(np.sqrt(pperd*conv))
    cd = cmap.sum()
    ind = (cmap == c_id)
    return (np.size(np.where(ind)) / (npix ** 2))

# Y-Z'-Y'' Rotation Matrix
def zyz(V,a,b,g,e1=X.copy(),e2=Y.copy(),e3=Z.copy()):
    V = V.rotate(e3, g)
    XP = e1.rotate(e3,g)
    YP = e2.rotate(e3, g)
    ZP = e3.rotate(e3, g)


    V = V.rotate(YP, b)
    XPP = XP.rotate(YP, b)
    YPP = YP.rotate(YP, b)
    ZPP = ZP.rotate(YP, b)

    return V.rotate(ZPP,a)

def tic():
    global tstart
    tstart = time.time()
    return tstart

def toc(str=""):
    global tstart
    print(str)
    print("Elapsed time: {0} seconds".format(time.time()-tstart))

def window_map(scene,az,el,dk,cf,pperd=0.5,fov=np.pi):
    w = int(np.sqrt(pperd * conv))  # Pixels over the whole sphere
    winsamps = 10
    wr = cf["winrad"]
    R = cf["aptoffr"]
    k1 = k2 = np.arange(-wr * 1.5, wr * 1.5, wr / winsamps)
    K1, K2 = np.meshgrid(k1, k2)
    K1 = np.reshape(K1, (np.size(K1), 1))
    K2 = np.reshape(K2, (np.size(K2), 1))
    r = np.sqrt(K1 ** 2 + (R - K2) ** 2)
    th = np.arctan2(K1, R - K2)
    gsfrac = np.zeros((len(r), 1))
    for i in range(0, len(K1)):
        if np.sqrt(K1[i] ** 2 + K2[i] ** 2) <= wr:
            cf2 = cf.copy()
            cf2["drumangle"] = cf2["drumangle"] + th[i]
            cf2["aptoffr"] = r[i]
            fbpos, fbdir, fbort = mountxform(ORG, Z, X, az, el, dk, cf2)
            O, D, xp, yp = camera(fbpos, fbdir, fbort, w, w, proj="sphere2", fov=fov)
            color = rt.raytrace(O, D, scene, O, trace="hit")

            gsfrac[i] = get_hit_frac(color, clrs["fbm"], pperd)

    gsfrac[np.sqrt(K1 ** 2 + K2 ** 2) > wr] = np.nan
    gsfrac = np.ma.masked_where(np.isnan(gsfrac), gsfrac)

    c = np.reshape(gsfrac, (len(k1), len(k2)))
    # fig = plt.figure(figsize=(4, 4))
    fig = plt.figure()
    plt.subplot()
    plt.pcolormesh(k1, k2, c)  # , vmin=0)#, vmax=0.0025)
    plt.xlim([-1 * wr * 1.1, wr * 1.1])
    plt.ylim([-1 * wr * 1.1, wr * 1.1])
    plt.xlabel("Window X (m)")
    plt.ylabel("Window Y (m)")
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('GS subtend fraction', rotation=270)
    plt.show()


def do_test(cf):
    # This is a test of the raytracer. The scene is only made up of our spherical camera and
    # a single unit sphere. Because it is easy to predict the solid angle of a sphere at a known
    # distance, we can verify that our camera can effectively return the solid angle to an arbitrary
    # accuracy and precision, independent of where the sphere is on the sky.

    spr = 1. # Sphere radius
    scene = [0]

    plt.figure(1)
    # Check for systematics due to pixel aliasing
    # Should do this at a distance that is larger than the things you care about.
    # in our case, a ground shield that is ~10 meters away
    if True:
        sppos, spdir, sport = mountxform(rt.vec3(0, 0, 20.), Z, X, 0., np.pi / 2., 0., cf)
        scene[0] = rt.Sphere(sppos, spr, rgb(5, 0., 0.), mirror=0.02)

        ang = 2 * np.arcsin(spr / sppos.dist())  # angular diameter of the sphere.
        sphcov = (1 - np.cos(ang / 2)) / 2  # Fractional area: solid angle divided by 4pi

        res = np.logspace(0,2)
        r = np.zeros([len(res),])
        for i in range(0,len(res)):
            pperd = res[i]  # pipxels per sq-deg
            w = int(np.sqrt(pperd * conv))  # Pixels over the whole sphere
            O, D, xp, yp = camera(rt.vec3(0, 0, 0), spdir, sport, w, w, proj="sphere2")
            color = rt.raytrace(O, D, scene, hitmap=True)
            cd = color.dist()
            ind = (cd == 5.)
            r[i] = (np.size(np.where(ind))/(w**2))/sphcov-1


        plt.subplot(211)
        plt.semilogx(res,r,'.')
        plt.title("Sphere at distance of 30m")
        plt.xlabel("Pixels per sq-deg")
        plt.ylabel("Fractional residuals")
        plt.ylim([-0.01, 0.01])
        plt.grid()

    # Look at accuracy over distance when you've optimized the resolution.
    if True:
        pperd = 7 # pipxels per sq-deg
        w = int(np.sqrt(pperd*conv)) # Pixels over the whole sphere
        dist = np.logspace(0,1.5)
        r = np.zeros([len(dist),])
        for i in range(0,len(dist)):
            sppos, spdir, sport = mountxform(rt.vec3(0, 0, dist[i]), Z, X, 0., np.pi/2., 0., cf)
            scene[0] = rt.Sphere(sppos, spr, rgb(5, 0., 0.), mirror=0.02)

            ang = 2 * np.arcsin(spr / sppos.dist())  # angular diameter of the sphere.
            sphcov = (1 - np.cos(ang / 2)) / 2  # Fractional area: solid angle divided by 4pi

            O, D, xp, yp = camera(rt.vec3(0, 0, 0), spdir, sport, w, w, proj="sphere2")
            color = rt.raytrace(O, D, scene, hitmap=True)
            cd = color.dist()
            ind = (cd == 5.)
            r[i] = (np.size(np.where(ind))/(w**2))/sphcov-1


        plt.subplot(212)
        plt.semilogx(dist,r)
        plt.title("Camera at {0} px/sq-deg resolution".format(pperd))
        plt.xlabel("Distance to sphere (m)")
        plt.ylabel("Fractional residuals")
        plt.ylim([-0.01, 0.01])
        plt.grid()

    plt.subplots_adjust(hspace=1.2)
    #plt.show()
    # Check for spatial systematics
    if True:
        az = np.arange(0, 2 * np.pi, np.pi / 2)
        el = np.arange(0, np.pi / 2, np.pi / 20)
        dk = np.arange(0, 1)  # range(0,2*np.pi,np.pi/2)
        rat = np.zeros([len(az), len(el)])

        pperd = 7  # pipxels per sq-deg
        w = int(np.sqrt(pperd * conv))  # Pixels over the whole sphere
        ext = [min(az), max(az), min(el), max(el)]
        ext = [i * 180 / np.pi for i in ext]
        cf["eloffz"] = 1.0
        for i in range(0, len(az)):

            for j in range(0, len(el)):
                for k in range(0, len(dk)):
                    sppos, spdir, sport = mountxform(rt.vec3(0, 0, 0.0), Z, X, az[i], el[j], dk[k], cf)
                    scene[0] = rt.Sphere(sppos, spr, rgb(5, 0., 0.), mirror=0.02)

                    ang = 2 * np.arcsin(spr / sppos.dist())  # angular diameter of the sphere.
                    sphcov = (1 - np.cos(ang / 2)) / 2  # Fractional area: solid angle divided by 4pi
                    O, D, xp, yp = camera(rt.vec3(0, 0, 0), spdir, sport, w, w, proj="sphere2")
                    color = rt.raytrace(O, D, scene, hitmap=True)
                    cd = color.dist()
                    ind = (cd == 5.)
                    rat[i, j] = ((np.size(np.where(ind)) / (w ** 2)) / sphcov) - 1

        plt.figure(2)
        plt.subplot(111)
        plt.imshow(rat, extent=ext)
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Fractional Residuals', rotation=270)
        plt.title("Camera at {0} px/sq-deg resolution, sphere ".format(pperd))
        plt.ylabel("Elevation (o)")
        plt.xlabel("Azimuth (o)")
        plt.show()

        # Settling on 7 pixels per sq-deg. Systematics due to pixelization
        # are on the sub-percent level.

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
    parser.add_argument("--wincam",
                        help="Save a snapshot of the typical view from the window (normally off)",
                        default=False,
                        action="store_true")
    parser.add_argument("--winmap",
                        help="Show the fractional area subtended by the groundshield on the window",
                        default=False,
                        action="store_true")
    parser.add_argument("--showtemp",
                        help="Estimate temperatures ''seen'' by rays.",
                        default=False,
                        action="store_true")
    parser.add_argument("--config",
                        help="Configuration file to use for dimensions, etc..., default ''BA''",
                        default=def_dict,
                        )
    parser.add_argument("--az",
                        help="Set telescope Azimuth in degrees",
                        default=0.,
                        type=float)
    parser.add_argument("--el",
                        help="Set telescope Elevation in degrees.",
                        default=90.,
                        type=float)
    parser.add_argument("--dk",
                        help="Set telescope Deck in degrees.",
                        default=0.,
                        type=float)
    parser.add_argument("--test",
                        help="Runs various diagnostics under the do_test func. Hardcoded. Default off",
                        default=False,
                        action="store_true")
    parser.add_argument("--mirror",
                        help="Gives you a mirror also.",
                        default=False,
                        action = "store_true"
                        )
    return parser, parser.parse_args()


if __name__ == "__main__":
    def_dir = op.join(op.expanduser("."), "data")
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
        filename = op.join(args.dir, args.title)

    if op.isfile(filename) and not args.ow:
        run_num = 0
        print("File: {0} already exists!".format(args.title))
        while op.isfile(filename):
            run_num = run_num + 1
            filename = op.join(args.dir, args.title + "{0}.csv".format(run_num))
        print("Using file: " + filename)

    if args.test:
        cf = configDict["test"]
    else:
        cf = configDict[args.config]
    t0 = time.time()

    if not args.showtemp:
        clrs = configDict["colors"]
    else:
        clrs = configDict["tempcolors"]

    rgb = rt.vec3
    defpos = rt.vec3(0.,0.,0.)
    defdir = Z
    defort = X

    # Create Groundshield
    gspos = defpos
    gsdir = defdir
    gsort = X
    gsrad = cf["gsrad"] # in meters
    gsheight = cf["gsheight"] # in meters
    nsides = cf["gssides"]

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

    # Describe somewhere how we encode colors
    floor = rt.Disc(rt.ORG, rt.Z, rt.X, 1e3, rgb(clrs["gnd"][0],0,0),mirror=0.)
    #floor = rt.CheckeredSphere(rt.vec3(0, 0, -99999.5), Z, X, 99999, diffuse=rgb(.75, .75, .75), mirror=0.25),
    scene = [
        floor,
    ]

    # Add a wall that peaks just above the groundshield to approximate diffraction
    dowall = False
    if dowall:
        N = 5
        th = 2*np.pi/180
        mgs = max(gsrad)
        RL = (N+1)*mgs
        HL = N*mgs*np.tan(th)+sum(gsrad)
        wall = rt.Cylinder(gspos,gsdir,gsort,RL,HL,diffuse=rgb(clrs["gnd"][0]/mgs/2,0,0),mirror=0.)
        scene.append(wall)
    gsclr = clrs["gsh"]

    # Add a ground shield
    if np.size(gsrad) > 1:
        rt.multipanel_gs(scene,gspos, gsdir, gsort, gsrad, gsheight, nsides, clr=rgb(gsclr, 0., 0.), mirr=1.0)
    elif gsrad > 0:
        rt.make_gs(scene,gspos, gsdir,gsort, gsrad*0.3, gsrad, gsheight, nsides, clr=rgb(gsclr, 0., 0.), mirr=1.0)

    # Run a test or make forebaffles.
    if args.test:
        #do_test(cf)
        sppos = rt.vec3(0,0,2.0)
        spr = 1.0
        scene.append(rt.Sphere(sppos, Z,X,spr, rgb(gsclr, 0., 0.), mirror=0.5))
    elif cf["nbaffles"]>0:
        scene = make_forebaffles(scene, cf.copy(), az, el, dk, clrs, mirr=0.01)

    rt.BOUNCES=0
    rt.L.z = 10
    rt.L.y = 0

    if args.wincam:
        pperd = 7  # pipxels per sq-deg
        w = int(np.sqrt(pperd * conv))  # Pixels over the whole sphere
        cf2 = cf.copy()
        fbpos, fbdir, fbort = mountxform(ORG, Z, X, az, el, dk, cf2)
        fov = np.pi

        O, D, xp, yp = camera(fbpos, fbdir, fbort, w, w, proj="sphere2", fov=fov, PLOT=False)
        if args.showtemp == False:
            color = rt.raytrace(O, D, scene, O, trace="hit",clrstop=clrs["gnd"])
            show_hit_map(xp, yp, color.sum(),filename,pperd)
        else:
            color = rt.raytrace(O, D, scene, O, trace="temp")
            if np.size(clrs["gnd"])>1:
                scene[0].diffuse.x = scene[0].diffuse.x*0+clrs["gnd"][1]
                if dowall:
                    scene[1].diffuse.x = scene[0].diffuse.x * 0 + clrs["gnd"][1]/mgs/2
                color = color - rt.raytrace(O, D, scene, O, trace="temp")
                show_temp_map(xp, yp, color.sum(), filename, log=True)
            else:
                show_temp_map(xp, yp, color.sum(), filename)


    #window_map(scene,az,el,dk,cf.copy())
    if args.winmap:
        pperd = 3  # pipxels per sq-deg
        w = int(np.sqrt(pperd * conv))  # Pixels over the whole sphere
        winsamps = 10
        wr = cf["winrad"]
        R = cf["aptoffr"]
        k1 = k2 = np.arange(-wr*1.5,wr*1.5,wr/winsamps)
        K1, K2 = np.meshgrid(k1,k2)
        K1 = np.reshape(K1,(np.size(K1),1))
        K2 = np.reshape(K2, (np.size(K2),1))
        fov = np.pi
        r = np.sqrt(K1**2+(R-K2)**2)
        th = np.arctan2(K1,R-K2)
        gsfrac = np.zeros((len(r),1))

        for i in range(0,len(K1)):
            if np.sqrt(K1[i]**2+K2[i]**2) <= wr:
                cf2 = cf.copy()
                cf2["drumangle"] = cf2["drumangle"] + th[i]
                cf2["aptoffr"] = r[i]
                fbpos, fbdir, fbort = mountxform(ORG, Z, X, az, el, dk, cf2)
                O, D, xp, yp = camera(fbpos,fbdir,fbort, w, w, proj="sphere2", fov=fov)
                print(np.shape(fbpos.components()))
                color = rt.raytrace(O, D, scene, O, trace="temp")

                gsfrac[i] = get_hit_frac(color,clrs["fbmain"],pperd)


        gsfrac[np.sqrt(K1**2+K2**2) > wr] = np.nan
        gsfrac = np.ma.masked_where(np.isnan(gsfrac),gsfrac)

        c = np.reshape(gsfrac,(len(k1),len(k2)))
        #fig = plt.figure(figsize=(4, 4))
        fig = plt.figure()
        plt.subplot()
        plt.pcolormesh(k1,k2, c)#, vmin=0)#, vmax=0.0025)
        plt.xlim([-1 * wr*1.1, wr*1.1])
        plt.ylim([-1 * wr*1.1, wr*1.1])
        plt.xlabel("Window X (m)")
        plt.ylabel("Window Y (m)")
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('GS subtend fraction', rotation=270)
        plt.show()

    # Make a mirror
    if args.mirror:
        #mirrcf = configDict["keckfff2014"].copy()
        mirrcf = configDict["BAfff"].copy()
        scene = make_mirror(scene,cf,mirrcf,az,el)

    # Makes a map in x/y coordinates of the fractional hit of a target
    if False:#args.fpmap:
        dks = [166, 202, -122, -86, -50, -14, 22, 58, 94, 130]
        drumpos = range(1,11)
        dks = [45]
        drumpos = [1]
        mirroffs = np.arange(0,2,0.1)
        totcov = np.zeros(np.size(mirroffs))
        for k in range(0,len(drumpos)):
            print(drumpos[k],dks[k])
            #mirrcf = configDict["keckfff"].copy()
            #mirrcf["offset"] = mirroffs[k]
            #scene = scene[0:-2]
            #scene = make_mirror(scene,cf,mirrcf)
            # Make a grid of x/y values
            fpres = 1/2 # pix per deg
            fps = 12  # FP size in deg
            #print(np.arange(-fps, fps, 1 / fpres)+1/fpres/2)


            x = y = (np.arange(-fps,fps,1/fpres)+1/fpres/2)*np.pi/180
            X0, Y0 = np.meshgrid(x,y)
            X0 = np.reshape(X0,(np.size(X0),1))
            Y0 = np.reshape(Y0, (np.size(Y0),1))
            r = np.sqrt(X0**2+(Y0)**2)
            th = np.arctan2(X0,Y0)

            #print(r*180/np.pi,th*180/np.pi)
            hitfrac = np.zeros((len(r), 1))
            fbpos, fbdir, fbort = mountxform(ORG, Z, X, az, el, dks[k]*np.pi/180, cf)
            fbort2 = fbdir.cross(fbort)
            #scene.append(rt.Cylinder(fbpos, fbdir, fbort, 0.01, cap=10))
            for j in range(0,len(X0)):

                D = zyz(fbdir.copy(),-th[j],-r[j],th[j],fbort,fbort2,fbdir)
                winsamps = 4
                wr = cf["winrad"] # window radius
                R = cf["aptoffr"] # window center distance
                k1 = k2 = np.arange(-wr,wr,2*wr/winsamps)+wr/winsamps
                K1, K2 = np.meshgrid(k1,k2)
                K1 = np.reshape(K1,(np.size(K1),1))
                K2 = np.reshape(K2, (np.size(K2),1))
                #d = np.sqrt(K1**2+(R-K2)**2)
                #phi = np.arctan2(K1,R-K2)
                winhit = np.zeros((len(K1),1))

                for i in range(0,len(K1)):
                    if np.sqrt(K1[i]**2+K2[i]**2) <= wr:

                        Oray = fbpos+(fbort*K1[i])+(fbort2*K2[i])
                        color = rt.raytrace(Oray, D, scene, Oray, trace="hit")
                        if color.sum() == mirrcf["color"]:
                            winhit[i] = 1
                ind = (np.sqrt(K1 ** 2 + K2 ** 2) < wr)
                hitfrac[j] = np.nansum(winhit[ind]) / np.size(winhit[ind])

            if True:
                #hitfrac[np.sqrt(X0**2+Y0**2) > fps*np.pi/180] = np.nan
                #hitfrac = np.ma.masked_where(np.isnan(hitfrac),hitfrac)
                #hitfrac = 10*np.log10(hitfrac+1e-12)
                c = np.reshape(hitfrac,(len(x),len(y)))
                pixconv = 150
                plt.figure(1)
                plt.figure(figsize=(1000/pixconv, 834/pixconv))
                plt.subplot()
                plt.pcolormesh(-1*y*180/np.pi,-1*x*180/np.pi, c, vmin=0)#, vmax=1.0)
                plt.xlim([-1 * fps*1., fps*1.])
                plt.ylim([-1 * fps*1., fps*1.])
                plt.xlabel("y (deg)")
                plt.ylabel("x (deg)")
                cbar = plt.colorbar()
                cbar.ax.set_ylabel('Subtend fraction', rotation=270)

                dirname = "../postings/2019mmdd_mirror_check/figs/"
                fname = dirname+args.config+"raytrace_pos{0}_back".format(drumpos[k])
                plt.savefig(fname+".png", dpi=1*pixconv)
                plt.close()
                #plt.show()
            ind = (np.sqrt(X0**2+Y0**2) > fps*np.pi/180)
            #totcov[k] = np.sum(hitfrac[ind])/np.size(hitfrac[ind])

        if False:
            fig = plt.figure()
            plt.plot(mirroffs,totcov)
            plt.show()

    # Get Camera View
    if args.cam:
        cf["aptoffr"] = 0
        cf["aproffz"] = cf["aptoffz"]+mirrcf["height"]
        fbpos, fbdir, fbort = mountxform(ORG, Z, X, az, el, dk, cf)
        w, h = (800, 400)
        cpos = rt.vec3(8.5, 0*-8.5, 3.)
        #cdir = (ORG - cpos).norm()
        cdir = (fbpos - cpos).norm()
        rt.L = cpos + Z
        cort2 = cdir.cross(Z).norm()
        cort = cdir.cross(cort2).norm()
        O, D, xp, yp = camera(cpos, cdir, cort2, w, h, proj="flat", fov=np.pi / 2.2)
        color = rt.raytrace(O, D, scene, O)
        color = rt.tone_map(color)
        RGB = [Image.fromarray((255 * np.clip(c, 0, 1).reshape((h, w))).astype(np.uint8), "L") for c in
               color.components()]
        Image.merge("RGB", RGB).save(filename + "_camview.png")
        print("Saving to : " + filename + "_camview.png")


    print("Complete!")
