import numpy as np
import geometry as rt
import gs_analysis as gsa
from config import configDict
import matplotlib.pyplot as plt
import argparse

ORG = rt.ORG
X = rt.X
Y = rt.Y
Z = rt.Z
az = 0
el = 0
dk = 0

rt.BOUNCES = 0

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--config",
                        help="Configuration file to use for dimensions, etc..., default ''BA''",
                        default=def_dict,
                        )
    parser.add_argument("--mirrconfig",
                        help="Configuration file to use for dimensions, etc..., default ''BAfff''",
                        default=def_mirrdict,
                        )
    parser.add_argument("--mirroffs",
                        help="Set mirror position in meters from the Theta (or DK) axis.",
                        default=0.,
                        type=float)
    parser.add_argument("--mirrtilt",
                        help="Set mirror tilt in degrees.",
                        default=45.,
                        type=float)
    parser.add_argument("--savedir",
                        help="Directory to save figures",
                        default='./')
    return parser, parser.parse_args()


if __name__ == "__main__":
    # Initialize some variables
    def_dict = "BA"
    def_mirrdict = "BAfff"
    parser, args = get_args()
    dirname = args.savedir

    cf = configDict[args.config].copy()
    mirrcf = configDict[args.mirrconfig].copy()
    mirrcf["tilt"] = args.mirrtilt
    # Make a scene
    scene = []
    scene = gsa.make_mirror(scene,cf,mirrcf,az,el)

    # We want to plot the fraction of the beam coupled to the mirror
    # as a function of position on the focal plane given:
    # Mirror location and Dk angle.
    # RX no. doesn't matter because the answer is the same, just shifted
    # in increments of 90 degrees.

    # We also want to make a similar plot that gives us the number of unique hits for each x/y pixel

    # Makes a map in x/y coordinates of the fractional hit of a target

    dks = range(0,360,45)
    offs = 0
    dks = [ii+offs for ii in dks]
    drumpos = range(0,len(dks))
    moffs = np.arange(0,2,0.1)
    for imoffs in range(0,len(moffs)):
        args.mirroffs = moffs[imoffs]
        for m in range(0 ,len(drumpos)):
            mirroffs = [args.mirroffs]

            for k in range(0, 1):
                mirrcf["offset"] = mirroffs[k]
                scene = []
                scene = gsa.make_mirror(scene,cf,mirrcf,az,el)

                # Make a grid of x/y values
                fpres = 2  # pix per deg
                fps = 15+8  # FP size in deg

                x = y = (np.arange(-fps, fps, 1 / fpres) + 1 / fpres / 2) * np.pi / 180
                X0, Y0 = np.meshgrid(x, y)
                X0 = np.reshape(X0, (np.size(X0), 1))
                Y0 = np.reshape(Y0, (np.size(Y0), 1))
                r = np.sqrt(X0 ** 2 + (Y0) ** 2)
                th = np.arctan2(X0, Y0)

                hitfrac = np.zeros((len(r), 1))
                hf_flat = np.zeros((len(r), 1))
                if m==0:
                    uniq_hit = np.zeros(np.size(mirroffs))

                fbpos, fbdir, fbort = gsa.mountxform(ORG, Z, X, az, el, dks[m] * np.pi / 180, cf)
                fbort2 = fbdir.cross(fbort)

                # Make a grid on the window. This will be the origin of the rays.
                for j in range(0, len(X0)):
                    D = gsa.zyz(fbdir.copy(), -th[j], -r[j], th[j], fbort, fbort2, fbdir)
                    winsamps = 8
                    wr = cf["winrad"]  # window radius
                    R = cf["aptoffr"]  # window center distance
                    k1 = k2 = np.arange(-wr, wr, 2 * wr / winsamps) + wr / winsamps
                    K1, K2 = np.meshgrid(k1, k2)
                    K1 = np.reshape(K1, (np.size(K1), 1))
                    K2 = np.reshape(K2, (np.size(K2), 1))
                    winhit = np.zeros((len(K1), 1))
                    V3 = rt.make_array(fbpos,np.ones(np.shape(K1)))
                    V1 = rt.make_array(fbort,K1)
                    V2 = rt.make_array(fbort2,K2)
                    Oray = V3 + V1 + V2
                    Dray = rt.make_array(D,np.ones(np.shape(K1)))

                    # Do a raytrace, return the color ID of the first this we hit.
                    color = rt.raytrace(Oray, Dray, scene, Oray, trace="hit")
                    ind = color.sum()==mirrcf["color"]*np.ones(np.shape(color.sum()))
                    winhit[ind] = 1
                    ind = (np.sqrt(K1 ** 2 + K2 ** 2) < wr)
                    hitfrac[j] = np.nansum(winhit[ind]) / np.size(winhit[ind])

                # Plot the fractional coupling with the mirror.
                if False:
                    c = np.reshape(hitfrac, (len(x), len(y)))
                    pixconv = 150
                    plt.figure(1,figsize=(1000 / pixconv, 834 / pixconv))
                    plt.subplot()
                    plt.pcolormesh(-1 * y * 180 / np.pi, -1 * x * 180 / np.pi, c, vmin=0 , vmax=1.0)
                    plt.xlim([-1 * fps * 1., fps * 1.])
                    plt.ylim([-1 * fps * 1., fps * 1.])
                    plt.xlabel("y (deg)")
                    plt.ylabel("x (deg)")
                    cbar = plt.colorbar()
                    cbar.ax.set_ylabel('Subtend fraction', rotation=270)


                    fname = dirname + args.config + "raytrace_pos{0}_{1}".format(drumpos[m],args.mirroffs)
                    print("saving: "+fname)
                    plt.savefig(fname + ".png", dpi=1 * pixconv)
                    #plt.show()
                    plt.close()

                ind = (np.sqrt(X0 ** 2 + Y0 ** 2) < fps * np.pi / 180)
                hf_flat[((hitfrac==1.0))]=1
                uniq_hit = uniq_hit + hf_flat

        # Plot unique hits
        if True:
            c = np.reshape(uniq_hit, (len(x), len(y)))
            pixconv = 150
            plt.figure(1,figsize=(1000 / pixconv, 834 / pixconv))
            plt.subplot()
            plt.pcolormesh(-1 * y * 180 / np.pi, -1 * x * 180 / np.pi, c, vmin=0, vmax=len(drumpos))
            plt.xlim([-1 * fps * 1., fps * 1.])
            plt.ylim([-1 * fps * 1., fps * 1.])
            plt.xlabel("y (deg)")
            plt.ylabel("x (deg)")
            cbar = plt.colorbar()
            cbar.ax.set_ylabel('Unique positions', rotation=270)
            plt.title("Mirror pos: {} m".format(args.mirroffs))
            fname = dirname + "unique_hits_off_{0}_tilt_{1}".format(imoffs,int(args.mirrtilt))
            print("saving: " + fname)
            plt.savefig(fname + ".png", dpi=1 * pixconv)
            #plt.show()
            plt.close()

