import numpy as np
import geometry as rt
import gs_analysis as gsa
from config import configDict
import matplotlib.pyplot as plt
import os.path as op
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
    parser.add_argument("--mirrconfig",
                        help="Configuration file to use for dimensions, etc..., default ''BA''",
                        default=def_mirrdict,
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
    parser.add_argument("--rays",
                        help="Traces rays from the window for troubleshooting purposes",
                        default=False,
                        action = "store_true"
                        )
    return parser, parser.parse_args()


if __name__ == "__main__":
    def_dir = op.join(op.expanduser("."), "data")
    def_filename = "raytrace"
    def_dict = "BA"
    def_mirrdict = "BAfff"
    parser, args = get_args()

    cf = configDict[args.config].copy()
    mirrcf = configDict[args.mirrconfig].copy()

    # Make a scene
    scene = []
    scene = gsa.make_mirror(scene,cf,mirrcf,az,el)

    # For each pixel on the focal plane shoot a bundle of rays
    # Each bundle of rays is in a grid

    # Makes a map in x/y coordinates of the fractional hit of a target
    if True  :  # args.fpmap:
        #dks = [166, 202, -122, -86, -50, -14, 22, 58, 94, 130]
        #drumpos = range(1 ,11)
        offs = 45
        dks = [0, 90, 180, 270]
        dks = [ii+offs for ii in dks]
        drumpos = range(0,4)

        for m in range(0 ,len(drumpos)):
            mirroffs = np.arange(0, 2, 0.1)
            #mirroffs = [0.8]
            totcov = np.zeros(np.size(mirroffs))
            for k in range(0, len(mirroffs)):
                # mirrcf = configDict["keckfff"].copy()
                mirrcf["offset"] = mirroffs[k]
                scene = []
                scene = gsa.make_mirror(scene,cf,mirrcf,az,el)

                # Make a grid of x/y values
                fpres = 2  # pix per deg
                fps = 15  # FP size in deg

                x = y = (np.arange(-fps, fps, 1 / fpres) + 1 / fpres / 2) * np.pi / 180
                X0, Y0 = np.meshgrid(x, y)
                X0 = np.reshape(X0, (np.size(X0), 1))
                Y0 = np.reshape(Y0, (np.size(Y0), 1))
                r = np.sqrt(X0 ** 2 + (Y0) ** 2)
                th = np.arctan2(X0, Y0)

                hitfrac = np.zeros((len(r), 1))
                fbpos, fbdir, fbort = gsa.mountxform(ORG, Z, X, az, el, dks[m] * np.pi / 180, cf)
                fbort2 = fbdir.cross(fbort)
                # scene.append(rt.Cylinder(fbpos, fbdir, fbort, 0.01, cap=10))
                for j in range(0, len(X0)):
                    D = gsa.zyz(fbdir.copy(), -th[j], -r[j], th[j], fbort, fbort2, fbdir)
                    winsamps = 4
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

                    color = rt.raytrace(Oray, Dray, scene, Oray, trace="hit")
                    ind = color.sum()==mirrcf["color"]*np.ones(np.shape(color.sum()))
                    winhit[ind] = 1
                    ind = (np.sqrt(K1 ** 2 + K2 ** 2) < wr)
                    hitfrac[j] = np.nansum(winhit[ind]) / np.size(winhit[ind])

                if False:
                    # hitfrac[np.sqrt(X0**2+Y0**2) > fps*np.pi/180] = np.nan
                    # hitfrac = np.ma.masked_where(np.isnan(hitfrac),hitfrac)
                    # hitfrac = 10*np.log10(hitfrac+1e-12)
                    print(np.max(hitfrac))
                    c = np.reshape(hitfrac, (len(x), len(y)))
                    pixconv = 150
                    plt.figure(1,figsize=(1000 / pixconv, 834 / pixconv))
                    plt.subplot()
                    plt.pcolormesh(-1 * y * 180 / np.pi, -1 * x * 180 / np.pi, c, vmin=0)  # , vmax=1.0)
                    plt.xlim([-1 * fps * 1., fps * 1.])
                    plt.ylim([-1 * fps * 1., fps * 1.])
                    plt.xlabel("y (deg)")
                    plt.ylabel("x (deg)")
                    cbar = plt.colorbar()
                    cbar.ax.set_ylabel('Subtend fraction', rotation=270)

                    dirname = "../postings/2019mmdd_mirror_check/figs/"
                    fname = dirname + args.config + "raytrace_pos{0}_back".format(drumpos[m])
                    plt.savefig(fname + ".png", dpi=1 * pixconv)
                    plt.show()
                    plt.close()

                ind = (np.sqrt(X0 ** 2 + Y0 ** 2) < fps * np.pi / 180)

                totcov[k] = np.sum(hitfrac[(ind&(hitfrac == 1.0))])/np.size(hitfrac[ind])

            if True:
                fig = plt.figure(1)
                plt.plot(mirroffs, totcov)
                plt.ylim((0,1.1))

        #plt.plot([0.8, 0.8],[-1,2],'k--')
        plt.plot([0.66, 0.66],[-1,2],'k-.')
        plt.ylim((0, 1.1))
        plt.grid()
        plt.legend(["RX0","RX1","RX2","RX3","Pos 1","Pos 2"])
        plt.title("RX0 offset: {}".format(offs))
        plt.ylabel("Frac. of focal plane w/ total mirror coupling")
        plt.xlabel("Mirror distance from Theta axis (m)")
        plt.show()

