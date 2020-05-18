from PIL import Image
import numpy as np
import time
import numbers
from functools import reduce
#import scratch as sc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def extract(cond, x):
    if isinstance(x, numbers.Number):
        return x
    else:
        return np.extract(cond, x)

class vec3():
    def __init__(self, x, y, z):
        (self.x, self.y, self.z) = (x, y, z)
    def __mul__(self, other):
        return vec3(self.x * other, self.y * other, self.z * other)
    def __add__(self, other):
        return vec3(self.x + other.x, self.y + other.y, self.z + other.z)
    def __sub__(self, other):
        return vec3(self.x - other.x, self.y - other.y, self.z - other.z)
    def dot(self, other):
        return (self.x * other.x) + (self.y * other.y) + (self.z * other.z)
    def cross(self, other):
        x = self.y*other.z-self.z*other.y
        y = self.z*other.x-self.x*other.z
        z = self.x*other.y-self.y*other.x
        return vec3(x,y,z)
    def dist(self):
        return np.sqrt(self.dot(self))
    def __abs__(self):
        return self.dot(self)
    def norm(self):
        mag = np.sqrt(abs(self))
        return self * (1.0 / np.where(mag == 0, 1, mag))
    def components(self):
        return (self.x, self.y, self.z)
    def extract(self, cond):
        return vec3(extract(cond, self.x),
                    extract(cond, self.y),
                    extract(cond, self.z))
    def place(self, cond):
        r = vec3(np.zeros(cond.shape), np.zeros(cond.shape), np.zeros(cond.shape))
        np.place(r.x, cond, self.x)
        np.place(r.y, cond, self.y)
        np.place(r.z, cond, self.z)
        return r
    def rotate(self,k,th):
        # k is the rotation axis vector
        # th is the angle in radians
        # V cos(th) + (KxV) sin(th) + K (K.V)(1-cos(th))
        k.norm()
        return self*np.cos(th) + k.cross(self)*np.sin(th) + k*k.dot(self)*(1-np.cos(th))
    def copy(self):
        return vec3(self.x,self.y,self.z)
    def sum(self):
        return self.x+self.y+self.z

def make_array(v,N):
    ones = np.ones(np.shape(N))*N
    V = vec3(v.x*ones,v.y*ones,v.z*ones)
    return V

rgb = vec3

(w, h) = (800, 600)         # Screen size
L = vec3(5.0, -10.0, 5.0)        # Point light position
E = vec3(0., -1, 0.35)     # Eye position
FARAWAY = 1.0e39            # an implausibly huge distance
BOUNCES = 3                # How many bounces before we stop?
defAMB = rgb(0.01,0.01,0.01)
AMBIENT = defAMB*10         # %-total light

# Define coordinate system
X = vec3(1.,0.,0.)
Y = vec3(0.,1.,0.)
Z = vec3(0.,0.,1.)
ORG = vec3(0.,0.,0.) # Origin

# Working on converting raytrace function into an object. No sure if necessary
class Trace:
    def __init__(self,ORG, DIR, scene, RO, bounce = 0):
        self.scene = scene
        self.O = ORG
        self.D = DIR
        self.bounce = bounce

    def light(self,org):
        return color

    def hit(self):
        return color

    def temp(self):
        return color


def raytrace(O, D, scene, RO, bounce = 0, trace="light", clrstop=0.):
    # O is the ray origin, D is the normalized ray direction
    # scene is a list of Sphere objects (see below)
    # bounce is the number of the bounce, starting at zero for camera rays

    distances = [s.intersect(O, D) for s in scene]
    nearest = reduce(np.minimum, distances)
    color = rgb(0, 0, 0)

    for (s, d) in zip(scene, distances):
        hit = (nearest != FARAWAY) & (d == nearest)
        if np.any(hit):
            dc = extract(hit, d)
            Oc = O.extract(hit)
            Dc = D.extract(hit)
            cc = s.light(Oc, Dc, dc, scene, RO, bounce,tracetype=trace,clrstop=clrstop)
            color += cc.place(hit)

    return color




def tone_map(color):
    r,g,b = color.components()
    lum = 0.2126*r+0.715*g+0.0722*b
    return color * (1/(1+lum))

## Geometry Classes
# Argumments are in the form of
# pos, dir, ort, opt1, opt2, ..., optN, diffuse, mirror
##
class Shape:
    def __init__(self, center,dir,ort,diffuse=rgb(1, 1, 1), mirror = 1.0):
        self.c = center
        self.dir = dir.norm()
        self.ort = ort.norm()
        self.ort2 = dir.cross(ort).norm()
        self.diffuse = diffuse
        self.mirror = mirror

    def intersect(self, O, D):
        return D.dist()*FARAWAY

    def diffusecolor(self, M):
        return self.diffuse

    def getnorm(self,vorg,vdir,vdist):
        return vdir*-1


    def light(self, O, D, d, scene, RO, bounce, tracetype="light", clrstop=0.):
        M = (O + D * d)                         # intersection point
        N = self.getnorm(O,D,d)      # normal
        toL = (L - M).norm()                    # direction to light
        toO = (RO - M).norm()                    # direction to ray origin
        nudged = M + N * .000001                  # M nudged to avoid itself


        if tracetype=="light":
            # Shadow: find if the point is shadowed or not.
            # This amounts to finding out if M can see the light
            # Returns 3-vector RGB values from 0-1
            light_distances = [s.intersect(nudged, toL) for s in scene]
            light_nearest = reduce(np.minimum, light_distances)
            seelight = light_distances[scene.index(self)] == light_nearest

            # Ambient
            color = AMBIENT

            # Lambert shading (diffuse)
            lv = np.maximum(N.dot(toL), 0)
            color += self.diffusecolor(M) * lv * seelight

            # Reflection
            if bounce < BOUNCES:
                rayD = (D - N * 2 * D.dot(N)).norm()

                color += raytrace(nudged, rayD, scene, RO, bounce + 1) * self.mirror

            # Blinn-Phong shading (specular)
            phong = N.dot((toL + toO).norm())
            color += rgb(1, 1, 1) * np.power(np.clip(phong, 0, 1), 50) * seelight
            #return color
        elif tracetype=="hit":
            # Only return the value of the object that is hit. No diffusion or light effects
            # Returns scalar which is mapped to objects
            color = self.diffuse
            if (bounce<BOUNCES) & (color.dist() != clrstop):
                rayD = (D - N * 2 * D.dot(N)).norm()
                color = raytrace(nudged, rayD, scene, RO, bounce + 1, trace=tracetype,clrstop=clrstop)  # * self.mirror
            #return color.sum()
        elif tracetype=="temp":
            # Compute temperature T "seen" by the ray using mirror values as emissivity ei and colors as temps Ti
            # Approximate T = (1-e0)*T0+e0*((1-e1)*T1+e1*(...))
            # Returns scalar of temperature after N bounces.
            color = self.diffuse
            if (bounce < BOUNCES):
                rayD = (D - N * 2 * D.dot(N)).norm()
                color = color*(1-self.mirror) + (raytrace(nudged, rayD, scene, RO, bounce + 1,trace=tracetype) * self.mirror)

        return color



class Disc(Shape):
    def __init__(self, center,dir,ort,rad=1,diffuse=rgb(1, 1, 1), mirror = 1.0):
        super().__init__(center,dir,ort,diffuse,mirror)
        self.r = rad

    def intersect(self, O, D):
        Q = O-self.c
        h = -self.dir.dot(Q)/D.dot(self.dir)
        pred = (h>0) & ((Q+D*h).dist() < self.r)
        return np.where(pred, h, FARAWAY)

    def getnorm(self,vcen,vdir,vdist):
        return (self.dir*np.sign(self.dir.dot(vdir))*-1).norm()

class Cylinder(Shape):
    def __init__(self, center, dir, ort, rad=1, cap=1,diffuse=rgb(1, 1, 1), mirror=1.0):
        super().__init__(center, dir, ort, diffuse, mirror)
        self.r = rad
        self.capa = center
        self.capb = center+dir*cap

    def intersect(self, O, D):
        def capcheck(P, capa, capb):
            AB = capb-capa
            AP = P - capa
            Pprj = capa + (AB * (AB.dot(AP) / AB.dot(AB)))
            PA = (Pprj - capa).dist()
            PB = (capb - Pprj).dist()
            return np.where((PA+PB)<=(AB.dist()+0.000001),True,False)
        AB = self.capb-self.capa
        AO = O-self.capa
        AOXAB = AO.cross(AB) # Cyl X-dir
        VXAB = D.cross(AB) # Cyl Y-dir
        ab2 = AB.dot(AB)
        a = VXAB.dot(VXAB)
        b = 2*VXAB.dot(AOXAB)
        c = AOXAB.dot(AOXAB)-self.r*self.r*ab2
        disc = (b**2)-(4*a*c)
        disc = np.where(disc<0,0,disc)
        sq = np.sqrt(disc)
        h0 = (-b - sq) / (2 * a)
        h1 = (-b + sq) / (2 * a)

        # Cylinder Caps
        # For two hits: If the first is within caps, return first.
        # If not, check second. If neither, miss.
        I0 = capcheck(O + D * h0, self.capa, self.capb)
        I1 = capcheck(O + D * h1, self.capa, self.capb)

        h0 = np.where((disc > 0) & (h0 > 0) & I0, h0, FARAWAY)
        h1 = np.where((disc > 0) & (h1 > 0) & I1, h1, FARAWAY)

        pred = h0 < h1

        return np.where(pred, h0, h1)

    def getnorm(self,vorg, vdir, vdist):
        P = vorg + (vdir * vdist)
        V1 = (self.dir.cross(P - self.c).norm()).norm()
        N = (V1.cross(self.dir)).norm()
        return N * np.sign(N.dot(vdir)) * -1

class Sphere(Shape):
    def __init__(self, center,dir,ort,rad=1,diffuse=rgb(1, 1, 1), mirror = 1.0):
        super().__init__(center,dir,ort,diffuse,mirror)
        self.r = rad

    def intersect(self, O, D):
        b = 2 * D.dot(O - self.c)
        c = abs(self.c) + abs(O) - 2 * self.c.dot(O) - (self.r * self.r)
        disc = (b ** 2) - (4 * c)
        sq = np.sqrt(np.maximum(0, disc))
        h0 = (-b - sq) / 2
        h1 = (-b + sq) / 2
        h = np.where((h0 > 0) & (h0 < h1), h0, h1)
        pred = (disc > 0) & (h > 0)
        return np.where(pred, h, FARAWAY)

    def getnorm(self,vcen,vdir,vdist):
        return ((vcen+vdir*vdist) - self.c) * (1. / self.r)

class CheckeredSphere(Sphere):
    def diffusecolor(self, M):
        checker = ((M.x * 2).astype(int) % 2) == ((M.y * 2).astype(int) % 2)
        return self.diffuse * checker

class Quad(Shape):
    #
    def __init__(self, center,dir,ort,sides = np.array([[-1,1],[1,1],[1,-1],[-1,-1]]),diffuse=rgb(1, 1, 1), mirror = 1.0):
        super().__init__(center,dir,ort,diffuse,mirror)
        self.s = sides + np.random.random(np.shape(sides))*1e-10 # jitter by an extremely small number
        self.ort2 = ort.cross(dir).norm()

    def get_slopes(self):
        x = 0
        y = 1
        m0 = (self.s[0, y] - self.s[1, y]) / (self.s[0, x] - self.s[1, x])
        m1 = (self.s[1, y] - self.s[2, y]) / (self.s[1, x] - self.s[2, x])
        m2 = (self.s[3, y] - self.s[2, y]) / (self.s[3, x] - self.s[2, x])
        m3 = (self.s[0, y] - self.s[3, y]) / (self.s[0, x] - self.s[3, x])

        b0 = self.s[0, y] - m0 * self.s[0, x]
        b1 = self.s[1, y] - m1 * self.s[1, x]
        b2 = self.s[3, y] - m2 * self.s[3, x]
        b3 = self.s[0, y] - m3 * self.s[0, x]

        return (m0,m1,m2,m3,b0,b1,b2,b3)

    def intersect(self, O, D):

        Q = O-self.c
        h = -self.dir.dot(Q)/D.dot(self.dir)

        # Define slopes of each side
        (m0,m1,m2,m3,b0,b1,b2,b3) = self.get_slopes()
        Py = self.ort.dot(Q+D*h)
        Px = self.ort2.dot(Q+D*h)
        pred = (h>0) & (Py < m0*Px+b0) & (Px < (Py-b1)/m1) & (Py > m2*Px+b2) & (Px > (Py-b3)/m3)

        return np.where(pred, h, FARAWAY)

    def getnorm(self,vcen,vdir,vdist):
        return (self.dir*np.sign(self.dir.dot(vdir))*-1).norm()

def make_gs(scene, pos,dir,ort,LR, UR, ht, nsides, clr=rgb(1.0,1.0,1.0), mirr=1.0):
    # pos   : position of bottom of the groundshield
    # dir   : direction
    # ort   : direction of primary vertex
    # LR    :lower radius from origin to vertex
    # UR    : Upper radius from orgin to vertex
    # nsides: # of sides of the shield

    ort2 = ort.cross(dir).norm()
    flare = np.arctan((UR-LR)/ht) # Flare angle of the panels
    ang0 = 2 * np.pi / nsides
    gsrad = (UR+LR)/2 # Distance from central axis to center of panel
    lside = LR*np.tan(ang0/2) # Length of bottom panel side
    uside = UR*np.tan(ang0/2) # Length of top panel side
    pdist = ht/2/np.cos(flare)
    verts = np.array([[-uside,pdist],[uside,pdist],[lside,-pdist],[-lside,-pdist]])
    gscen = dir*(ht/2)
    ipos0 = gscen+ort*gsrad
    idir0 = (ort*-1).rotate(ort2,-flare).norm()
    iort0 = dir.rotate(ort2,-flare).norm()

    ang = 0
    for i in range(0,nsides):
        ipos = ipos0.rotate(dir,ang)+pos
        iort = iort0.rotate(dir,ang).norm()
        idir = idir0.rotate(dir,ang).norm()

        scene.append(Quad(ipos, idir, iort, verts, diffuse=clr,mirror=mirr))
        ang += ang0

    return scene

def multipanel_gs(scene, pos,dir,ort,radlist, htlist, nsides, clr=rgb(1.0,1.0,1.0), mirr=1.0):
    # radlist : a list with the bottom radius of the bottom panel and then the radii of the top panels
    # htlist : A list of all of the heights of the radii (presumably starting with zero).

    scene.append(Disc(pos,dir,ort,rad=radlist[0]/np.cos(2*np.pi/nsides),diffuse=clr,mirror=mirr))

    for i in range(0,len(radlist)-1):
        LR = radlist[i]
        UR = radlist[i + 1]
        off = htlist[i]
        ht = htlist[i + 1] - off
        scene = make_gs(scene, pos+dir*off, dir, ort, LR, UR, ht, nsides, clr=clr, mirr=mirr)

    return scene

if __name__ == "__main__":
    pos0 = vec3(0,3.,0)
    verts = np.array([[-1,1],[1,1],[1,-1],[-1,-1]])
    # scene = [
    #     Quad(pos0,Z,Y,verts,diffuse=rgb(1.,1.,1.),mirror=0.3),
    #     Quad(pos0+Z*2.5+Y*0.5, Y, Z, verts, diffuse=rgb(1, 1, 1), mirror=0.3),
    #     Quad(pos0+Z*5,Z,Y, verts, diffuse=rgb(1, 1, 1), mirror=0.3),
    #     Quad(pos0+Z*2.5+X*0.5, X, Z, verts, diffuse=rgb(1, 0, 0), mirror=0.3),
    #     Quad(pos0+Z*2.5-X*0.5, X, Z, verts, diffuse=rgb(0, 1, 0), mirror=0.3),
    #     ]
    odir = Z.copy()
    oort = Y.copy().rotate(Z,np.pi/4)
    scene = [
        Quad(pos0,odir.rotate(X,np.pi/2),oort.rotate(X,np.pi/2),verts)
    ]

    RL = [2.79,5.05,7.34]
    HT = [0,0.94,3.97]

    #scene = make_gs(scene,opos,odir,X,0.1,0.3,0.3,8, clr=rgb(0.1,
    # 0.5,0.1),mirr=0.75)
    #scene = multipanel_gs(scene, opos, odir, oort, RL,HT, 8, clr=rgb(0.1, 0.5, 0.1), mirr=0.1*1.0)

    r = float(w) / h
    # Screen coordinates: x0, y0, x1, y1.
    S = (-1., 1. / r + .25, 1., -1. / r + .25)
    x = np.tile(np.linspace(S[0], S[2], w), h)
    y = np.repeat(np.linspace(S[1], S[3], h), w)

    t0 = time.time()
    Q = vec3(x, 0, y)
    color = raytrace(E, (Q - E).norm(), scene, E)
    print("Took {0}".format(time.time() - t0))

    #color = tone_map(color)
    rgb = [Image.fromarray((255 * np.clip(c, 0, 1).reshape((h, w))).astype(np.uint8), "L") for c in color.components()]
    Image.merge("RGB", rgb).save("fig.png")

