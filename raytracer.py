from PIL import Image
import numpy as np
import time
import numbers
from functools import reduce

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

rgb = vec3

(w, h) = (800, 600)         # Screen size
L = vec3(-5.0, 0.0, 20.0)        # Point light position
E = vec3(0., -1, 0.35)     # Eye position
FARAWAY = 1.0e39            # an implausibly huge distance
BOUNCES = 3                # How many bounces before we stop?
defAMB = rgb(0.01,0.01,0.01)
AMBIENT = defAMB*10         # %-total light

def raytrace(O, D, scene, bounce = 0, hitmap=False):
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
            if not hitmap:
                Oc = O.extract(hit)
                Dc = D.extract(hit)
                cc = s.light(Oc, Dc, dc, scene, bounce)
                color += cc.place(hit)
            else:
                color += s.diffuse.place(hit)
    return color

class Cylinder:
    def __init__(self, center, r, dir, cap=1.0, diffuse=rgb(1, 1, 1), mirror = 0.5):

        self.c = center
        self.r = r
        self.dir = dir.norm()
        self.cap = cap
        self.diffuse = diffuse
        self.mirror = mirror
        self.capa = self.c
        self.capb = self.c + self.dir*self.cap


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

    def diffusecolor(self, M):
        return self.diffuse

    def light(self, O, D, d, scene, bounce):
        M = (O + D * d)                         # intersection point
        N = (M - self.c) * (1. / self.r)        # normal
        toL = (L - M).norm()                    # direction to light
        toO = (E - M).norm()                    # direction to ray origin
        nudged = M + N * .0001                  # M nudged to avoid itself

        # Shadow: find if the point is shadowed or not.
        # This amounts to finding out if M can see the light
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
            color += raytrace(nudged, rayD, scene, bounce + 1) * self.mirror

        # Blinn-Phong shading (specular)
        phong = N.dot((toL + toO).norm())
        color += rgb(1, 1, 1) * np.power(np.clip(phong, 0, 1), 50) * seelight
        return color



class Sphere:
    def __init__(self, center, r, diffuse=rgb(1, 1, 1), mirror = 0.5):
        self.c = center
        self.r = r
        self.diffuse = diffuse
        self.mirror = mirror

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

    def diffusecolor(self, M):
        return self.diffuse

    def light(self, O, D, d, scene, bounce):
        M = (O + D * d)                         # intersection point
        N = (M - self.c) * (1. / self.r)        # normal
        toL = (L - M).norm()                    # direction to light
        toO = (E - M).norm()                    # direction to ray origin
        nudged = M + N * .0001                  # M nudged to avoid itself

        # Shadow: find if the point is shadowed or not.
        # This amounts to finding out if M can see the light
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
            color += raytrace(nudged, rayD, scene, bounce + 1) * self.mirror

        # Blinn-Phong shading (specular)
        phong = N.dot((toL + toO).norm())
        color += rgb(1, 1, 1) * np.power(np.clip(phong, 0, 1), 50) * seelight
        return color

class CheckeredSphere(Sphere):
    def diffusecolor(self, M):
        checker = ((M.x * 2).astype(int) % 2) == ((M.y * 2).astype(int) % 2)
        return self.diffuse * checker

if __name__ == "__main__":

    #Swapped for cylinder
    #dummyvar = Cylinder(vec3(-2.75, .1, 3.5), 0.6, vec3(0.0,1.0,0.0),diffuse=rgb(1., .572, .184))
    #Sphere(vec3(-2.75, .1, 3.5), .6, rgb(1., .572, .184)),
    scene = [
        #Sphere(vec3(.75, .1, 1.), 0.6, rgb(0, 0, 1)),
        Sphere(vec3(-.75, 2.25, 0.1), .6, rgb(.5, .223, .5)),
        Cylinder(vec3(-2.75,3.5,0.0), 0.5, vec3(0.0,1.0,1.0),diffuse=rgb(1., .0, .0)),
        Cylinder(vec3(.75,1.0,-0.5), 5, vec3(0.0,0.0,1.0),cap=2.0, diffuse=rgb(0.0, .0, 1.0)),
        CheckeredSphere(vec3(0,0, -99999.5), 99999, rgb(.75, .75, .75), 0.25),
        ]

    r = float(w) / h
    # Screen coordinates: x0, y0, x1, y1.
    S = (-1., 1. / r + .25, 1., -1. / r + .25)
    x = np.tile(np.linspace(S[0], S[2], w), h)
    y = np.repeat(np.linspace(S[1], S[3], h), w)

    t0 = time.time()
    Q = vec3(x, 0, y)
    color = raytrace(E, (Q - E).norm(), scene,hitmap=False)
    print("Took {0}".format(time.time() - t0))


    rgb = [Image.fromarray((255 * np.clip(c, 0, 1).reshape((h, w))).astype(np.uint8), "L") for c in color.components()]
    Image.merge("RGB", rgb).save("fig.png")
