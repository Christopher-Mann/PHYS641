from __future__ import division
import numpy as np
import healpy as hp
from matplotlib import pyplot as plt

F = False
T = True

lmax = 2500

# choose whether or not to re-run intensive things
redo = F

""" 
    <><><><><><><><><><><><<><><><><> 
    <><><><><>  PROBLEM 1  <><><><><> 
    <><><><><><><><><><><><<><><><><> 
"""

print "\n\n Problem 1\n\n"

data = np.genfromtxt("example_ps.txt")

chop = np.arange(lmax)
#chop = np.arange(len(data))

TT = data[chop,0] # = l(l+1)C_l / (2*pi)
EE = data[chop,1]
TE = data[chop,2]
BB = data[chop,3]

l = np.arange(len(TT))

Cl = TT*2*np.pi/( l*(l+1) )
Cl[0]=0


plt.figure(num=1)
plt.clf()

plt.subplot(2,1,1)
plt.title("Example Power Spectrum")
plt.plot(TT,'b-',lw=0.5)
plt.xlim(0,lmax)
plt.ylabel(r"$\ell (\ell+1)C_\ell/2\pi$",fontsize=15)

plt.subplot(2,1,2)

plt.semilogy(Cl,'b-',lw=0.5)
plt.xlim(0,lmax)
plt.ylabel(r"$C_\ell$",fontsize=15)
plt.xlabel(r"$\ell$",fontsize=18)
plt.savefig("figures/Fig1_prob1_input_PS.png")





"""
part A)



The variance should be:
    
    <T^2> = < (SUM alm Ylm)^2 >   -- SUM is over l,m
          = < SUM alm^2 Ylm^2 >   -- because cross terms have zero average
          = < SUM Cl Ylm^2 >
          = 1/4pi SUM Cl
          = 1/4pi SUM_l Cl*(2l+1) -- SUM is now over just l
"""

sum_term = 0
for l in range(lmax):
    sum_term += Cl[l]*(2*l+1)

var = 1./4./np.pi * sum_term

print "Expected map variance: %.0f"%var

"""
This produces a variance of: 12067
"""




"""
part B)
"""

# when chopping beyond l=2500, using synalm:
# 1st set of alm from synalm: 0-2500      : len = 2500
# 5th set of alm from synalm: 9993-12489  : len = 2496


# order is (l,m):  (e.g. lmax=4)
# (0,0), (1,0), (2,0), (3,0), (4,0), 
#        (1,1), (2,1), (3,1), (4,1), 
#               (2,2), (3,2), (4,2), 
#                      (3,3), (4,3), 
#                             (4,4)



myalm_list=[]

# this will generate a list of Cl values in the same indexing scheme
# that alm2map wants input alm values
#  i.e. 0 to lmax for m=0, then 1 to lmax for m=1, then 2 to lmax for m=2... etc.
Cl_in_ind_order = []
for i in range(len(Cl)):
    Cl_in_ind_order.append(Cl[i:])
Cl_in_ind_order=np.concatenate(Cl_in_ind_order)

# for each index, we draw a random number (complex) using the Cl values
myalms = np.zeros(len(Cl_in_ind_order),dtype='complex')
myalms.real=np.random.normal(scale=np.sqrt(Cl_in_ind_order/2))
myalms.imag=np.random.normal(scale=np.sqrt(Cl_in_ind_order/2))


if redo:
    mymap = hp.alm2map(myalms,1024)
mymapvar = np.std(mymap)**2

print "Measured map variance: %.0f"%mymapvar

# look at a map produced from my generated alm
plt.figure(num=2)
plt.clf()
hp.mollview(map=mymap,fig=2,max=500,min=-500,title="Using my own alm")
plt.savefig("figures/Fig2_prob1_my_generated_alm_map.png")


"""
The variance of my map is: 11987

This is very close to my expected value (12067)

"""



"""
Part C)
"""
if redo:
    print "    Calculating PS from my map:"
    myPS = hp.anafast(mymap)
    print "    done"

"""
Yes, they do agree (Fig 3).  
However, my generated PS seems to go beyond
lmax, dropping steeply befor leveling off at 1e-8.  
Not sure what's happening there
"""



plt.figure(num=3)
plt.clf()

plt.title("Example Power Spectrum")
plt.semilogy(myPS,'r-',lw=0.5,label="My generated PS")
plt.semilogy(Cl, 'b-',lw=0.5,label="Original PS")
plt.xlim(0,5000)
plt.legend()
plt.xlim(0,3000);plt.ylim(1e-9,2e3)
plt.ylabel(r"$C_\ell$",fontsize=15)
plt.xlabel(r"$\ell$",fontsize=15)
plt.savefig("figures/Fig3_prob1_my_generated_PS_vs_input.png")








"""
Part D)
"""

# 1) Generate alm from Cl using synalm
# 2) Create a map from those alm using alm2map
# 3) Generate a power spectrum from this map using anafast
if redo:
    print "    Calculating alm from Cl"
    alm = hp.synalm(Cl)
    
    print "    Calculating map from alm"
    map = hp.alm2map(alm,1024)
    
    print "    Regenerating Cls from map"
    PS = hp.anafast(map)
    print "    done"


# This compares the alm I generated myself against those from healpy.synalm
plt.figure(num=4)
plt.clf()
plt.subplot(2,2,1)
plt.plot(myalms,'r-',lw=0.5);plt.ylabel("My a_lm");plt.title("Zoomed out");plt.ylim(-50,50)
plt.subplot(2,2,2)
plt.plot(myalms,'r-',lw=0.5);plt.xlim(-1000,25000);plt.title("Zoomed in");plt.ylim(-50,50)
plt.subplot(2,2,3)
plt.plot(alm,'b-',lw=0.5);plt.ylabel("healpy a_lm");plt.ylim(-50,50)
plt.subplot(2,2,4)
plt.plot(alm,'b-',lw=0.5);plt.xlim(-1000,25000);plt.ylim(-50,50)
plt.savefig("figures/Fig4_prob1_my_generated_alm_vs_healpy_alm.png")


mapvar = np.std(map)**2
print "Map variance using healpy functions: %.0f"%mapvar


plt.figure(num=5)
plt.clf()
hp.mollview(map=map,fig=5,max=500,min=-500,title="Using healpy.synalm for alm")
plt.savefig("figures/Fig5_prob1_healpy_generated_alm_map.png")

"""
Figures 2, 4, and 5 compare the results between generating my own alm values
and letting healpy do it for me.

Figure 4 shows a direct comparison of the alm values.  They are qualitatively
similar, just with different random draws (but with the same drawing probabilities).

Figures 2 and 5 show maps generated by my own alm values and those from healpy.
Again, different random draws gives a different realization, but the variance is
the same in both cases.

"""











""" 
    <><><><><><><><><><><><<><><><><> 
    <><><><><>  PROBLEM 2  <><><><><> 
    <><><><><><><><><><><><<><><><><> 
"""




print "\n\n Problem 2\n\n"

"""
-------------------
Part A)


    
A 20 degree side is 1/18 or 0.055 of the full 360 degrees.

Yll has l periods across the equator (lecture 10  slide 10)

1 period across 20 degree box is l = 2pi/x where x is the box size in radians
Alternatively, l = 360/20 = 18 for the longest possible period

This corresponds to k = 1, therefore l = 18k or k = l/18

We can't look at l lower than 18 in this field 
(unless you can resolve half wavelenthgs of features in which case you might
be able to get some information down to l=9)





-------------------
Part B)

NOTATION:  SUM_x^y is a sum from x to y

N_alm = SUM_18^lmax ( 2l + 1 )

... but I imagine you might have to count up in l by 18's to correspond to 
    whole number k values...

For the number of k-modes interior to kmax, we would need to look at
what pairs of (kx,ky) integer values fall within circle of radius kmax
eg. for kmax<1 (0,0) only
    for kmax=1 (0,0) (0,1) (1,0) (0,-1) (-1,0)
    for kmax=2 (0,0) 
               (0,1) (1,0) (0,-1) (-1,0)       magnitude 1
               (1,1) (1,-1) (-1,1) (-1,-1)     magnitude sqrt(2)
               (0,2) (2,0) (0,-2) (-2,0)       magnitude 2
    etc.
    
I'm not sure how to count these as a function of kmax in closed form...
... but it will approach pi*kmax^2 as kmax gets large
  (My reasoning here is N=number of vertices on grid inside kmax circle,
  and pi*kmax^2 is area of circle.  Could imagine drawing dots on each vertex
  then shifting them 0.5 vertiall and horizontall, now they're in the centre
  of each grid square.  There are essentially the same number of dots here
  as the area of the circle, especially as radius gets large)


To convert between CMB power spectrum to flat-sky one, I'd do the following:
    1) Choose my box size and generate some random pixel data
    2) Make a grid of (kx,ky) modes and find the k=sqrt(kx^2+ky^2) for each
    3) Convert each k in the grid to its closest l value (following l=18k)
    4) Have a sort of look-up system to find the Cl value for each of the
       grid spaces.  Now each k on the grid has a Cl value (i.e. power spectrum)
       that corresponds to CMB observations.
    5) The random pixel data is convolved with this power spectrum

These steps are carried out below
"""

l = np.arange(len(TT))

#closest multiple of 18 that doesn't exceed lmax
kmax = np.floor(lmax/18.)

n = 1000
x = np.arange(n)

# !!! for some reason python is tripping on this next line when I run entire script
#  If I just run the single line (I'm using Spyder) it works fine... weird 
x[n/2:] = x[n/2:]-n

# generate random pixel data
dat = np.random.randn(n,n)
datft=np.fft.fft(dat)

# build the different (kx,ky) modes
kx = np.repeat([x],n,axis=0)
ky = np.transpose(kx)
k = np.sqrt(kx**2 + ky**2)


#convert k-values to l-values (whole numbers)
flatls= np.round(k*18)
#initialized a power spectrum matrix
flatPS=np.zeros((n,n))


# Take every k-value in grid, if k>kmax, set PS to zero there
# otherwise take the Cl value of the closest corresponding l
for i in range(n):
    for j in range(n):
        if k[i,j]>kmax:
            flatPS[i,j] = 0
        else:
            flatCl = Cl[np.where(l==flatls[i,j])]
            #print flatCl
            flatPS[i,j] = flatCl
# I realize this is the dreaded slow python nested "for loop"
# but I couldn't think of a more elegant way to do it


# replace the typical 1/r noise with the PS we just created
pk = flatPS
#pk = 1/k**2

dat_back = np.fft.ifft2(datft*np.sqrt(pk))
dat_back=np.real(dat_back)

vardat=np.std(dat_back)**2

#if i scale flat data by sqrt(CMBvar/flatvar) I get the same variance
# as the full CMB map
vardat2=np.std(dat_back*np.sqrt(var/vardat))**2


from matplotlib.colors import LogNorm

plt.figure(num=6)
plt.clf()
plt.title("Power Spectrum (k>kmax set to zero)")
plt.imshow(flatPS,vmin=1e-5,vmax=1e1,norm=LogNorm())
plt.colorbar()
plt.savefig("figures/Fig6_prob2_PS_for_flat_map.png")



plt.figure(num=7)
plt.clf()
plt.title("Flat-sky CMB map\n$\sigma=%.5f$"%(np.sqrt(vardat)))
plt.imshow(dat_back)
plt.colorbar()
plt.savefig("figures/Fig7_prob2_flat_sky_CMB.png")

plt.figure(num=8)
plt.clf()
plt.title("Flat-sky CMBs map (scaled)\n$\sigma=%.0f, \sigma^2=%.0f$"%(np.sqrt(vardat2),vardat2))
plt.imshow(dat_back*np.sqrt(var/vardat),vmin=-500,vmax=500)
plt.colorbar()
plt.savefig("figures/Fig8_prob2_flat_sky_CMB(scaled_for_variance).png")

"""
There is a strange vertical symmetry I see in this figure...
I suspect it's some subtlety of the fourier transform, but I'm not
sure what it is.

Otherwise it looks pretty good I think.
"""







"""
-------------------
Part C)

maximum pixel will be set by the smallest resolution modes, corresponding to
lmax or kmax.

Since I set lmax = 2500, that makes kmax = 138.  That implies there can be 
138 wavelengths across the 20 degree box.  I would think you'd want a minimum of
something like 4 pixels per wavelength, resulting in Npix = 552.

In the above plots (fig 7) I chose 1000 pixels, giving about 7 pixel resolution 
on the smallest structures.






-------------------
Part D)


In Part A of Problem 1 we learned that the variance of the CMB full map is

 <T^2> = < (SUM alm Ylm)^2 >   -- SUM is over l,m
          = < SUM alm^2 Ylm^2 >   -- because cross terms have zero average
          = < SUM Cl Ylm^2 >
          = 1/4pi SUM Cl
          = 1/4pi SUM_l Cl*(2l+1) -- SUM is now over just l
          
          ~ 12000

This needs to be the same as the variance of the 2D FT
We know:
    f(x) = 1/N SUM_k F(k) exp(i2pi (k dot x)/N)  
            (where k and x are vectors depending on dimension)

Variance:

 <f^2> = 1/(npix^2) SUM F(k)^2  exp(i2pi...)exp(-i2pi...)  {conjugate cancels}
       = 1/(npix^2) SUM P(k)

        {this SUM is over kx,ky}

"""
# analytic variance of flat map
anavar=1./n**2 * np.sum(flatPS)
ratio = var/anavar

print "Flat map analytic variance: %.5f"%anavar
print "  off by factor %.0f from non-flat CMB variance"%ratio

"""
There is this factor of nearly 10^7 between my predicted variance here
and that of my non-flat CMB map.  This is the amplitude scaling between
two systems (like we had an 18-fold angular scaling?).

"""






""" 
    <><><><><><><><><><><><<><><><><> 
    <><><><><>  PROBLEM 3  <><><><><> 
    <><><><><><><><><><><><<><><><><> 
"""

# see paper notes
print "\n\n Problem 3\n\n"




#physical constants
T0 = 2.725          # K         (CMB temp)
h  = 6.626196e-27   # erg s     (Planck const)
kb = 1.380658e-16   # erg K^-1  (Boltz const)
c  = 3.0e10         # cm/s      (speed of light)

# instrument constants
BW=30e9     # Hz  (bandwidth)
v0=150e9    # Hz  (central freq)
lamb = c/v0 # cm  (wavelength and detector size)



# [erg s^-1 cm^-2 Hz^-1 sr^-1]
def Bv(v,T):
    x = h*v/(kb*T)    
    return (2*h*v**3)/(c**2) /(np.exp(x)-1)



"""
Part A)
"""

# To get ergs per second per Hz
#   Take Planck function
#       - multiply by beam size (sr)
#       - multilpy by detector size (cm^2)
ergpsphz = Bv(v0,T0) * 1. * lamb**2
print "ergs per second per Hz: %.3e"%ergpsphz
"""
Get 1.525e-16 ergs per second per Hz
"""



"""
Part B)
"""


# To get photons per second: 
#   Take Planck function
#     - multiply by beam size
#     - multiply by detector size
#     - divide by energy per photon
#     - multiply by bandwidth 
photps= Bv(v0,T0)* 1.* lamb**2 / (h*v0) * BW 
print "photons per second: %.3e"%photps
"""
Get 4.602e+09 photons per second
"""



"""
if incoming rate is similar to the frequency, then we're in continuous limit

Incoming rate:  4.64e9   photons/second
Frequency:      1.50e11  Hz

Incoming rate is roughly factor 32 smaller.  Seems like a decently large factor.
    - should get small 150Ghz wiggles in time spacing of 4.6GHz
    - could fit 32 oscillations of one photon between each arrival

Leads me to say we're in the shot noise limit?  
    This doesn't sound right for a microwave telescope though...
    I'll work out both.






What is fractional error ( dT/T or dn/n )? 
if shot limit, fractional error is 1/sqrt(nt)  {n=rate, t=time}

dn/n = 1/np.sqrt(nt)
     = 1 / np.sqrt( 4.64e9 s^-1 * 1s)
     = 1.474e-05 
     = 15 ppm


if continuous, fractional error is 1/sqrt(Bt)  {B=bandwidth}

dT = T/np.sqrt(Bt)
   = 2.725 K / np.sqrt( 30e9 s^-1 * 1s)
   = 1.573e-05 K 
   = 16 uK

It seems like we've done a factor of a few better than the Planck 143 detectors
of 50 uK s^1/2.  Simply improving detectors won't have too much effect (factor of 3).
You'd have to play with the bandwidth and integration time to see any real improvement.

( I don't understand where we get [s^1/2] units in the Planck noise.  The values
  in the sqrt() have cancelling units)

"""


















