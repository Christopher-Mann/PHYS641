from __future__ import division
import numpy as np
import simple_read_ligo as rl
from matplotlib import pyplot as plt



""" SET THESE VARIABLES
 -- do you want to make plots in a folder called 'figures'? (probably not)
 -- where is your LIGO data located?
 -- are we looking at Livingston or Hanford?
"""

# Run with which="H" first, then with which="L" (sorry, order matters here)

makeplots = False
data_location = "./LIGO_tutorial_data/"
which = "H"  













# we need this "shifted" vector because the discrete fourier transform
# only looks at -n/2 to n/2 and assumes periodicity
def make_ft_vec(n):
    #make a routine to make a vector that goes from
    #0 to n/2, then -n/2 up to -1
    x=np.arange(n)
    half = int(n/2)
    x[half:]=x[half:]-n
    assert(x[-1]==-1)
    return x*1.0  #make sure we return it as a float, not an int


def gaussian_smoothing(dat,FWHM):
    """
    Takes some data, convolves it with a gaussian
      (by multiplying in Fourier space, then FT back)
    """
    n = len(dat)
    x = make_ft_vec(n)                  # get this special "shifted" vector
    sig = FWHM/np.sqrt(8*np.log(2))     # sets std dev of kernal so we can specify FWHM 
    smooth_vec = np.exp(-0.5*x**2/sig**2) # gaussian smoothing shape
    smooth_vec = smooth_vec/smooth_vec.sum() # normalizes so we're not scaling
    datft = np.fft.rfft(dat)            # FT data
    vecft = np.fft.rfft(smooth_vec)     # FT smoothing kernal
    dat_smoothft = datft*vecft          # multiply together = convolution when we FT back
    dat_smooth = np.fft.irfft(dat_smoothft,n=n) # FT back to real space
    return dat_smooth
    



# Event labels, data files, and template files all in the same order
eventname        = [           'GW170104',                          'GW150914',                          'LVT151012',                        'GW151266']
H_datafiles      = ['H-H1_LOSC_4_V1-1167559920-32.hdf5','H-H1_LOSC_4_V2-1126259446-32.hdf5','H-H1_LOSC_4_V2-1128678884-32.hdf5','H-H1_LOSC_4_V2-1135136334-32.hdf5']
L_datafiles      = ['L-L1_LOSC_4_V1-1167559920-32.hdf5','L-L1_LOSC_4_V2-1126259446-32.hdf5','L-L1_LOSC_4_V2-1128678884-32.hdf5','L-L1_LOSC_4_V2-1135136334-32.hdf5']
template_datafiles = [  'GW170104_4_template.hdf5',         'GW150914_4_template.hdf5',           'LVT151012_4_template.hdf5',     'GW151226_4_template.hdf5']





if which=="H":
    Hanford = True
    SNR_H = np.zeros(len(eventname))
if which=="L":
    Hanford = False
    SNR_L = np.zeros(len(eventname))






for k in range(len(eventname)):
    
    if Hanford:
        fname = data_location + H_datafiles[k]
    else:
        fname = data_location + L_datafiles[k]
    
    template_name = data_location + template_datafiles[k]
    
    print "reading file:",fname.lstrip(data_location) 
    strain,dt,utc=rl.read_file(fname)
    tH,tL=rl.read_template(template_name)
    
    
    
    """
    A) Coming up with a noise model
    """
    
    
    
    # let's make a window function
    # it should drop to zero at the edges of our data but be close to 1 away from edges
    
    n = len(strain)
    x = np.linspace(-1.,1.,n)*np.pi
    """ window could be the one used in class (cosine function) """
    #window = 0.5 + 0.5*np.cos(x)
    """ or we can try tophat-looking function so it's flat in the middle """
    #step_in = np.int(n/10)
    #window=np.ones(len(x))
    #window[:step_in] =np.linspace(0,1,step_in)
    #window[-step_in:]=np.linspace(1,0,step_in)
    """ or we can try sinusoid edges with flat middle to keep it smooth """
    step_in = np.int(n/50)
    window=np.ones(len(x))
    window[:step_in] =-0.5*np.cos(np.arange(0,step_in)/step_in*(np.pi))+0.5
    window[-step_in:]=(-0.5*np.cos(np.arange(0,step_in)/step_in*(np.pi))+0.5)[::-1]
    # window function plotted in figure 1
    
    
    
    # Look at the power spectrum of the strain, then smooth
    strain_ft = np.fft.rfft(strain*window)
    PS_raw = np.abs(strain_ft)**2
    PS_smooth = gaussian_smoothing(PS_raw,5.)
    """
    I didn't explicitly deal with spikes/peaks in the power spectrum while smoothing.
     Choosing a FWHM of 5 seemed a good balance between capturing the peaks
     of the power spectrum, but still suppressing the noise fairly well
     Figure 3 shows some examples of peaky locations in the power spectrum
    """
    
    
    """
    B) Using the noise model to search for GW event
    """
    
    
    # converting from data point number to frequency
    dnu = 1/(n*dt) #dt is time between measurements (I think...)
    nu = np.arange(len(PS_smooth))*dnu
    
    
    ## Noise = 1/power spectrum
    Ninv = 1/PS_smooth
    """ !!! I don't really understand why Ninv is 1/PS 
            it seems as if noise is defined in Fourier space?
            
            noise is the power in some little range
            statstically time invariant, so freq space has to have the
            same info, can treat it as uncorrelated noise 
                (correlated in time-domain)
    """
    # Set the weights to zero for bad regions (high and low freqs)
    highcut = 1630;lowcut=7
    Ninv[nu<lowcut]=0
    Ninv[nu>highcut]=0
    # whiten noise by applying half of the full N_inv scaling power
    dat_whitened_ft = strain_ft*np.sqrt(Ninv)
    dat_whitened = np.fft.irfft(dat_whitened_ft,n)
    
    # Template:
    tH_ft = np.fft.rfft(tH*window)
    tH_ft_whitened = tH_ft*np.sqrt(Ninv)# Model gets other half of N_inv whitening
    tH_whitened = np.fft.irfft(tH_ft_whitened)
    
    # replacing numerator of (A^T N^inv d)/(A^T N^inv A) with
    # the pre-whitened correlation of A with d
    top = np.fft.irfft( np.conj(tH_ft_whitened) * dat_whitened_ft )
    # because correlation is invFT( FT(template)_conj  * FT(data) )
    
    # Template and Signal is plotted in Figure 4
    
    """
    C) noise for the event
    """
    
    # sigma_m = (A^T N^inv A)^(-1/2)
    # SNR = (A^T N^inv d)/sqrt(A^T N^inv A)
    
    """
    # Notes to myself:
    # just look at noise in the match filter
    # look at the RMS of the bottom of Fig 4 (away from window dip or spike)
    #  just use peak value and typical RMS... scalar is okay, we're just interested
    #  in the peak response, not the "sort of" matches a few shifts away
    """
    # a typical undisturbed section of the match filter (no window now spike)
    peakind = np.where(np.abs(top)==np.max(np.abs(top)))[0]
    RMS_begin = int(peakind+0.005*len(top))
    RMS = np.std(top[RMS_begin:62400])
    SNR = np.abs(top[peakind]/RMS)
    # Shown in Figure 4 bottom panel
    
    print "    Detection SNR: %7.3f"%SNR 
    

    """
    Notes to myself:
     !!! how do I combine the L and H datasets for part c) ?
     !!!  need noise and SNR for L, H, and L+H data
     
         - adding chi^2 of each together
         - SNR^2 = chi^2 (more or less)
         
         Easy:
         - sqrt( SNR^2_L + SNR^2_H ) = SNR_both
         Better:
         - fit single amplitude to two templates to two data sets with two noise models
         - A = vector(AH,AL), d = vector(dH,dL), N = [ NH  (0) ]
                                                     [ (0)  NL ]
    """
    if Hanford:
        SNR_H[k] = SNR
    else:
        SNR_L[k] = SNR
#    
#    SNR_total = np.hypot(SNR_H,SNR_L)
#    
#    print "SNR Hanford:     %.3f\nSNR Livingston: %.3f\nSNR combined    %.3f"%(SNR_H,SNR_L,SNR_total)
#    
    
    
    
    
    """
    D) Freq where half weight is above vs below
    """
    
    
    """
    The following gives an errorbar on amplitude of my GW detection
    given a template and a noise model
    
    variance = (AT Ni A)^-1
        = 1 / SUM(A^2 / N)   if N is diagonal (which is true for FT stationary data)
        = sigam^2     (where A^2 = Aconj * A)
        
        w = 1/sig^2
        
        
        plot A^2 / N
    """
    
    # weights of the data points
    w = np.abs(tH_ft_whitened)**2 * Ninv 
    ncsw = np.cumsum(w)/np.sum(w) # normalized cumulative sum of weights
    midfreq = min(nu[ncsw>0.5])   # where does cumulative sum pass 0.5?
    # Plotted in figure 5
    
    
    print "    Freq with equal weight on either side: %5.0f Hz"%midfreq
    
    
    
    
    if makeplots:
        # This is what the strain data look like and our window function
        plt.figure(num=1,figsize=(12,7))
        plt.clf()
        
        plt.subplot(2,1,1)
        plt.plot(strain,'b-',linewidth=0.2,label="raw strain");plt.legend()
        plt.subplot(2,1,2)
        plt.plot(window,'r-',linewidth=0.5,label="window");plt.legend()
        plt.savefig("figures/"+eventname[k]+"_"+which+"_fig1_strain+window.png")
        
        
        
        # This is what the FT power spectrum of the data look like
        plt.figure(num=2)
        plt.clf()
        plt.loglog(nu,PS_raw,'b-',linewidth=0.2,label="raw power spectrum")
        plt.loglog(nu,PS_smooth,'r-',linewidth=0.2,label="smoothed power spectrum")
        plt.fill_between([0.001,lowcut],1e-25,1e-50 ,color='r',alpha=0.2,label='set weight=0')
        plt.fill_between([lowcut,highcut],1e-25,1e-50 ,color='g',alpha=0.2)
        plt.fill_between([highcut,1e4],1e-25,1e-50 ,color='r',alpha=0.2)
        plt.xlim(1e-2,3e3);plt.ylim(1e-46,1e-28)
        plt.legend()
        plt.savefig("figures/"+eventname[k]+"_"+which+"_fig2_Power_spectrum.png")
        
        
        ## Here we zoom in on places to see the effect of smoothing
        plt.figure(num=3,figsize=(7,6))
        plt.clf()
        plt.subplot(2,2,1)
        plt.loglog(PS_raw,'b-')
        plt.loglog(PS_smooth,'r-'); plt.ylim(1e-41,1e-31);plt.xlim(1030,1400)
        plt.xticks([1200])
        
        plt.subplot(2,2,2)
        plt.loglog(PS_raw,'b-')
        plt.loglog(PS_smooth,'r-'); plt.ylim(1e-41,1e-31);plt.xlim(3650,4350)
        
        
        plt.subplot(2,2,3)
        plt.loglog(PS_raw,'b-')
        plt.loglog(PS_smooth,'r-'); plt.ylim(1e-41,1e-31);plt.xlim(9440,9880)
        plt.xticks([9600])
        
        plt.subplot(2,2,4)
        plt.loglog(PS_raw,'b-')
        plt.loglog(PS_smooth,'r-'); plt.ylim(1e-41,1e-31);plt.xlim(46570,47620)
        plt.xticks([47000])
        plt.savefig("figures/"+eventname[k]+"_"+which+"_fig3_close_ups.png")
        
        
        # this is a template and the signal
        plt.figure(num=4,figsize=(14,7))
        plt.clf()
        plt.subplots_adjust(bottom=0.1)
        
        plt.subplot(2,1,1)
        plt.plot(tL,lw=0.5)
        plt.title("Template")
        
        plt.subplot(2,1,2)
        plt.plot(top,lw=0.5)
        plt.xlabel("Shift of match filter")
        plt.text(peakind+100,top[peakind]*0.9,'Peak = %.3f'%(top[peakind]))
        plt.text(20000,RMS*3,'RMS = %.3f'%(RMS))
        plt.text(40000,top[peakind]*0.9,'SNR = %.3f'%(SNR),fontsize=20)
        plt.savefig("figures/"+eventname[k]+"_"+which+"_fig4_template+match_filter.png")
        
        
        
        
        
        
        # What freq has equal weight on either side?
        plt.figure(num=5)
        plt.clf()
        plt.title(r"Half weight on either side of $\nu=%.0f$ Hz"%midfreq)
        plt.loglog(nu,w)
        plt.axvline(midfreq,color='r')
        plt.text(midfreq*1.05,1e24,r'$\nu=%.0f$ Hz'%midfreq,color='r')
        plt.xlim(1,3000)
        plt.xlabel(r'$\nu$');plt.ylabel("weight (A$^2$/N)")
        plt.savefig("figures/"+eventname[k]+"_"+which+"_fig5_freq_midweights.png")



if Hanford==False:
    SNR_total = np.hypot(SNR_H,SNR_L)
    for i in range(len(eventname)):
        print "\nEvent: %s"%(eventname[i])
        print "SNR Hanford:    %7.3f\nSNR Livingston: %7.3f\nSNR combined:   %7.3f"%(SNR_H[i],SNR_L[i],SNR_total[i])



"""
When you run Handford first, then Livingston
you should get the following output:

reading file: H-H1_LOSC_4_V1-1167559920-32.hdf5
    Detection SNR:   6.981
    Freq with equal weight on either side:   131 Hz
reading file: H-H1_LOSC_4_V2-1126259446-32.hdf5
    Detection SNR:  17.064
    Freq with equal weight on either side:   140 Hz
reading file: H-H1_LOSC_4_V2-1128678884-32.hdf5
    Detection SNR:   6.226
    Freq with equal weight on either side:   113 Hz
reading file: H-H1_LOSC_4_V2-1135136334-32.hdf5
    Detection SNR:   9.564
    Freq with equal weight on either side:   101 Hz


reading file: -L1_LOSC_4_V1-1167559920-32.hdf5
    Detection SNR:   9.006
    Freq with equal weight on either side:   109 Hz
reading file: -L1_LOSC_4_V2-1126259446-32.hdf5
    Detection SNR:  12.566
    Freq with equal weight on either side:   158 Hz
reading file: -L1_LOSC_4_V2-1128678884-32.hdf5
    Detection SNR:   4.866
    Freq with equal weight on either side:   135 Hz
reading file: -L1_LOSC_4_V2-1135136334-32.hdf5
    Detection SNR:   6.967
    Freq with equal weight on either side:   153 Hz
    

Event: GW170104
SNR Hanford:      6.981
SNR Livingston:   9.006
SNR combined:    11.395

Event: GW150914
SNR Hanford:     17.064
SNR Livingston:  12.566
SNR combined:    21.192

Event: LVT151012
SNR Hanford:      6.226
SNR Livingston:   4.866
SNR combined:     7.902

Event: GW151266
SNR Hanford:      9.564
SNR Livingston:   6.967
SNR combined:    11.832

"""



