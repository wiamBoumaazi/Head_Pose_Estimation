#### Author : hjsong

#### Change file name headPoseCC  headPoseFileName
#### Change cutoff # read the comments in cutoff
#### Change uptoWhichFrame

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


from numpy.core.fromnumeric import shape
from scipy.signal import savgol_filter
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import interp1d
from scipy.signal import butter,filtfilt

headPoseCC="Bill_yaw"  # findex, head pose in angles,  pitch roll yaw
#headPoseCC="Rsl/Oct/adamOrigETH_Rodr_1006"
#headPoseCC="Rsl/Oct/bilOrigETH_Rodr_1002"  # head pose in angles pitch roll yaw
#headPoseCC="Rsl/Oct/bilHeadPostYawPitchETH_1001"  # head pose in angles pitch roll yaw
#headPoseCC="Rsl/Oct/bilHeadPostYawPitchBasic_1001"  # head pose in angles pitch roll yaw
headPoseFileName="Bill_noseTip"  # findex, n_noseTipsX
#headPoseFileName="Rsl/AugEnd/headposeOutput_yaw.adam"  # n_noseTipsX n_leftEyeCornerX
#eyePoseFileName="Rsl/AugEnd/eyeOutput.bil.0829_02"  # pupil center of left eye and right eye  
#eyePoseFileName="Rsl/AugEnd/eyeOutput0829_01.adam" 

uptoWhichFrame = 1400  # specify which frame range you would like to draw, i.e. 0 frame ~ uptoWhichFrame frame
froMmil = 0    # specify from when (in milisec) you would like to draw. Since the data is huge, need to subset. From 1400mil (~1400/4 frame), data is shown 

# los pass filter
#https://medium.com/analytics-vidhya/how-to-filter-noise-with-a-low-pass-filter-python-885223e5e9b7
T = 6.25        # Sample Period in sec, 1500 frame --> 1500/240=6.25 sec 
fs = 240.0      # sample rate, Hz
cutoff = 2     # desired cutoff frequency of the filter, Hz , slightly higher than actual signla freq. Hz 
                #(4 head movement/6.25sec=0.64Hz )
nyq = 0.5 * fs  # Nyquist Frequency
order = 2       # sin wave can be approx represented as quadratic
n = int(T * fs) # total number of samples

def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


index=(int)(1000/240)


def fileReadHeadPose(headPoseFileName, fIndex):

    """
    Find the largest contour on an image divided by a midpoint and subsequently the eye position

    Parameters
    ----------
    headPoseFileName : String
        Filename to include facial landmakrs (frame index, noseTip in X (2D coord), left eye corner in X)
        how many : how many frames to read from 0 to how many -1 
    Returns 
    -------
        xx : array of int (frame index)
        n_noseTipsX : array of int (noseTipxs in X)
        n_leftEyeCornerX  : array of int (leftEyeCorner in X)
    """
    xx=[]
    x_coarse_mil = []
    x_fine_mil = []
    n_noseTipsX=[]
    n_fine_noseTipsX=[]   # for every mil sec 
    n_leftEyeCornerX=[]
    #n_gazePitchYaw_0=[]
    #n_gazePitchYaw_1=[]

    f=open(headPoseFileName, 'r')
    
    for i in range (fIndex+1): 
        line = f.readline()
        if line.startswith("#") == True : 
            line = f.readline()
        items=line.split() 
        xx.append((items[0]))
        n_noseTipsX.append((items[1]))  
       # n_leftEyeCornerX.append(int(items[2].strip()))

        #n_gazePitchYaw_0.append (float(items[3]))
        #n_gazePitchYaw_1.append (float(items[4]))

    for i in range (fIndex+1) : 
        x_coarse_mil.append ((float)(i*index))  #0, 4, 8, .. fIndex*4 

    for i in range (int(index*(fIndex)+1)) : 
        x_fine_mil.append (i)    #0, 1, ...,  index*fIndex


    noseTip_interprep =  interp1d(x_coarse_mil, n_noseTipsX)

    for i in range (int(index*(fIndex)+1)) : 
        n_fine_noseTipsX.append (
            (float) (noseTip_interprep ( x_fine_mil[i] ) ))

    f.close()
    return  x_fine_mil, n_fine_noseTipsX #, n_gazePitchYaw_0, n_gazePitchYaw_1
    

def fileReadHeadPoseCC (fname,fIndex):
    x =  []  # frame scale  starting from 0 
    x_mil = [] # 
    x_coarse_mil = [] 
    x_fine_mil = []
    n_pitch  =[]
    n_roll =[]
    n_yaw =[]
    n_yaw_fine_mil=[]

    f=open(fname, 'r')
    for i in range (fIndex+1): 
        line = f.readline()
        if line.startswith("#") == True : 
            line = f.readline()
        items=line.split() 
        x.append(i)    
        n_pitch.append(float(items[1]))
        n_roll.append(float(items[2]))
        n_yaw.append(float(items[3].strip()))

    
    for i in range (fIndex+1) : 
        x_coarse_mil.append ((float)(i*index))  #0, 4, 8, .. fIndex*4 

    for i in range (int(index*(fIndex)+1)) : 
        x_fine_mil.append (i)    #0, 1, ...,  index*fIndex

    yaw_interprep =  interp1d(x_coarse_mil, n_yaw)

    for i in range (int(index*(fIndex)+1)) : 
        n_yaw_fine_mil.append (
            (float) (yaw_interprep ( x_fine_mil[i] ) ))

    return n_yaw_fine_mil




def drawPlot(fIndex,headPoseFileName) :
    

    x_fine_mil, n_fine_noseTipsX=fileReadHeadPose(headPoseFileName, fIndex)
    n_yaw_fine_mil =fileReadHeadPoseCC (headPoseCC,fIndex)
    assert len(x_fine_mil) == len(n_yaw_fine_mil)

    v_lp_headYaw_fine_mil = [0]
    v_lp_headYaw_fine_mil_1000 = []
    
    v_right_eye = [0]  # velocity for right eye 
    v_left_eye = [0]   # velocity for left eye
    v_rel_left_eye = [0] # velocity for  left eye relative to left eye corner
    v_n_noseTipsX = [0]


    n_lp_headYaw_fine_mil = butter_lowpass_filter(n_yaw_fine_mil, cutoff, fs, order)

    # hjsong, Shift right X into left side, to easily visualize in a plot 
    for i in range (len(n_fine_noseTipsX)) :
        n_fine_noseTipsX[i]=n_fine_noseTipsX[i]- int(min (n_fine_noseTipsX)*0.66)

    for i in range (len(x_fine_mil)-1) :       
        v_lp_headYaw_fine_mil.append ( float( ((n_lp_headYaw_fine_mil[i+1] -  n_lp_headYaw_fine_mil[i])) ) )  #240 fps
    #f1_v_headYaw_fine_mil = savgol_filter(v_headYaw_fine_mil, window, polyorder) #lowess filter, window size, poly order
    for i in range (len(x_fine_mil)) :    
        v_lp_headYaw_fine_mil_1000.append ( v_lp_headYaw_fine_mil [i] *1000 )  



    spacing = 50
    lw=1.5


    fig, axs=plt.subplots (2,2)  #  row, col 

    for i in range(2) :  # should be changed to 4
            #set grid
        minorLocator = MultipleLocator (spacing)
        axs[0][i].yaxis.set_minor_locator (minorLocator)
        axs[0][i].xaxis.set_minor_locator (minorLocator)
        axs[0][i].yaxis.set_major_locator (MultipleLocator(spacing))
        axs[0][i].xaxis.set_major_locator (MultipleLocator(spacing))
        axs[0][i].grid (which='both')
    

        axs[0][i].plot(x_fine_mil, n_lp_headYaw_fine_mil, color='deepskyblue', label='n_lp_headYaw_fine_mil', linewidth=lw)
        axs[0][i].plot(x_fine_mil, v_lp_headYaw_fine_mil_1000, color='red', label='v_lp_headYaw_fine_mil_1000', linewidth=lw)

        axs[0][i].set_ylabel('velocity(degree/sec)')
        axs[0][i].legend()
       
        axs[0][i].set_xlim (froMmil+1400 * (i), froMmil+1400 * (i+1))  
        axs[0][i].set_ylim (-400, 400)  

        
        
        spacing = 50
        
        minorLocator = MultipleLocator (spacing)
        axs[1][i].yaxis.set_minor_locator (minorLocator)
        axs[1][i].xaxis.set_minor_locator (minorLocator)
        axs[1][i].yaxis.set_major_locator (MultipleLocator(spacing))
        axs[1][i].xaxis.set_major_locator (MultipleLocator(spacing))

        axs[1][i].grid (which='both')
        


        #velocity 
        #ax2.plot(x, v_right_eye, color='blue', label='velocity right eye', linewidth=lw)
        #ax2.plot(x, v_left_eye, color='green', label='velocity left eye', linewidth=lw)
        #ax2.plot(x, v_rel_left_eye, color='yellow', label='velocity left eye (relative to left eye center)', linewidth=lw)
        #axs[1][i].plot(x, f1_v_rel_left_eye, color='red', label='velocity relative left eye (filter)', linewidth=lw)
        #axs[1][i].plot(x, v_gazeYaw, color='red', label='v_gazeYaw', linewidth=lw)
        #axs[1][i].plot(x, v_headYaw, color='indigo', label='v_headYaw', linewidth=lw)
        #axs[1][i].plot(x, n_noseTipsX, color='green', label='n_noseTipsX',linewidth=lw)
        axs[1][i].plot(x_fine_mil, n_fine_noseTipsX, color='red', label='n_fine_noseTipsX',linewidth=lw)

        axs[1][i].set_xlabel('milSec' )
        axs[1][i].set_xlim (froMmil+1400 * (i), froMmil+1400 * (i+1))
        #axs[1][i].set_ylim (-30, 30)
        axs[1][i].set_ylabel('location(pixel/frame)')

        axs[1][i].legend()
    
    plt.show()


drawPlot(uptoWhichFrame,  headPoseFileName)  # frame 0 to frame 1400

    
