import imageIO as io
import numpy as np
from scipy import ndimage
from scipy import signal
import glob
import unwrap
import bilagrid

io.baseInputPath = './'

 #gets the png's in a folder and puts them in out.
def getPNGsInDir(path):
    fnames = glob.glob(path+"*.png")
    out=[]
    for f in fnames:
        #print f
        imi = io.imread(f)
        out.append(imi)
    return out

#converts the png images in a path path to a npy file at pathOut
def convertToNPY(path, pathOut):
    L=getPNGsInDir(path)
    V=np.array(L)
    np.save(pathOut, V)

# writes the frames of video to path.
def writeFrames(video, path):
    nFrame=video.shape[0]	
    for i in xrange(nFrame):
        pathi=path+str('%03d'%i)+'.png'
        #if i%10==0: print i 
        io.imwrite(video[i], pathi)
    print 'wrote'+path+'\n'

#Convert an RGB video to YUV.
def RGB2YUV(video):
    RGB2YUVmatrix=np.transpose([[0.299,  0.587,  0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.10001]])
    return np.dot(video[:, :, :], RGB2YUVmatrix)

# Convert an YUV video to RGB.
def YUV2RGB(video):
    YUV2RGBmatrix=np.transpose([[1, 0, 1.13983], [1, -0.39465, -0.58060], [1, 2.03211, 0]])
    return np.dot(video[:, :, :], YUV2RGBmatrix)

############ Functions for Eulerian linear video magnification [1] ##########

# apply a Gaussian blur to an input "video" with standard deviation "sigma"
def lowPass(video, sigma):
    filteredVideo = ndimage.filters.gaussian_filter(video, (0, sigma, sigma, 0))

    return filteredVideo

# temporally bandpass filter a video using a Butterworth IIR filter
#   low - low frequency cutoff
#   high - high frequency cutoff
#   order - order of Butterworth filter
def timeBandPassButter(video, low, high, order=2):
    B,A = signal.butter(order, [low, high], 'bandpass')

    butterVideo = np.zeros_like(video)

    for i in xrange(video.shape[0]):
        #print("filtering frame " + str(i))

        for j in xrange(1, len(B)):
            if ((i-j) >= 0):
                butterVideo[i] += -A[j]*butterVideo[i-j]
                butterVideo[i] += B[j]*video[i-j]
            else:
                butterVideo[i] += -A[j]*butterVideo[0]
                butterVideo[i] += B[j]*video[0]

        butterVideo[i] += video[i]*B[0]

    return butterVideo/A[0]

# Linear video magnification using Butterworth IIR filter
#   video - input video to magnify
#   sigmaS - spatial low pass filter (Gaussian) standard deviation
#   low - temporal low frequency cutoff
#   high - temporal high frequency cutoff
#   alphaY - luminance gain
#   alphaUV - chrominance gain
def videoMagButter(video, sigmaS, low, high, order, alphaY, alphaUV):
    #np.savetxt("output.txt", video[:,203,176,:])

    video = RGB2YUV(video)
    
    lowpassedVideo = lowPass(video, sigmaS)

    tbpVideo = timeBandPassButter(lowpassedVideo, low, high, order)

    tbpVideo[:,:,:,0] *= alphaY
    tbpVideo[:,:,:,1:] *= alphaUV

    video += tbpVideo

    return YUV2RGB(video)

############ Functions for phase based motion amplification [2] #############

# Generate a Fourier mask, used for complex steerable pyramid generation
#   height - height of mask
#   width - width of mask
#   radiusStart - minimum radius (frequency) of mask (inclusive)
#   radiusStop - maximum radius (frequency) of mask (exclusive)
#   angleStart - minimum angle (orientation) of mask (inclusive)
#   angleStop - maximum angle (orientation) of mask (exclusive)
#   sigma - standard deviation of blur applied to mask (to limit sidelobes)
def generateMask(height, width, radiusStart, radiusStop, angleStart, angleStop, sigma=10):
    mask = np.zeros((height, width))

    for y in xrange(height):
        for x in xrange(width):
            ny = (float(y)/(height-1) - 0.5)*2
            nx = (float(x)/(width-1) - 0.5)*2

            r = np.sqrt(ny**2 + nx**2)
            th = np.arctan2(ny, nx)

            if (r >= radiusStart) and (r < radiusStop):
                angleDiff = th - angleStart

                while angleDiff < 0:
                    angleDiff += 2*np.pi

                while angleDiff >= 2*np.pi:
                    angleDiff -= 2*np.pi

                if (angleDiff < (angleStop - angleStart)) and (angleDiff >= 0):
                    mask[y,x] = 1

    # Gaussian blur to try to reduce artifacts
    mask = ndimage.filters.gaussian_filter(mask, sigma, mode='nearest')

    return mask

# generate an array of Fourier masks to cover frequency space
#   height and width specify size of mask
#   numOrientations specifies the number of frequency orientations (angles) to create a mask for
#   numLevels specifies the number of frequency bands to create a mask for
#       there will always be an initial omnidirectional low pass, and a high pass residual
#   total number of masks generated is numOrientations*numLevels + 2
def generateMasks(height, width, numOrientations=4, numLevels=2):
    masks = np.zeros((numOrientations*numLevels+2, height, width))

    # low pass
    masks[0] = generateMask(height, width, 0, (1.0/(numLevels+1))**2, 0, 2*np.pi)
    # high pass
    masks[1] = np.ones((height, width)) - generateMask(height, width, 0, 1.0, 0, 2*np.pi)

    for l in xrange(numLevels):
        rStart = (float(l+1.0)/(numLevels+1.0))**2
        rStop = (float(l+2.0)/(numLevels+1.0))**2

        for o in xrange(numOrientations):
            oStart = 2*np.pi*float(o)/(numOrientations*2)
            oStop = 2*np.pi*float(o+1.0)/(numOrientations*2)

            masks[l*numOrientations + o + 2] = generateMask(height, width, rStart, rStop, oStart, oStop)

    return masks

# generate a complex steerable pyramid [1] from image "im", using Fourier masks "ms"
def generatePyramid(im, ms):
    im_fft = np.fft.fftshift(np.fft.fft2(im))

    ffts = np.ones_like(ms, dtype=np.dtype(complex))

    filtered_angle = np.ones_like(ms)
    filtered_mag = np.ones_like(ms)

    for i in xrange(ms.shape[0]):
        # use the mask to select the region of frequency space we want
        ffts[i] = im_fft * ms[i]

        # return to spatial space
        filtered = np.fft.ifft2(np.fft.ifftshift(ffts[i]))

        filtered_angle[i] = np.angle(filtered)
        filtered_mag[i] = np.abs(filtered)

    return (filtered_mag, filtered_angle)

# reconstruct an image from a complex steerable pyramid, pyramid
#   pyramid is a tuple of (magnitudePyramid, phasePyramid)
def reconstruct(pyramid):
    output = np.zeros_like(pyramid[0][0])

    for i in xrange(pyramid[0].shape[0]):
        # we only care about the real part
        real = np.cos(pyramid[1][i]) * pyramid[0][i]

        # if it was one of the non-residual filters, we need to double it in order to 
        # take into account it's opposite orientation counterpart. since mirroring around
        # frequency 0 simply takes the conjugate of the Fourier transform result, we can
        # simply double the real part (and safely ignore the imaginary)
        if (i > 1):
            real *= 2

        output += real

    return output

# create a complex steerable pyramid for each frame in a video, with numOrientations and numLevels
def videoPyramid(video, numOrientations=4, numLevels=2):
    print("Generating Fourier masks")
    ms = generateMasks(video.shape[1], video.shape[2], numOrientations, numLevels)

    video_apyr = np.zeros((video.shape[0], numOrientations*numLevels + 2, video.shape[1], video.shape[2], 3))
    video_mpyr = np.zeros_like(video_apyr)

    for j in xrange(3):
        print("Channel: " + str(j))
        for i in xrange(video.shape[0]):
            print("\tFrame: " + str(i+1) + "/" + str(video.shape[0]))
            pyr = generatePyramid(video[i,:,:,j], ms)
            video_apyr[i,:,:,:,j] = pyr[1]
            video_mpyr[i,:,:,:,j] = pyr[0]

    return (video_mpyr, video_apyr)

# apply phase-based motion magnification technique
#   phasePyramid - complex steerable pyramid of video phases
#   magnitudePyramid - complex steerable pyramid of video magnitudes
#   low - temporal low frequency cutoff
#   high - temporal high frqeuency cutoff
#   order - order of temporal filter (IIR)
#   gains_y - gains to apply to each level of the pyramid
#   gains_uv (optional) - if used, apply a different gain to the chroma components of the
#       complex steerable pyramid. not sure if necessary
def videoMagPhase(phasePyramid, magnitudePyramid, low, high, order, gains_y, gains_uv=None, frame_write_path=None):
    for i in xrange(phasePyramid.shape[1]):
        print("Processing layer: " + str(i+1) + "/" + str(phasePyramid.shape[1]))
        phases = phasePyramid[:,i,:,:,:]

        if gains_uv is not None:
            print("\tRGB to YUV")
            phases = RGB2YUV(phases)

        print("\tButterworth filter")
        tbpPhases = timeBandPassButter(phases - phases[0], low, high, order)

        print("\tPhase amplification")
        diffPhase = np.diff(tbpPhases, n=1, axis=0)

        diffPhase = (diffPhase + np.pi) % (2 * np.pi) - np.pi

        if gains_uv is not None:
            diffPhase[:,:,:,0] *= gains_y[i]
            diffPhase[:,:,:,1:] *= gains_uv[i]
        else:
            diffPhase *= gains_y[i]

        tbpPhases[1:,:,:,:] = np.cumsum(diffPhase, axis=0) + tbpPhases[0,:,:,:]

        if gains_uv is not None:
            print("\tYUV to RGB")
            phases= YUV2RGB(phases + tbpPhases)
        else:
            phases = (phases + tbpPhases)

        # print("\tPhase denoising")
        # phases = ndimage.filters.median_filter(phases, (1, 3, 3, 1))
        # for j in xrange(phases.shape[0]):
        #     print("\t\tFrame " + str(j+1) + "/" + str(phases.shape[0]))
        #     phases[j] = bilagrid.bilateral_grid(phases[j], max(phases.shape[1], phases.shape[2])/50.0, 0.4)

        phasePyramid[:,i,:,:,:] = phases

    print("Reconstructing")

    frame = np.zeros((phasePyramid.shape[2], phasePyramid.shape[3], phasePyramid.shape[4]))

    if frame_write_path is None:
        video = np.zeros((phasePyramid.shape[0], phasePyramid.shape[2], phasePyramid.shape[3], phasePyramid.shape[4]))

    for i in xrange(phasePyramid.shape[0]):
        print("Reconstructing frame " + str(i+1) + "/" + str(phasePyramid.shape[0]))

        for j in xrange(3):
            frame[:,:,j] = reconstruct((magnitudePyramid[i,:,:,:,j], phasePyramid[i,:,:,:,j]))

        if frame_write_path is not None:
            io.imwrite(frame, frame_write_path + str(i).zfill(5) + ".png")
        else:
            video[i] = frame

    if frame_write_path is None:
        return video
    else:
        return None

########### References ##################
#   [1] http://people.csail.mit.edu/mrub/papers/vidmag.pdf
#   [2] http://people.csail.mit.edu/billf/www/papers/phase-video.pdf

