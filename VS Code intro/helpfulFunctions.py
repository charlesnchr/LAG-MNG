import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def calcPSF(w,pixelSize,NA,wavelength):
    """
    Calculates the phase mask to reproduce the 4 main aberations for high NA objectives
    ----------
    w: width of square mask
    pixelSize: Pixel size in nm
    NA: Numerical aperture of focussing lens
    wavelength: wavelength
    
    Returns
    -------
    mask:
        W x W square array containing the phase mask
    
    """

    x = np.linspace(-w/2,w/2,w)
    y = np.linspace(-w/2,w/2,w)
    # This provides us with two helpful matrices we can use as a grid of x,y co-ordinates
    X,Y = np.meshgrid(x,y)

    # We can then calculate an array representing the distance from the central element
    Pupil = np.sqrt(X**2+Y**2)
    Pupil[Pupil<w/8]=1
    Pupil[Pupil>=w/8]=0
    plt.imshow(Pupil, cmap='gray')
    plt.show()

    psf = np.abs(np.fft.fftshift(np.fft.fft2(Pupil)))
    psf = psf**2

    return psf



def loadImage(path):

    '''
    Load an image as a numpy array
    
    ------
    path: path to image as str 
    
    Returns:
    ------
    img: numpy array of image 
    '''
    img =  np.asarray(Image.open(path))

    return img



