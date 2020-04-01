import numpy as np
import numpy.fft

def getProfileCounts(img):
    """
    Calculates the average value of pixels around rings of constant thickness starting from the middle of th image.
    Useful for Fourier ring analysis 
    ----------
    img: image 
    
    Returns
    -------
    profileSums:
        2D array of PSF normalised between 0 and 1
    
    References
    ----------
    [1] P. A. Stokseth (1969), "Properties of a defocused optical system," J. Opt. Soc. Am. A 59:1314-1321. 
    """
    dims = img.shape
    x_centre = np.floor(dims[0]/2)
    y_centre = np.floor(dims[1]/2);
    r_max = int(x_centre)
    profileSums = np.zeros(r_max);
    profileCounts = np.ones(r_max);

    for column in range(dims[0]):
        for row in range(dims[1]):
            thisDistance = int(np.round(np.sqrt((row-x_centre)**2 + (column-y_centre)**2)))
            if thisDistance < r_max-1:       
                profileSums[thisDistance] = profileSums[thisDistance] + img[row, column]
                profileCounts[thisDistance] = profileCounts[thisDistance] + 1
    
    profileSums = profileSums/profileCounts
    return profileSums

def calcPSF(xysize,pixelSize,NA,emission,rindexObj,rindexSp,depth):
    """
    Generate the aberrated emission PSF using P.A.Stokseth model.
    Parameters
    ----------
    xysize: number of pixels
    pixelSize: size of pixels (in nm)
    emission: emission wavelength (in nm)
    rindexObj: refractive index of objective lens
    rindexSp: refractive index of the sample
    depth: imaging height above coverslip (in nm)
    Returns
    -------
    psf:
        2D array of PSF normalised between 0 and 1
    
    References
    ----------
    [1] P. A. Stokseth (1969), "Properties of a defocused optical system," J. Opt. Soc. Am. A 59:1314-1321. 
    """

    #Calculated the wavelength of light inside the objective lens and specimen
    lambdaObj = emission/rindexObj
    lambdaSp = emission/rindexSp

    #Calculate the wave vectors in vaccuum, objective and specimens
    k0 = 2*np.pi/emission
    kObj = 2*np.pi/lambdaObj
    kSp = 2*np.pi/lambdaSp

    #pixel size in frequency space
    dkxy = 2*np.pi/(pixelSize*xysize)

    #Radius of pupil
    kMax = (2*np.pi*NA)/(emission*dkxy)

    klims = np.linspace(-xysize/2,xysize/2,xysize)
    kx, ky = np.meshgrid(klims,klims)
    k = np.hypot(kx,ky)
    pupil = k
    pupil[pupil<kMax]=1
    pupil[pupil>=kMax]=0

    #sin of objective semi-angle
    sinthetaObj = (k*(dkxy))/kObj
    sinthetaObj[sinthetaObj>1] = 1

    #cosin of objective semi-angle
    costhetaObj = np.finfo(float).eps+np.sqrt(1-(sinthetaObj**2))

    #sin of sample semi-angle
    sinthetaSp = (k*(dkxy))/kSp
    sinthetaSp[sinthetaSp>1] = 1

    #cosin of sample semi-angle
    costhetaSp = np.finfo(float).eps+np.sqrt(1-(sinthetaSp**2))

    #Spherical aberration phase calculation
    phisa = (1j*k0*depth)*((rindexSp*costhetaSp)-(rindexObj*costhetaObj))
    #Calculate the optical path difference due to spherical aberrations
    OPDSA = np.exp(phisa)

    #apodize the emission pupil
    pupil = (pupil/np.sqrt(costhetaObj))


    #calculate the spherically aberrated pupil
    pupilSA = pupil*OPDSA

    #calculate the coherent PSF
    psf = np.fft.ifft2(pupilSA)

    #calculate the incoherent PSF
    psf = np.fft.fftshift(abs(psf)**2)
    psf = psf/np.amax(psf)

    return psf

def edgeTaper(I,PSF):
    """
    Taper the edge of an image with the provided point-spread function. This 
    helps to remove edging artefacts when performing deconvolution operations in 
    frequency space. The output is a weighted sum of a blurred and original version of
    the image with the weighting matrix determined in terms of the tapering PSF
    
    Parameters
    ----------
    I: Image to be tapered

    PSF: Point-spread function to be used for taper

        
    Returns
    -------
    tapered: Image with tapered edges
    
    """

    PSFproj=np.sum(PSF, axis=0) # Calculate the 1D projection of the PSF
    # Generate 2 1D arrays with the tapered PSF at the leading edge
    beta1 = np.pad(PSFproj,(0,(I.shape[1]-1-PSFproj.shape[0])),'constant',constant_values=(0))
    beta2 = np.pad(PSFproj,(0,(I.shape[0]-1-PSFproj.shape[0])),'constant',constant_values=(0))
    
    # In frequency space replicate the tapered edge at both ends of each 1D array
    z1 = np.fft.fftn(beta1) # 1D Fourier transform 
    z1 = abs(np.multiply(z1,z1)) # Absolute value of the square of the Fourier transform
    z1=np.real(np.fft.ifftn(z1)) # Real value of the inverse Fourier transform
    z1 = np.append(z1,z1[0]) # Ensure the edges of the matrix are symetric 
    z1 = 1-(z1/np.amax(z1)) # Normalise

    z2 = np.fft.fftn(beta2)
    z2 = abs(np.multiply(z2,z2))
    z2=np.real(np.fft.ifftn(z2))
    z2 = np.append(z2,z2[0])
    z2 = 1-(z2/np.amax(z2))

    # Use matrix multiplication to generate a 2D edge filter
    q=np.matmul(z2[:,None],z1[None,:])

    # Generate a blured version of the image
    padx = int(np.floor((I.shape[0]-PSF.shape[0])/2))
    pady = int(np.floor((I.shape[1]-PSF.shape[1])/2))
    PSFbig =np.pad(PSF, ((padx,padx),(pady,pady)), 'constant', constant_values=0)
    PSFbig = np.resize(PSFbig,(I.shape[0],I.shape[1]))
    OTF = np.real(np.fft.fft2(PSFbig))
    Iblur = np.multiply(np.fft.fft2(I),OTF)
    Iblur = np.real(np.fft.ifft2(Iblur))

    #calculate the tapered image as the weighted sum of the blured and raw image
    tapered = np.multiply(I,q)+np.multiply((1-q),Iblur)
    Imax = np.amax(I)
    Imin = np.amin(I)

    # Bound the output by the min and max values of the oroginal image
    tapered[tapered < Imin] = Imin
    tapered[tapered > Imax] = Imax

    return tapered

def drawGauss (std,width):
    """
    Generate a 2D Gaussian. Output is always square
    
    Parameters
    ----------
    std: Standard deviation of gaussian

    width: Width of square output

    Returns
    -------
    arg: Square array with 2D Gaussian function centred about the middle
    """

    width = np.linspace(-width/2,width/2,width) # Genate array of values around centre
    kx, ky = np.meshgrid(width,width) # Generate square arrays with co-ordinates about centre
    kx = np.multiply(kx,kx) # Calculate 2D Gaussian function
    ky = np.multiply(ky,ky)
    arg = np.add(kx,ky)
    arg = np.exp(arg/(2*std*std))
    arg = arg/np.sum(arg)

    return arg;

def getOneDPSF(w,kx):
    
    Xs = np.linspace(-1,1,w)
    PSF = np.sinc(kx*Xs)**2
    return PSF

def getPupil(w,px,NA,wave):
    
    dkxy = (2*np.pi)/(w*px)

    # define the coordiante matrices we will need
    x = np.linspace(-w/2,w/2,w)
    y = np.linspace(-w/2,w/2,w)
    X,Y = np.meshgrid(x,y)
    R = np.sqrt(X**2+Y**2)
    
    # Determine the coefficients form the constants
    dkxy = (2*np.pi)/(w*px)
    kMax = (2*np.pi*NA)/(wave*dkxy)
    
    # And then draw the basic pupil
    pupil = np.zeros((w,w))
    pupil[R<kMax] = 1
    
    # We now need to apodise the pupil to account for the effects of high numerical aperture
    
    
    return pupil

def getSimpleOTF(w,px,NA,wave):
    
    dkxy = (2*np.pi)/(w*px)

    # define the coordiante matrices we will need
    x = np.linspace(-w/2,w/2,w)
    y = np.linspace(-w/2,w/2,w)
    X,Y = np.meshgrid(x,y)
    R = np.sqrt(X**2+Y**2)
    
    # Determine the coefficients form the constants
    dkxy = (2*np.pi)/(w*px)
    kMax = (2*np.pi*NA)/(wave*dkxy)
    
    # And then draw the basic pupil
    pupil = np.zeros((w,w))
    pupil[R<kMax] = 1
    
    PSFc = np.fft.fft2(pupil)
    PSFi = PSFc**2
    OTF = real((np.fft.ifft2(PSFi)))
    
    return OTF

def getOTF(w,px,NA,wave):
    
    dkxy = (2*np.pi)/(w*px)

    # define the coordiante matrices we will need
    x = np.linspace(-w/2,w/2,w)
    y = np.linspace(-w/2,w/2,w)
    X,Y = np.meshgrid(x,y)
    R = np.sqrt(X**2+Y**2)
    
    # Determine the coefficients form the constants
    dkxy = (2*np.pi)/(w*px)
    kMax = (2*np.pi*NA)/(wave*dkxy)
    
    # And then draw the basic pupil
    pupil = np.zeros((w,w))
    pupil[R<kMax] = 1
    
    PSFc = np.fft.fft2(pupil)
    PSFi = PSFc**2
    OTF = real((np.fft.ifft2(PSFi)))
    
    return OTF

def calcDebye(pupil,pmask,width,pol):
    
    Ex    
    PSFc = np.fft.fft2(pupil)
    PSFi = PSFc**2
    OTF = real((np.fft.ifft2(PSFi)))
    
    return OTF
