from getopt import getopt
import sys, os, math, time
import numpy
from .aperture_glare_fft import ApertureGlareFFT

numpy.set_printoptions(threshold=sys.maxsize)

helpText = """
python -m fourier-glare

-a <image>
--aperture=<image>
    REQUIRED, unless providing a PSF (see below).
    Path to the image file that depicts the aperture for which you
    want to compute a PSF.

-i <image>
--image=<image>
    OPTIONAL.
    Path to the image file that you want to apply the PSF to as a
    glare function.

-p <image>
--psf=<image>
    OPTIONAL. Skips aperture PSF computation.
    Already-computed PSF you want to apply to an image file.

-t <path>
--tmp=<path>
--temp=<path>
    OPTIONAL.
    Directory where temporary files are stored during image generation.
    If omitted, will use the directory 'tmp' relative to the package.

-h
--help
    Display this help text and exit.
"""

if len(sys.argv[1:]) == 0:
	options = None
else:
	options = getopt(sys.argv[1:], 'a:i:p:t:h', ['aperture=', 'image=', 'psf=', 'temp=', 'tmp=', 'help'])
	options = options[0]
if options is None or len(options) == 0:
	print(helpText)
	sys.exit()
# print(options)
aperturePath = None
psfPath = None
imagePath = None
tempPath = None
for o, a in options:
	if o in ('-h', '--help'):
		print(helpText)
		sys.exit()
	if o in ('-a', '--aperture'):
		if os.path.isfile(os.path.realpath(a)):
			aperturePath = a
		else:
			print("Supplied aperture path was not a valid filename")
			sys.exit()
	if o in ('-i', '--image'):
		if os.path.isfile(a):
			imagePath = a
		else:
			print("Supplied image path was not a valid filename")
			sys.exit()
	if o in ('-p', '--psf'):
		if os.path.isfile(a):
			psfPath = a
		else:
			print("Supplied PSF path was not a valid filename")
			sys.exit()
	if o in ('-t', '--tmp', '--temp'):
		if os.path.isdir(a):
			tempPath = a
		else:
			print("Supplied temporary path was not a valid directory")
			sys.exit()

globalDebug = True
ag = ApertureGlareFFT()

# Set paths
if aperturePath is not None:
	ag.setAperture(aperturePath)
if imagePath is not None:
    ag.setImage(imagePath)
if tempPath is not None:
	ag.setTemp(tempPath)
if psfPath is not None:
	ag.setPSF(psfPath)

# Run initial calculations
ag.prepare()

# Generate the PSF, if we weren't supplied one
if psfPath is None:
	ag.generatePSF(wavelengthStep = 5)

# Apply the PSF to the image, if one was supplied
if imagePath is not None:
	ag.diffract()
