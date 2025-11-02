import os, math
import OpenEXR, Imath, numpy
import imageio.v3 as iio
from .color_utils import ColorUtils
from .image_utils import ImageUtils

import warnings
warnings.filterwarnings("error")

class ApertureGlareFFT:
	SCALE_WORK = 2048
	SCALE_CHROMATIC_BLUR = 256
	SCALE_FRESNEL_KERNEL = 512
	WAVELENGTH_LOW = 380
	WAVELENGTH_HIGH = 770
	WAVELENGTH_MID = 575

	PATH_TMP_FILES = os.path.join(os.path.dirname(__file__), "tmp")
	FILENAME_FRESNEL_KERNEL_CACHE = "E_kernel.exr"
	FILENAME_PSF_PRESCALE = "PSF_prescale.exr"
	FILENAME_PSF_NORMALIZED = "PSF_normalized.exr"
	FILENAME_DIFFRACTION_IMAGE = "diffraction.exr"
	FILENAME_CHROMATIC_PSF_PATTERN = "chromatic_psf_%s.exr"
	DEFAULT_APERTURE_PATH = os.path.join(os.path.dirname(__file__), "apertures/fft_aperture_octround_very-dirty.png")

	# Various setters

	def setAperture(self, aperturePath):
		"""
		Initialize the ApertureGlareFFT with a supplied aperture image

		Parameters
		----------
		aperturePath : str
			Path to the aperture image
		"""
		self.aperturePath = aperturePath

		# Load the aperture file
		print("Loading aperture", end='')
		self.apertureFile = iio.imread(self.aperturePath)
		print("...done")

		# Establish dimensions
		self.aSize = (self.apertureFile.shape[1], self.apertureFile.shape[0])
		if (self.aSize[0] != self.aSize[1]):
			raise Exception("Aperture image must be square")

		# Convert to dictionary for further use
		self.apertureData = ImageUtils.cvToDict(self.apertureFile, normalize = True)

	def setImage(self, imagePath):
		"""
		Set the image path to which we'll apply computed aperture glare

		Parameters
		----------
		imagePath : str
			Path to the base image
		"""
		self.imagePath = imagePath

		# Load the target file
		print("Loading image", end='')
		self.imageFile = OpenEXR.InputFile(self.imagePath)
		print("...done")

		# Establish dimensions
		header = self.imageFile.header()
		dw = header['dataWindow']
		self.iSize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

		# Assemble the data in a format we can read
		self.imageData = dict()
		print("Reading channels", end='')
		for c in header['channels']:
			I = self.imageFile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
			I = numpy.frombuffer(I, dtype=numpy.float32)
			I = numpy.reshape(I, self.iSize)
			I = numpy.float64(I)
			self.imageData[c] = I
		print("...done")

	def setPSF(self, psfPath):
		"""
		Set the path to the already-computed PSF that we want to apply to a base iamge

		Parameters
		----------
		psfPath : str
			Path to the PSF
		"""
		self.psfPath = psfPath

		return

	def setTemp(self, tempPath):
		"""
		Set the temporary output file path

		Parameters
		----------
		tempPath : str
			Path to the temp directory
		"""
		self.PATH_TMP_FILES = tempPath

	# Data Preparation

	def prepare(self):
		"""
		Determine our working dimensions and initialize our helper data
		"""
		self.determineWorkingDimensions()
		if hasattr(self, 'imagePath'):
			self.prepareWorkingImage()
		if hasattr(self, 'aperturePath'):
			self.prepareApertureImage()
			self.prepareFresnelKernel()

	def determineWorkingDimensions(self):
		"""
		Decide on the dimensions of the images we're going to operate on,
		based on any supplied images.
		"""
		# TODO: Allow for user override
		print("Determining working dimensions", end='')
		# Determine maximum dimension of input data
		if hasattr(self, 'aSize') is False:
			self.aSize = [0, 0]
		if hasattr(self, 'iSize') is False:
			self.iSize = [0, ]
		self.SCALE_WORK = max(max(self.aSize), max(self.iSize))

		# Find nearest power of two
		self.SCALE_WORK = 1<<(self.SCALE_WORK-1).bit_length()

		# Set the aperture scale to this value
		self.SCALE_APERTURE = int(self.SCALE_WORK + 0.)

		# Double the actual work scale
		self.SCALE_WORK *= 2
		print("...working in", self.SCALE_WORK)

	def prepareWorkingImage(self):
		"""
		Initialize the padded working image
		"""
		print("Padding and scaling input image", end='')
		self.padX = self.imageData[list(self.imageData.keys())[0]].shape[1] * 0.5
		self.padY = self.imageData[list(self.imageData.keys())[0]].shape[0] * 0.5
		for c in self.imageData.keys():
			self.imageData[c] = ImageUtils.padAndResize(self.imageData[c], self.padX, self.padY, self.SCALE_WORK, self.SCALE_WORK)
		print("...done")

	def prepareApertureImage(self):
		"""
		Initialize the padded aperture image
		"""
		print("Padding and scaling aperture image", end='')
		for c in self.apertureData.keys():
			self.apertureData[c] = ImageUtils.padAndResize(self.apertureData[c], 0, 0, self.SCALE_APERTURE, self.SCALE_APERTURE)
		print("...done")

	def prepareFresnelKernel(self):
		"""
		Prepare the Fresnel kernel helper image, which is used to
		apply an initial distortion to the supplied aperture image.
		"""
		print("Computing E kernel")
		"""
		e^((i*pi)/(l*d)(x^2+y^2))
		e^(iz) = cos(z) + i sin (z)
		z = pi / (l*d) * (x^2+y^2)
		l*d = q
		We want the midpoint to be 1, and the "edge" to have the second period
		ring.

		If x and y are centered such that the midpoint is 0 and the extents are
		+/-1, then z has a range of 0 to pi/q * 2

		Thus, q should have a value around 1/4
		"""
		E = numpy.ndarray((self.SCALE_APERTURE, self.SCALE_APERTURE), dtype=numpy.complex128)
		fresnelKernelPath = os.path.join(os.path.realpath(self.PATH_TMP_FILES), self.FILENAME_FRESNEL_KERNEL_CACHE)
		try:
			raise Exception("Forcing E rebuild") # For debugging
			if os.path.isfile(fresnelKernelPath):
				print("\tLoading E from cache")
				imgE = ImageUtils.readEXR(fresnelKernelPath)
				eSize = imgE[list(imgE.keys())[0]].shape
				if (eSize[0] != self.SCALE_FRESNEL_KERNEL or eSize[1] != self.SCALE_FRESNEL_KERNEL):
					raise Exception("E kernel cache scale does not match work scale")
				for c in ['R', 'G']:
					E_c = ImageUtils.padAndResize(imgE[c], 0, 0, self.SCALE_APERTURE, self.SCALE_APERTURE)
					if (c == 'G'):
						E_c = 1j*E_c # The G channel encodes the imaginary component
					E += E_c
				print("\t...done")
			else:
				raise Exception("E kernel cache file not found")
		except Exception as e:
			print("\t", e)
			print("\tComputing E fresh")
			# l = 575. * 1e-9 # Assume wavelength l is 575 nm
			# d = 8. * 1e-3 # Assume pupil distance d is 8mm
			# d = 8. * 1e10 # Arbitrarily chosen
			ld = 0.25 # Wavelength * distance from aperture to sensor
			w = (1j*math.pi)/(ld)
			E_helper = numpy.ndarray((self.SCALE_FRESNEL_KERNEL, self.SCALE_FRESNEL_KERNEL), dtype=numpy.complex128)
			for y in range(0, self.SCALE_FRESNEL_KERNEL):
				p_y = ImageUtils.pixel_to_uv(y, self.SCALE_FRESNEL_KERNEL)
				for x in range(0, self.SCALE_FRESNEL_KERNEL):
					p_x = ImageUtils.pixel_to_uv(x, self.SCALE_FRESNEL_KERNEL)
					coord = p_x**2 + p_y**2
					E_helper[y][x] = numpy.exp(w * coord)
			print("\t...done")

			print("\tWriting to cache")
			# Make sure we have the directory first
			os.makedirs(os.path.dirname(fresnelKernelPath), exist_ok=True)
			imgE = {"R": numpy.real(E_helper), "G": numpy.imag(E_helper)}
			ImageUtils.writeEXR(fresnelKernelPath, imgE)
			print("\t...done")

			# Scale it up to the aperture image size
			print("\tScaling to work size")
			imgE['R'] = ImageUtils.padAndResize(imgE['R'], 0, 0, self.SCALE_APERTURE, self.SCALE_APERTURE)
			imgE['G'] = ImageUtils.padAndResize(imgE['G'], 0, 0, self.SCALE_APERTURE, self.SCALE_APERTURE)
			E = 1j*imgE['G'] + imgE['R']
			print("\t...done")
		print("...done")
		self.E = E

	# PSF Generation

	def generatePSF(self, wavelengthStep = 5):
		self.mutateAperture()
		self.computeAperturePSF()
		self.chromaticScalePSF(stepSize = wavelengthStep)
		self.normalizePSF()

	def mutateAperture(self):
		"""
		Mutate the supplied aperture image with the Fresnel kernel
		"""
		print("Mutating aperture by E")
		realMutation = {} # For diagnostics
		imagMutation = {} # For diagnostics
		for c in self.apertureData.keys():
			self.apertureData[c] = numpy.multiply(self.apertureData[c], self.E)
			realMutation[c] = self.apertureData[c].real
			imagMutation[c] = self.apertureData[c].imag
		# Write out diagnostic images
		realMutationPath = os.path.join(os.path.realpath(self.PATH_TMP_FILES), "apertureMutation_real.exr")
		imagMutationPath = os.path.join(os.path.realpath(self.PATH_TMP_FILES), "apertureMutation_imag.exr")
		ImageUtils.writeEXR(realMutationPath, realMutation)
		ImageUtils.writeEXR(imagMutationPath, imagMutation)
		print("...done")

	def computeAperturePSF(self):
		"""
		Compute the monochrome PSF for the aperture
		"""
		print("Computing single-wavelength aperture PSF")
		self.psf = dict()
		for c in self.apertureData.keys():
			print("\tGenerating monochromatic FT from channel %s" % c)
			self.psf[c] = numpy.fft.fft2(self.apertureData[c], norm="ortho")
			self.psf[c] = numpy.abs(numpy.fft.fftshift(self.psf[c]))
			self.psf[c] = self.psf[c]**2
			print("\t...done")
		ImageUtils.writeEXR(os.path.join(os.path.realpath(self.PATH_TMP_FILES), "baseAperture.exr"), self.psf)
		print("...done")

	def chromaticScalePSF(self, stepSize = 5):
		"""
		Layer scaled versions of the PSF together at wavelength
		intervals.

		Parameters
		----------
		stepSize : int, optional
			The step between each evaluated wavelength, in nm
			(default is 5)
		"""
		print("Applying chromatic scaling to PSF")
		self.psf_out = dict()
		key_map = {'R': 0, 'G': 1, 'B': 2}
		fake_image = numpy.ndarray((1, 1, 3), dtype=numpy.float32) # Trivial image used to get color values
		numSteps = math.floor((self.WAVELENGTH_HIGH+1 - self.WAVELENGTH_LOW) / stepSize)
		print("\tNum Steps: %s" % numSteps)
		for l in range(self.WAVELENGTH_LOW, self.WAVELENGTH_HIGH+1, stepSize): # Evaluate wavelengths in stepSize nm hops
			print("\tComputing wavelength", l)
			rgb_factors = ColorUtils.get_rgb_from_wavelength(l, fake_image)
			# print("RGB equivalent is", rgb_factors)

			# Determine amount by which to scale the image and then pad or crop it to maintain size
			# TODO: This isn't really the right way to do this; we should be sampling different values based on intensity. This works well enough as a starter, though.
			scale = float(l) / float(self.WAVELENGTH_MID)
			scaleSize = [round(s * scale) for s in self.psf['R'].shape]
			cropAmount = [dim - scale for (dim, scale) in zip(self.psf['R'].shape, scaleSize)]
			cropAmount = [c * 0.5 for c in cropAmount]

			# Iterate over the channels in the PSF
			for c in self.psf.keys():
				if (c not in self.psf_out.keys()):
					self.psf_out[c] = numpy.zeros((self.psf[c].shape[1], self.psf[c].shape[0]), dtype=numpy.complex128)
				self.psf_scaled = ImageUtils.padAndResize(self.psf[c], cropAmount[1], cropAmount[0], scaleSize[1], scaleSize[0], False)
				self.psf_scaled *= rgb_factors[key_map[c]]
				addToPSF = numpy.multiply(self.psf_scaled, 1. / float(l)) # / numSteps # TODO: Should we be scaling by the number of slices here, or are we already intrinsically doing that?
				self.psf_out[c] += addToPSF

				writeOutEXRs = False # Toggle to debug
				if writeOutEXRs:
					exrPath = os.path.join(os.path.realpath(self.PATH_TMP_FILES), self.FILENAME_CHROMATIC_PSF_PATTERN) % l
					if (os.path.isfile(exrPath)):
						existingData = ImageUtils.readEXR(exrPath)
					else:
						existingData = dict()
					existingData[c] = self.psf_scaled
					ImageUtils.writeEXR(exrPath, existingData)
			print("\t...done")
		# DEBUG
		ImageUtils.writeEXR(os.path.join(os.path.realpath(self.PATH_TMP_FILES), 'chromatic_psf.exr'), self.psf_out)
		self.psf = self.psf_out
		print("...done")

	def normalizePSF(self):
		"""
		Normalize the PSF across all channels
		"""
		print("Normalizing chromatic PSF")
		self.normalizePSFbyChannel()
		self.normalizePSFtoUnit()

		psfNormalizedPath = os.path.join(os.path.realpath(self.PATH_TMP_FILES), self.FILENAME_PSF_NORMALIZED)
		ImageUtils.writeEXR(psfNormalizedPath, self.psf)
		print("...done")

	def normalizePSFbyChannel(self):
		"""
		Find the largest value across all channels and scale the other
		channels by so that they each have that value
		"""
		print("\tScaling values")
		psfMax = {c: numpy.max(self.psf[c]) for c in self.psf.keys()}
		psfMaxVal = numpy.amax(list(psfMax.values()))
		psfNormFactor = {c: numpy.divide(psfMaxVal, psfMax[c]) for c in psfMax.keys()}
		for c in self.psf.keys():
			print("\t\tScaling %s by:" % c, psfNormFactor[c])
			self.psf[c] = numpy.multiply(self.psf[c], psfNormFactor[c])
		print("\t...done")

	# EXPERIMENTAL
	def normalizePSFbyChannelMean(self):
		"""
		An experiment normalizing each channel by the largest mean value
		across all channels.
		"""
		print("\tScaling values")
		# psfPrescalePath = os.path.join(os.path.realpath(self.PATH_TMP_FILES), self.FILENAME_PSF_PRESCALE)
		# ImageUtils.writeEXR(psfPrescalePath, self.psf)
		psfMean = {c: numpy.mean(self.psf[c]) for c in self.psf.keys()}
		psfMaxMeanVal = numpy.amax(list(psfMean.values()))
		psfNormFactor = {c: numpy.divide(psfMaxMeanVal, psfMean[c]) for c in psfMean.keys()}
		for c in self.psf.keys():
			print("\t\tScaling %s by:" % c, psfNormFactor[c])
			self.psf[c] = numpy.multiply(self.psf[c], psfNormFactor[c])
		print("\t...done")

	def normalizePSFtoUnit(self):
		"""
		Normalize the PSF to unit values, so as to not introduce
		improper pixel magnitude scaling
		"""
		print("\tNormalizing to unit")
		psfScaleFactor = {c: numpy.max(self.psf[c]) for c in self.psf.keys()}
		psfMaxScale = numpy.amax(list(psfScaleFactor.values()))
		for c in self.psf.keys():
			self.psf[c] = numpy.divide(self.psf[c], psfMaxScale)
		print("\t...done")


	# Image Diffraction

	def diffract(self):
		"""
		Perform our FFTs and generate our diffracted image
		"""
		# TODO: Break this up into smaller methods
		diffData = dict()
		fftRealData = dict()
		fftImaginaryData = dict()
		fftAperture = dict()
		scaledAperture = dict()
		print("Beginning per-channel FFT")
		for c in ['R', 'G', 'B']:
			print("\tWorking on channel %s" % c)

			print("\t\tScaling Aperture Image")
			imageScaleRealPSF = ImageUtils.padAndResize(numpy.real(self.psf[c]), 0, 0, self.SCALE_WORK, self.SCALE_WORK)
			imageScaleImagPSF = ImageUtils.padAndResize(numpy.imag(self.psf[c]), 0, 0, self.SCALE_WORK, self.SCALE_WORK)
			imageScalePSF = imageScaleRealPSF + 1j*imageScaleImagPSF
			scaledAperture[c] = imageScalePSF
			print("\t\t...done")

			print("\t\tComputing FFTs")
			fftAperture[c] = numpy.fft.fft2(imageScalePSF, norm="ortho")
			fftImage = numpy.fft.fft2(self.imageData[c], norm="ortho")
			fftRealData[c] = numpy.real(fftImage)
			fftImaginaryData[c] = numpy.imag(fftImage)
			print("\t\t...done")

			print("\t\tComputing diffracted channel %s" % c)
			diffData[c] = numpy.multiply(fftAperture[c], fftImage)
			diffData[c] = numpy.fft.ifft2(diffData[c], norm="ortho")
			diffData[c] = numpy.fft.fftshift(diffData[c])
			# ifftImageData[c] = numpy.fft.ifft2(fftImage, norm="ortho")
			# ifftImageData[c] = numpy.fft.fftshift(ifftImageData[c])
			print("\t\t...done")

			print("\t...done")
		print("...done computing FFT and IFFT")
		ImageUtils.writeEXR(os.path.join(os.path.realpath(self.PATH_TMP_FILES), "fftAperture_real.exr"), {c: numpy.real(fftAperture[c]) for c in fftAperture})
		ImageUtils.writeEXR(os.path.join(os.path.realpath(self.PATH_TMP_FILES), "fftAperture_imag.exr"), {c: numpy.imag(fftAperture[c]) for c in fftAperture})
		ImageUtils.writeEXR(os.path.join(os.path.realpath(self.PATH_TMP_FILES), "scaledAperture_real.exr"), {c: numpy.real(scaledAperture[c]) for c in scaledAperture})
		ImageUtils.writeEXR(os.path.join(os.path.realpath(self.PATH_TMP_FILES), "scaledAperture_imag.exr"), {c: numpy.imag(scaledAperture[c]) for c in scaledAperture})

		print("Writing diffraction image")
		for c in diffData.keys():
			diffData[c] = numpy.abs(diffData[c])
			diffData[c] = ImageUtils.padAndResize(diffData[c], -self.padX, -self.padY, (self.iSize[1] + self.padX * 2), (self.iSize[0] + self.padY * 2), False)
		diffractionPath = os.path.join(os.path.realpath(self.PATH_TMP_FILES), self.FILENAME_DIFFRACTION_IMAGE)
		ImageUtils.writeEXR(diffractionPath, diffData)
		ImageUtils.writeEXR(os.path.join(os.path.realpath(self.PATH_TMP_FILES), "image-fft_real.exr"), fftRealData);
		ImageUtils.writeEXR(os.path.join(os.path.realpath(self.PATH_TMP_FILES), "image-fft_imag.exr"), fftImaginaryData);
		print("...done")
