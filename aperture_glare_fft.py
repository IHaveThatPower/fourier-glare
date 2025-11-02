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

	REF_WAVELENGTH_TIMES_APERTURE_DISTANCE = 0.25 # Used in computing the E-kernel and the K scaling factor

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
			# See docblock above about the "q" value, computed here as ld
			# l = 575. * 1e-9 # Assume wavelength l is 575 nm
			# d = 8. * 1e-3 # Assume pupil distance d is 8mm
			# d = 8. * 1e10 # Arbitrarily chosen
			# TODO: Explore this in more detail. Why are the physical values so unintuitive?
			ld = ApertureGlareFFT.REF_WAVELENGTH_TIMES_APERTURE_DISTANCE # Wavelength * distance from aperture to sensor
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
		"""
		Generate the PSF from the aperture

		Parameters
		----------
		wavelengthStep : int
			Number of steps to use when computing per-wavelength PSFs
		"""
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
			self.psf[c] = self.psf[c]**2 * (1. / ApertureGlareFFT.REF_WAVELENGTH_TIMES_APERTURE_DISTANCE**2) # This is the "K" scale factor
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
		numSteps = math.floor((self.WAVELENGTH_HIGH+1 - self.WAVELENGTH_LOW) / stepSize)
		print("\tNum Steps: %s" % numSteps)
		for l in range(self.WAVELENGTH_LOW, self.WAVELENGTH_HIGH+1, stepSize): # Evaluate wavelengths in stepSize nm hops
			print("\tComputing wavelength %snm" % (l))
			# Wavelength-based scaling factor
			# TODO: According to Ritschel 2009, this should be 575 / l, but that results in blue being the "outer" value,
			# which doesn't match their actual results. Flipping it to l / 575 results in expected results. Why?
			scale_factor = float(l) / float(self.WAVELENGTH_MID) # float(self.WAVELENGTH_MID) / float(l)
			# PSF dimensions when scaled by this factor
			scale_dim = [round(s * scale_factor) for s in self.psf['R'].shape]
			# Amount by which to crop resulting scaled image to keep it the same size as the base PSF size
			crop_amount = [(dim - scale) * 0.5 for (dim, scale) in zip(self.psf['R'].shape, scale_dim)]
			# Spectral factors for this wavelength, as RGB values
			spectral_rgb = ColorUtils.get_rgb_from_wavelength(l)

			# Apply to each channel in the base PSF
			for c in self.psf.keys():
				if (c not in self.psf_out.keys()): # Initialize the channel
					self.psf_out[c] = numpy.zeros((self.psf[c].shape[1], self.psf[c].shape[0]), dtype=numpy.complex128)
				self.psf_scaled = ImageUtils.padAndResize(self.psf[c], crop_amount[1], crop_amount[0], scale_dim[1], scale_dim[0], False)
				self.psf_scaled *= spectral_rgb[c]
				self.psf_out[c] += self.psf_scaled
		# DEBUG
		ImageUtils.writeEXR(os.path.join(os.path.realpath(self.PATH_TMP_FILES), 'chromatic_psf.exr'), self.psf_out)
		self.psf = self.psf_out
		print("...done")

	def normalizePSF(self):
		"""
		Normalize the PSF across all channels
		"""
		print("Normalizing chromatic PSF")
		self.normalizePSFChannelsToUnit()
		psfNormalizedPath = os.path.join(os.path.realpath(self.PATH_TMP_FILES), self.FILENAME_PSF_NORMALIZED)
		ImageUtils.writeEXR(psfNormalizedPath, self.psf)
		print("...done")

	def normalizePSFChannelsToUnit(self):
		"""
		Compute the total value of each channel in the PSF and scale
		it down so that it equals 1
		"""
		for c in self.psf.keys():
			psf_sum = numpy.sum(self.psf[c])
			psf_unit = self.psf[c].shape[1] * self.psf[c].shape[0]
			psf_scale = psf_unit / psf_sum
			print("\tScaling PSF %s-channel by %s" % (c, psf_scale))
			self.psf[c] *= psf_scale
			print("\t...done")

	# Image Diffraction

	def diffract(self):
		"""
		Perform our FFTs and generate our diffracted image
		"""
		diffData = dict()
		fftRealData = dict()
		fftImaginaryData = dict()
		fftAperture = dict()
		scaledAperture = dict()
		print("Beginning per-channel FFT")
		for c in ['R', 'G', 'B']:
			print("\tWorking on channel %s" % c)
			scaledAperture[c] = self.scaleApertureForDiffraction(self.psf[c])

			print("\t\tComputing FFTs")
			fftAperture[c] = numpy.fft.fft2(scaledAperture[c], norm="ortho")
			fftImage = numpy.fft.fft2(self.imageData[c], norm="ortho")
			fftRealData[c] = numpy.real(fftImage)
			fftImaginaryData[c] = numpy.imag(fftImage)
			print("\t\t...done")

			diffData[c] = self.computeDiffractedChannel(fftAperture[c], fftImage)
			print("\t...done")
		print("...done computing FFT and IFFT")
		ImageUtils.writeEXR(os.path.join(os.path.realpath(self.PATH_TMP_FILES), "fftAperture_real.exr"), {c: numpy.real(fftAperture[c]) for c in fftAperture})
		ImageUtils.writeEXR(os.path.join(os.path.realpath(self.PATH_TMP_FILES), "fftAperture_imag.exr"), {c: numpy.imag(fftAperture[c]) for c in fftAperture})
		ImageUtils.writeEXR(os.path.join(os.path.realpath(self.PATH_TMP_FILES), "scaledAperture_real.exr"), {c: numpy.real(scaledAperture[c]) for c in scaledAperture})
		ImageUtils.writeEXR(os.path.join(os.path.realpath(self.PATH_TMP_FILES), "scaledAperture_imag.exr"), {c: numpy.imag(scaledAperture[c]) for c in scaledAperture})

		print("Writing diffraction image")
		diffractionPath = os.path.join(os.path.realpath(self.PATH_TMP_FILES), self.FILENAME_DIFFRACTION_IMAGE)
		ImageUtils.writeEXR(diffractionPath, diffData)
		ImageUtils.writeEXR(os.path.join(os.path.realpath(self.PATH_TMP_FILES), "image-fft_real.exr"), fftRealData);
		ImageUtils.writeEXR(os.path.join(os.path.realpath(self.PATH_TMP_FILES), "image-fft_imag.exr"), fftImaginaryData);
		print("...done")

	def scaleApertureForDiffraction(self, channel_psf):
		"""
		Scale the aperture image prior to diffracting it

		Returns
		-------
		ndarray
			Scaled N-D array equal to the work scale, from the PSF for the channel
		"""
		print("\t\tScaling Aperture Image")
		imageScaleRealPSF = ImageUtils.padAndResize(numpy.real(channel_psf), 0, 0, self.SCALE_WORK, self.SCALE_WORK)
		imageScaleImagPSF = ImageUtils.padAndResize(numpy.imag(channel_psf), 0, 0, self.SCALE_WORK, self.SCALE_WORK)
		imageScalePSF = imageScaleRealPSF + 1j*imageScaleImagPSF
		return imageScalePSF
		print("\t\t...done")

	def computeDiffractedChannel(self, fftAperture, fftImage):
		"""
		Convolve an image by using a supplied FFT of the image and
		a supplied FFT of the convolution PSF

		Parameters
		----------
		fftAperture : ndarray
			The PSF FFT
		fftImage : ndarray
			The image FFT

		Returns
		-------
		ndarray
			The convolved channel data
		"""
		print("\t\tComputing diffracted channel %s" % c)
		# Multiply FFT of image by FFT of aperture
		diffData = numpy.multiply(fftAperture, fftImage)
		# IFFT & FFT Shift
		diffData = numpy.fft.ifft2(diffData, norm="ortho")
		diffData = numpy.fft.fftshift(diffData)

		# Take the absolute value of the result
		diffData = numpy.abs(diffData)

		# Scale the resulting data back down toe the original size
		diffData = ImageUtils.padAndResize(diffData, -self.padX, -self.padY, (self.iSize[1] + self.padX * 2), (self.iSize[0] + self.padY * 2), False)

		print("\t\t...done")
		return diffData
