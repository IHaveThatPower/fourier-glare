from operator import itemgetter
import os
import numpy, cv2

class ColorUtils:
	cmf = {}

	@staticmethod
	def get_rgb_from_wavelength(l, fake_image):
		"""
		Given a wavelength and a 1x1x3 ndarray, determine the XYZ color
		value for that wavelength, assign those values to the ndarray,
		then ask cv2 to convert from XYZ to RGB for us, returning the
		RGB values.

		Parameters
		----------
		l : int
			The wavelength to convert
		fake_image : ndarray
			The 1x1x3 container we'll use for color interchange

		Returns
		-------
		list
			A 3-value list of RGB factors
		"""
		xyz = ColorUtils.lamba_to_xyz(l)
		fake_image[0][0] = xyz
		rgb = cv2.cvtColor(fake_image, cv2.COLOR_XYZ2RGB)
		return rgb[0][0]

	@staticmethod
	def lamba_to_xyz(l):
		"""
		Given a wavelength, convert it to XYZ colorspace.

		Parameters
		----------
		l : float
			The wavelength

		Returns
		-------
		list
			a list of XYZ values corresponding to the given wavelength
		"""
		if (len(ColorUtils.cmf.keys()) == 0):
			ColorUtils.loadCMF()
		return ColorUtils.cmf[l]

	@staticmethod
	def loadCMF():
		"""
		Load the wavelength-to-XYZ values from the library reference
		"""
		ref_file = open(os.path.join(os.path.dirname(__file__), "wave_to_xyz_color_matching.txt"))
		ref_data = ref_file.read()
		ref_file.close()
		ColorUtils.cmf = {}
		for i, line in enumerate(ref_data.split("\n")):
			if i == 0: continue # Skip header
			if line == "": continue # Skip empty strings
			line_data = [numpy.float64(l) for l in line.split("\t")]
			ColorUtils.cmf[int(line_data[0])] = list(line_data[1:])

	# CURRENTLY UNUSED
	@staticmethod
	def lambda_from_xyz(xyz):
		"""
		Given a value in XYZ colorspace, compute a best approximation of
		the wavelength corresponding to the indicated color.

		Parameters
		----------
		xyz : list
			The XYZ colorspace values

		Returns
		-------
		float
			The wavelength estimate
		"""
		x = xyz[0]
		y = xyz[1]
		z = xyz[2]

		# Based on Z
		# e^(0.051 * sqrt(-2 * ln(z / 1.839)) + ln(449.8)) = L
		lam_z = numpy.exp(0.051 * numpy.sqrt(-2 * numpy.log(z / 1.839)) + numpy.log(449.8))

		# Based on Y
		# e^(0.051 * sqrt(-2 * ln(y / 1.014)) + ln(449.8)) = L
		lam_y = numpy.exp(0.051 * numpy.sqrt(-2 * numpy.log(y / 1.014)) + numpy.log(449.8))

		return (lam_z + lam_y) * 0.5

