from operator import itemgetter
import os
import numpy, cv2

class ColorUtils:
	cmf = {}

	# Constant parameters for the piecewise Gaussian functions
	# used in wavelength-to-XYZ color conversion
	cie_g = {
		"x": [
			[599.8, 0.0264, 0.0323],
			[442., 0.0624, 0.0374],
			[501.1, 0.0490, 0.0382]
		],
		"y": [
			[568.8, 0.0213, 0.0247],
			[530.9, 0.0613, 0.0322]
		],
		"z": [
			[437., 0.0845, 0.0278],
			[459., 0.0385, 0.0725]
		]
	}
	# Scalar parameters for the results of the Gaussian functions
	cie_scalar = {
		"x": [1.056, 0.362, -0.065],
		"y": [0.821, 0.286],
		"z": [1.217, 0.681]
	}

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
		# result = {}
		# for axis in ["x", "y", "z"]:
		# 	# Collect computed t_params
		# 	t = [ColorUtils.t_param(axis, l, t_n) for t_n in range(0, len(ColorUtils.cie_g[axis]))]
		# 	result[axis] = ColorUtils.xyz_part(axis, l, *t)
		# x, y, z = itemgetter('x', 'y', 'z')(result)
		# return [x, y, z]
		# print("NEW\t%s\t%s" % (l, [x, y, z]))
		# old = ColorUtils.lamba_to_xyz_old(l)
		# return old
		# print("OLD\t%s\t%s" % (l, old))
		if (len(ColorUtils.cmf.keys()) == 0):
			ColorUtils.loadCMF()
		# print("REF\t%s\t%s" % (l, cmf[l]))
		# exit()
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

	@staticmethod
	def t_param(axis, l, t):
		"""
		Compute a partial component of the matching function

		Parameters
		----------
		axis : string
			The axis (X, Y, Z)
		l : float
			The wavelength
		t : int
			The parameter being computed

		Returns
		-------
		float
			The computed t parameter
		"""
		ref_l = ColorUtils.cie_g[axis][t][0]
		t_val = ColorUtils.cie_g[axis][t][1] if l < ref_l else ColorUtils.cie_g[axis][t][2]
		t_val *= (l - ref_l)
		return t_val

	@staticmethod
	def xyz_part(axis, l, t1, t2, t3=None):
		part_val = 0
		t = [t1, t2, t3]
		for i in range(0, len(t)):
			if t[i] is None:
				continue
			part_val += ColorUtils.cie_scalar[axis][i] * numpy.exp(-0.5 * t[i] * t[i])
		return part_val

	# DEPRECATED
	# Constant parameters for the piecewise Gaussian functions
	# used in wavelength-to-XYZ color conversion
	cie_g_old = {
		"x": [
			[599.8, 37.9, 31.0],
			[442., 16., 26.7],
			[501.1, 20.4, 26.2]
		],
		"y": [
			[568.8, 46.9, 40.5],
			[530.9, 16.3, 31.1]
		],
		"z": [
			[437., 11.8, 36.],
			[459., 26., 13.8]
		]
	}
	# Scalar parameters for the results of the Gaussian functions
	cie_scalar_old = {
		"x": [1.056, 0.362, -0.065],
		"y": [0.821, 0.286],
		"z": [1.217, 0.681]
	}

	# DEPRECATED
	@staticmethod
	def cie_gaussian_old(l, mu, s_1, s_2):
		"""
		Given a wavelength, mu, sigma_1 and sigma_2, return the result
		of the gaussian function.

		Parameters
		----------
		l : int
			The wavelength
		mu : float
			The mu value of the gaussian function
		s_1 : float
			The sigma_1 value of the gaussian function
		s_2 : float
			The sigma_2 value of the gaussian function

		Returns
		-------
		float
			The computed gaussian function
		"""
		l_mu = -0.5 * (l - mu)**2
		if (l < mu):
			return numpy.exp(l_mu / s_1**2)
		else:
			return numpy.exp(l_mu / s_2**2)

	# DEPRECATED
	@staticmethod
	def lamba_to_xyz_old(l):
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
		x = \
				ColorUtils.cie_scalar_old["x"][0] * ColorUtils.cie_gaussian_old(l, *ColorUtils.cie_g_old["x"][0]) + \
				ColorUtils.cie_scalar_old["x"][1] * ColorUtils.cie_gaussian_old(l, *ColorUtils.cie_g_old["x"][1]) + \
				ColorUtils.cie_scalar_old["x"][2] * ColorUtils.cie_gaussian_old(l, *ColorUtils.cie_g_old["x"][2])
		y = \
				ColorUtils.cie_scalar_old["y"][0] * ColorUtils.cie_gaussian_old(l, *ColorUtils.cie_g_old["y"][0]) + \
				ColorUtils.cie_scalar_old["y"][1] * ColorUtils.cie_gaussian_old(l, *ColorUtils.cie_g_old["y"][1])
		z = \
				ColorUtils.cie_scalar_old["z"][0] * ColorUtils.cie_gaussian_old(l, *ColorUtils.cie_g_old["z"][0]) + \
				ColorUtils.cie_scalar_old["z"][1] * ColorUtils.cie_gaussian_old(l, *ColorUtils.cie_g_old["z"][1])
		return [x, y, z]

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

