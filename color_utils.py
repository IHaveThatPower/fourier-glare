import numpy, cv2

class ColorUtils:
	@staticmethod
	def cie_gaussian(l, mu, s_1, s_2):
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
		x = 1.056 * ColorUtils.cie_gaussian(l, 599.8, 37.9, 31.0) + 0.362 * ColorUtils.cie_gaussian(l, 442., 16., 26.7) - 0.065 * ColorUtils.cie_gaussian(l, 501.1, 20.4, 26.2)
		y = 0.821 * ColorUtils.cie_gaussian(l, 568.8, 46.9, 40.5) + 0.286 * ColorUtils.cie_gaussian(l, 530.9, 16.3, 31.1)
		z = 1.217 * ColorUtils.cie_gaussian(l, 437., 11.8, 36.) + 0.681 * ColorUtils.cie_gaussian(l, 459., 26., 13.8)
		return [x, y, z]

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
	
