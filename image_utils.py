import math
import numpy, OpenEXR, Imath, cv2

class ImageUtils:
	@staticmethod
	def readEXR(path):
		"""
		Read in an EXR file at the supplied path and return the image data
		as a dictionary of channel data

		Parameters
		----------
		path : string
			Path to read the image from

		Returns
		-------
		dict
			Dictionary of image data, indexed by channel
		"""
		img = OpenEXR.InputFile(path)
		header = img.header()
		dw = header['dataWindow']
		iSize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
		img_data = dict()
		for c in header['channels']:
			img_c = img.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
			img_c = numpy.frombuffer(img_c, dtype=numpy.float32)
			# print(img_c)
			img_c = numpy.reshape(img_c, iSize)
			img_data[c] = img_c
		return img_data

	@staticmethod
	def writeEXR(path, channelData, header = None):
		"""
		Write out an EXR file to the provided path, based on the supplied
		dictionary of per-channel values, and an optional header dictionary

		Parameters
		----------
		path : string
			Where to write out the image
		channelData : dict
			The dictionary containing per-channel data
		header : dict (optional)
			Additional header data to write out
		"""
		if header is None:
			channels = list(channelData.keys())
			shape = channelData[channels[0]].shape
			if (len(shape) == 1):
				header = OpenEXR.Header(1, shape[0])
			else:
				header = OpenEXR.Header(shape[1], shape[0])
			for c in channelData.keys():
				header['channels'][c] = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
		exr = OpenEXR.OutputFile(path, header)
		outputData = dict()
		print("Writing out %s" % path)
		for c in channelData.keys():
			realData = numpy.real(channelData[c])
			# print("\tMax/min for channel %s real values: %s / %s" % (c, numpy.max(realData), numpy.min(realData)))
			try:
				toFloat32 = realData.astype(numpy.float32, casting='unsafe') # Allow to clamp to max value of format
			except RuntimeWarning:
				realData = numpy.clip(realData, a_min=numpy.finfo(numpy.float32).min, a_max=numpy.finfo(numpy.float32).max)
				toFloat32 = realData.astype(numpy.float32, casting='same_kind')
				# print("Failed to cast real data to float32 for channel %s" % c)
				# print(numpy.max(realData))
				# print(numpy.min(realData))
				# exit()
			toBytes = toFloat32.tobytes()
			outputData[c] = toBytes # numpy.real(channelData[c]).astype(numpy.float32, casting='same_kind').tobytes()
		if 'A' not in outputData.keys():
			outputData['A'] = numpy.ones(shape, dtype=numpy.float32).tobytes()
		exr.writePixels(outputData)
		exr.close()
		print("Written")

	@staticmethod
	def pad(channelData, padX, padY):
		"""
		Given a single channel's worth of data and numeric values for
		X and Y axis padding, pad the channel data by the indicated amount.

		If X and Y are fractional values, the floor value will be used for
		the negative dimension and the ceil value will be used for the
		positive.

		Parameters
		----------
		channelData : ndarray
			One channel's worth of image data
		padX : float
			Amount by which to pad in the X direction
		padY : float
			Amount by which to pad in the Y direction

		Returns
		-------
		ndarray
			The padded channel data
		"""
		if (padX == 0 and padY == 0): # Noop
			return channelData
		padX = [int(x) for x in [math.floor(padX), math.ceil(padX)]]
		padY = [int(y) for y in [math.floor(padY), math.ceil(padY)]]
		if (padX[0] < 0 and padY[0] < 0):
			channelData = channelData[-padY[0]:padY[1], -padX[0]:padX[1]]
		elif (padX[0] < 0 and padY[0] >= 0):
			channelData = channelData[:, -padX[0]:padX[1]]
			channelData = numpy.pad(channelData, ((padY[0], padY[1]), (0, 0)))
		elif (padY[0] < 0 and padX[0] >= 0):
			channelData = channelData[-padY[0]:padY[1],:]
			channelData = numpy.pad(channelData, ((0,0), (padX[0], padX[1])))
		else:
			channelData = numpy.pad(channelData, ((padY[0], padY[1]), (padX[0], padX[1])))
		return channelData

	@staticmethod
	def chooseInterpolation(currentSize, targetSize):
		"""
		Given a current size and a target size, choose the best
		interpolation method to use when scaling

		Parameters
		----------
		currentSize | iterable
			The current image size
		targetSize | iterable
			The target image size

		Returns
		-------
		int
			cv2 constant corresponding to the identified interpolation
		"""
		interp = cv2.INTER_CUBIC
		# If both axes are being reduced, switch to AREA
		#if (currentSize[0] > targetSize[0] and currentSize[1] > targetSize[1]):
		#	interp = cv2.INTER_AREA
		return interp

	@staticmethod
	def padAndResize(channelData, padX, padY, sizeX, sizeY, padFirst = True):
		"""
		Given a single channel's data, pad and resize the data based on
		supplied values. An optional parameter indicates whether or not
		to pad first, then resize (default) or to resize first, then pad.

		Parameters
		----------
		channelData : ndarray
			N-D Array of channel data
		padX : float
			Amount by which to pad in the X axis
		padY : float
			Amount by which to pad in the Y axis
		sizeX : float
			Target X-axis size for the returned result
		sizeY : float
			Target Y-axis size for the returned result
		padFirst : bool (optional)
			Whether or not we pad first (default) or resize first

		Returns
		-------
		ndarray
			The resized and padded channel data
		"""
		sizeX = int(sizeX)
		sizeY = int(sizeY)
		if padFirst:
			channelData = ImageUtils.pad(channelData, padX, padY)
			interp = ImageUtils.chooseInterpolation(channelData.shape, (sizeY, sizeX))
			channelData = cv2.resize(channelData, (sizeX, sizeY), interpolation=interp)
		else:
			interp = ImageUtils.chooseInterpolation(channelData.shape, (sizeY, sizeX))
			channelData = cv2.resize(channelData, (sizeX, sizeY), interpolation=interp)
			channelData = ImageUtils.pad(channelData, padX, padY)
		return channelData

	@staticmethod
	def pixel_to_uv(x, dimension):
		"""
		Convert a pixel column/row value to a fractional U/V value
		based on the supplied maximum dimension

		Parameters
		----------
		x : int
			The pixel position along its axis
		dimension : int
			The maximum dimension along the axis

		Returns
		-------
		float
			The U/V fractional value corresponding to this position
		"""
		return (float(x) / float(dimension - 1.) - 0.5) * 2.

	@staticmethod
	def cvToDict(image, includeAlpha = False, normalize = True):
		"""
		Convert a cv2-formatted ndarray of image data into a per-channel
		dictionary of ndarrays

		Parameters
		----------
		image : ndarray
			The image to be converted
		includeAlpha : bool (optional)
			Whether or not to include the Alpha channel in the returned result
		normallize : bool (optional)
			Whether or not to normalize the values (from 0-255 to 0-1)

		Returns
		-------
		dict
			Dictionary of color values
		"""
		channelMapping = {'R': 0, 'G': 1, 'B': 2, 'A': 3}
		dimensions = image.shape[0:2]
		channels = len(image[0][0])
		imDict = {'R': None, 'G': None, 'B': None}
		if (channels == 4 and includeAlpha):
			imDict['A'] = None
		for c in imDict.keys():
			C = image[:,:,channelMapping[c]]
			C = numpy.float64(C)
			imDict[c] = C
			if normalize:
				imDict[c] = numpy.divide(C, 255.0)
			if (type(imDict[c]) is None):
				if c == 'A':
					imDict[c] = numpy.ones(dimensions, dtype=numpy.float64)
				else:
					imDict[c] = numpy.zeros(dimensions, dtype=numpy.float64)
		for c in imDict.keys():
			if imDict[c] is None:
				raise Exception("Channel data should not be none for channel %s" % c)
		return imDict

	@staticmethod
	def dictToCv(imDict):
		"""
		Given an dictionary of per-channel data, convert it to a 3-array
		instead, for use with cv2 functions.

		Parameters
		----------
		imDict : dict
			The dictionary representation of the image to be converted

		Returns
		-------
		ndarray
			The N-D array image representation CV2 wants
		"""
		channels = len(imDict.keys())
		shape = list(list(imDict.values())[0].shape)
		shape.append(channels)
		shape = tuple(shape)
		image = numpy.zeros(shape, dtype=numpy.float32)
		cIdxMap = {'R': 0, 'G': 1, 'B': 2, 'A': 3}
		firstKey = list(imDict.keys())[0]
		for y in range(0, len(imDict[firstKey])):
			for x in range(0, len(imDict[firstKey][y])):
				for c in imDict.keys():
					cIdx = cIdxMap[c]
					image[y][x][cIdx] = imDict[c][y][x]
		return image
