import math
import numpy, OpenEXR, Imath, cv2

class ImageUtils:
	"""
	Read in an EXR file at the supplied path and return the image data
	as a dictionary of channel data

	@param	string path
	@return	dict
	"""
	@staticmethod
	def readEXR(path):
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

	"""
	Write out an EXR file to the provided path, based on the supplied
	dictionary of per-channel values, and an optional header dictionary

	@param	string path
	@param	dict channelData
	@param	dict [optional] header
	@return	void
	"""
	@staticmethod
	def writeEXR(path, channelData, header = None):
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
			print("\tMax/min for channel %s real values: %s / %s" % (c, numpy.max(realData), numpy.min(realData)))
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

	"""
	Given a single channel's worth of data and numeric values for
	X and Y axis padding, pad the channel data by the indicated amount.

	If X and Y are fractional values, the floor value will be used for
	the negative dimension and the ceil value will be used for the
	positive.

	@param	ndarray channelData
	@param	float padX
	@param	float padY
	@return	ndarray
	"""
	@staticmethod
	def pad(channelData, padX, padY):
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

	"""
	Given a current size and a target size, choose the best
	interpolation method to use when scaling

	@param	iterable currentSize
	@param	iterable targetSize
	@return	cv2 constant
	"""
	@staticmethod
	def chooseInterpolation(currentSize, targetSize):
		interp = cv2.INTER_CUBIC
		# If both axes are being reduced, switch to AREA
		#if (currentSize[0] > targetSize[0] and currentSize[1] > targetSize[1]):
		#	interp = cv2.INTER_AREA
		return interp

	"""
	Given a single channel's data, pad and resize the data based on
	supplied values. An optional parameter indicates whether or not
	to pad first, then resize (default) or to resize first, then pad.

	@param	ndarray channelData
	@param	float padX
	@param	float padY
	@param	float sizeX
	@param	float sizeY
	@param	bool [optional] padFirst
	@return	ndarray
	"""
	@staticmethod
	def padAndResize(channelData, padX, padY, sizeX, sizeY, padFirst = True):
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
		"""Convert a pixel column/row value to a fractional U/V value
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

	"""
	Convert a cv2-formatted ndarray of image data into a per-channel
	dictionary of ndarrays

	@param	ndarray image
	@param	bool [optional] includeAlpha
	@param	bool [optional] normalize
	@return	dict
	"""
	@staticmethod
	def cvToDict(image, includeAlpha = False, normalize = True):
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

	"""
	Given an dictionary of per-channel data, convert it to a 3-array
	instead, for use with cv2 functions.

	@param	dict imDict
	@return	ndarray
	"""
	@staticmethod
	def dictToCv(imDict):
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


