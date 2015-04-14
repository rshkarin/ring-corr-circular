import PIL
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pylab

COLOR_MAP = 'Greys_r'

def main():
	slice_path = "tomo_0227.tif"
	image_slice = Image.open(slice_path)
	slice_array = np.array(image_slice)
	n_angles = 1700
	angle_step = np.pi / float(n_angles)
	

	f = pylab.figure()

	filteredImage = filterRings(slice_array, n_angles, angle_step)

	f.add_subplot(2, 1, 1)
	#plt.figure("Original slice")
	pylab.imshow(slice_array, cmap=plt.get_cmap(COLOR_MAP))
	#plt.show(1)

	f.add_subplot(2, 1, 2)
	#plt.figure("Filtered rings")
	pylab.imshow(filteredImage, cmap=plt.get_cmap(COLOR_MAP))
	#plt.show(1)

	pylab.show()


def bilinear_interp(im, x, y):
	# http://stackoverflow.com/questions/12729228/simple-efficient-bilinear-interpolation-of-images-in-numpy-and-python
	x = np.asarray(x)
	y = np.asarray(y)
	x0 = np.floor(x).astype(int)
	x1 = x0 + 1
	y0 = np.floor(y).astype(int)
	y1 = y0 + 1
	x0 = np.clip(x0, 0, im.shape[1]-1);
	x1 = np.clip(x1, 0, im.shape[1]-1);
	y0 = np.clip(y0, 0, im.shape[0]-1);
	y1 = np.clip(y1, 0, im.shape[0]-1);
	Ia = im[ y0, x0 ]
	Ib = im[ y1, x0 ]
	Ic = im[ y0, x1 ]
	Id = im[ y1, x1 ]
	wa = (x1-x) * (y1-y)
	wb = (x1-x) * (y-y0)
	wc = (x-x0) * (y1-y)
	wd = (x-x0) * (y-y0)
	return wa*Ia + wb*Ib + wc*Ic + wd*Id

def filterRings(image, thetaLen, angleStepRad):
	rhoLen = np.size(image, 0)

	# Allocate result array
	fourierAngleData = np.zeros((thetaLen))
	fourierAngleIdxX = np.zeros((thetaLen), dtype=int)
	fourierAngleIdxY = np.zeros((thetaLen), dtype=int)

	# Find center of the grid
	center = rhoLen / 2 - 1

	# Filtered image
	filteredImage = np.zeros_like(image)
	filteredImageOnes = np.zeros_like(image)

	# FFP
	print rhoLen, thetaLen

	for rho in range(rhoLen):
		for thetaId in range(thetaLen):
			sradius = center - rho # signed radius

			theta = thetaId * angleStepRad

			# Compute grid coordinates [0 ... rhoLen]
			gx = center - sradius * np.cos(theta)
			gy = center - sradius * np.sin(theta)

			# Take the values and indices
			#fourierAngleData[thetaId] = image[int(gy)][int(gx)] #bilinear_interp(image, gx, gy)
			fourierAngleData[thetaId] = bilinear_interp(image, gx, gy)
			fourierAngleIdxX[thetaId] = int(np.ceil(gx))
			fourierAngleIdxY[thetaId] = int(np.ceil(gy))

		# Take FFT of angular strip
		fourierData = np.fft.fft(fourierAngleData)
		
		fourierData[0] = 0.0
		fourierData[1] = 0.0
		#fourierData[2] = 0.0
		fourierData[len(fourierData) - 1] = 0.0
		fourierData[len(fourierData) - 2] = 0.0
		#fourierData[len(fourierData) - 3] = 0.0
		
		filteredData = np.fft.ifft(fourierData)
		
		# Put filtered data on the slice
		for idx in range(thetaLen):
			#filteredImage[fourierAngleIdxY[thetaId]][fourierAngleIdxX[thetaId]] = np.real(filteredData2[i])
			x = fourierAngleIdxX[idx]
			y = fourierAngleIdxY[idx]

			filteredImage[y][x] = np.real(filteredData[idx])
		
	return filteredImage

if __name__ == "__main__":
	main()

