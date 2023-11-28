import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage import io, color, img_as_ubyte
import cv2

# GLCM properties
def contrast_feature(matrix_coocurrence):
	contrast = graycoprops(matrix_coocurrence, 'contrast')
	return contrast

def dissimilarity_feature(matrix_coocurrence):
	dissimilarity = graycoprops(matrix_coocurrence, 'dissimilarity')	
	return dissimilarity

def homogeneity_feature(matrix_coocurrence):
	homogeneity = graycoprops(matrix_coocurrence, 'homogeneity')
	return homogeneity

def energy_feature(matrix_coocurrence):
	energy = graycoprops(matrix_coocurrence, 'energy')
	return energy

def correlation_feature(matrix_coocurrence):
	correlation = graycoprops(matrix_coocurrence, 'correlation')
	return correlation

def asm_feature(matrix_coocurrence):
	asm = graycoprops(matrix_coocurrence, 'ASM')
	return asm

class GLCM:
	def feature(pathfile):
		img = io.imread(pathfile)
		#Convert to Greyscale
		gray = color.rgb2gray(img)
		image_name_gray = 'hasil_grayscale.png'
		output_path_gray = '/Applications/XAMPP/xamppfiles/htdocs/cauliflower/asset/'+image_name_gray
		cv2.imwrite(output_path_gray, (gray * 255).astype('uint8'))
		image = img_as_ubyte(gray)
		bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
		inds = np.digitize(image, bins)
		max_value = inds.max()+1
		matrix_coocurrence = graycomatrix(inds, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=max_value, normed=False, symmetric=False)	
		(con, dis, hom, eng, corr, asm) = (contrast_feature(matrix_coocurrence), dissimilarity_feature(matrix_coocurrence), homogeneity_feature(matrix_coocurrence), energy_feature(matrix_coocurrence), correlation_feature(matrix_coocurrence), asm_feature(matrix_coocurrence))
		arr_glcm = np.hstack((con,dis,hom,eng,corr,asm))

		return arr_glcm
