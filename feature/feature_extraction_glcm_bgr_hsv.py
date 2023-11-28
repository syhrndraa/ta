import numpy as np
from rembg import remove
from PIL import Image
import cv2

# myclass
import sys
sys.path.append('/Users/syahrindradr/Documents/dzaky/VSCode/SKRIPSI/TA')
import svm_sklearn_bgr_glcm as clsprediction
import feature_glcm as fglcm
import feature_bgr as fbgr

class testing:
    def try_classification(file_path):
        inputimage = Image.open(file_path) # load image
        new_size = (300, 300) #resize image
        resized_image = inputimage.resize(new_size, Image.ANTIALIAS)
        image_name_resize = 'hasil_resize.png'
        output_path_resize ='/Applications/XAMPP/xamppfiles/htdocs/cauliflower/asset/'+image_name_resize
        resized_image.save(output_path_resize)
        arr_glcm=fglcm.GLCM.feature(output_path_resize)
        output_img = remove(resized_image) #remove background
        image_name_rembg = 'hasil_rembg.png'
        output_path_rembg ='/Applications/XAMPP/xamppfiles/htdocs/cauliflower/asset/'+image_name_rembg
        output_img.save(output_path_rembg) # save image
        arr_bgr=fbgr.BGR.feature(Image.open(output_path_rembg))
        myimg = cv2.imread(output_path_rembg)
        image_name_hsv = 'hasil_hsv.png'
        output_path_hsv = '/Applications/XAMPP/xamppfiles/htdocs/cauliflower/asset/'+image_name_hsv
        output_hsv = cv2.cvtColor(myimg, cv2.COLOR_BGR2HSV)
        cv2.imwrite(output_path_hsv, output_hsv)
        arr_hsv = fbgr.BGR.feature(Image.open(output_path_hsv))
        arr=np.hstack([[arr_bgr],[arr_hsv],arr_glcm])
        print(arr)
        output=clsprediction.svm_sklearn.classification(arr)

        if(output[0]==0):
            print("Cauliflower No Disease")
            return "Cauliflower No Disease", 'asset/'+image_name_rembg,'asset/'+image_name_hsv, arr
        elif(output[0]==1):
            print("Cauliflower Black Rot")
            return "Cauliflower Black Rot",'asset/'+image_name_rembg,'asset/'+image_name_hsv, arr
        elif(output[0]==2):
            print("Cauliflower Downy Mildew")
            return "Cauliflower Downy Mildew",'asset/'+image_name_rembg,'asset/'+image_name_hsv, arr
            
# testing.try_classification("C:/xampp/htdocs/tomatoclass/asset/Tomato_mosaic_virus2.png")