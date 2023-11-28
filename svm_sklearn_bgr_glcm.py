from sklearn import preprocessing
from sklearn.svm import SVC
import pandas as pd
from sklearn import preprocessing

class svm_sklearn:
    def classification(arr_predict):
        # read data training
        df = pd.read_csv(r'/Users/syahrindradr/Documents/dzaky/VSCode/SKRIPSI/TA/modellingcsv/modelling_glcm_rgb_hsv500.csv')
        # get unique label
        df['label'].unique()
        df['label'] = df['label'].map({'Cauliflower_No_disease' :0, 'Cauliflower_Black_rot' :1, 'Cauliflower_Downy_mildew' :2}).astype(int) #mapping numbers
        # drop label on data train
        x_data = df.drop(['label'], axis=1)
        y_data = df['label']

        # get min and max (optional)
        MinMaxScaler = preprocessing.MinMaxScaler()
        X_data_minmax = MinMaxScaler.fit_transform(x_data)
        data = pd.DataFrame(X_data_minmax, columns=['R','G','B','A','H','S','V','contrast_0','contrast_45','contrast_90','contrast_135','disimilarity_0','disimilarity_45','disimilarity_90','disimilarity_135','homogenity_0','homogenity_45','homogenity_90','homogenity_135','energy_0','energy_45','energy_90','energy_135','correlation_0','correlation_45','correlation_90','correlation_135','asm_0','asm_45','asm_90','asm_135'])
        
        # get classification
        model_SVC = SVC(C=1,gamma=0.001, kernel='poly')
        model_SVC.fit(x_data, y_data) 
        y_pred=model_SVC.predict(arr_predict)

        # Print results
        print("Hasil prediksi:", y_pred)
        
        return y_pred

    
