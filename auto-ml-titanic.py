
# ## Titanic Auto Machine Learning

# Otomatik makine öğrenmesi ile Titanic yarışmasına çözüm üretilecektir.

get_ipython().system('pip install pycaret')


import pycaret
print(pycaret.__version__)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train=pd.read_csv('/kaggle/input/titanic/train.csv')
test=pd.read_csv('/kaggle/input/titanic/test.csv')
sub=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')


# ## EDA

pip install ydata_profiling 


from ydata_profiling import ProfileReport
profile = ProfileReport(train, title = 'Titanic Profile Report')
profile.to_notebook_iframe()


pip install --upgrade numpy xarray pycaret


# ## Pycaret Model Kurulumu

from pycaret import classification


clsf1 = classification.setup(data=train,
                             target='Survived',
                            #numerric_imputation='mean'
                            ignore_features = ['PassengerId','Name','Ticket','Cabin'],
                             normalize =True,
                             session_id=42)


# ## En iyi Modeli Bul

classification.compare_models()


# ## En İyi Model Cross Validation

# Cross validation ile en iyi modeli kullanarak kendimize bir model oluşturuyoruz.

classification_catboost = classification.create_model('catboost')


# ## En İyi Model İçin İnce Ayar (Fine Tuning)

tuned_catboost_model = classification.tune_model(classification_catboost)


# ## En İyi Modelin Parametreleri

print(tuned_catboost_model.get_all_params())


# ## Performance ROC Curve (ROC Preformans Eğrisi)

classification.plot_model(tuned_catboost_model)


# ## Öğrenme Eğrisi 

classification.plot_model(estimator = tuned_catboost_model, plot = 'learning')


# ## Hata Matrisi

# plot confusion matrix
classification.plot_model(tuned_catboost_model, plot = 'confusion_matrix')


classification.plot_model(tuned_catboost_model, plot = 'error')


# ## Özelliklerin Önemleri

classification.plot_model(tuned_catboost_model, plot = 'feature')


# ## Modelin Değerlendirilmesi

classification.evaluate_model(tuned_catboost_model)


# ## Pycaret ile Çoklu Model Uygulama

# Modelleri Tanımlama

gbc = classification.create_model('gbc');
lgbm = classification.create_model('lightgbm');
rf = classification.create_model('rf')

# Modelleri Karıştırma

blend = classification.blend_models(estimator_list=[tuned_catboost_model,gbc,lgbm,rf])


# ## En İyi Modele İnce Ayar Yapma (Fine Tuning)

# Sonuç değişmediği için kullanılmadı
# tuned_blend = classification.tune_model(blend) 


predictions = classification.predict_model(tuned_catboost_model, data=test)
predictions.head()


classification.save_model(tuned_catboost_model, 'titanic_model')


sub['Survived'] = round(predictions['prediction_label']).astype(int)
sub.to_csv('submission.csv',index=False)
sub.head(10)



