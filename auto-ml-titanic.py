#!/usr/bin/env python
# coding: utf-8

# ![image.png](attachment:9a91a790-0115-43a5-8226-a9bcc20919aa.png)

# ## Titanic Auto Machine Learning

# Otomatik makine öğrenmesi ile Titanic yarışmasına çözüm üretilecektir.

# In[1]:


get_ipython().system('pip install pycaret')


# In[2]:


import pycaret
print(pycaret.__version__)


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


train=pd.read_csv('/kaggle/input/titanic/train.csv')
test=pd.read_csv('/kaggle/input/titanic/test.csv')
sub=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')


# ## EDA

# In[5]:


pip install ydata_profiling 


# In[6]:


from ydata_profiling import ProfileReport
profile = ProfileReport(train, title = 'Titanic Profile Report')
profile.to_notebook_iframe()


# In[7]:


pip install --upgrade numpy xarray pycaret


# ## Pycaret Model Kurulumu

# In[8]:


from pycaret import classification


# In[9]:


clsf1 = classification.setup(data=train,
                             target='Survived',
                            #numerric_imputation='mean'
                            ignore_features = ['PassengerId','Name','Ticket','Cabin'],
                             normalize =True,
                             session_id=42)


# ## En iyi Modeli Bul

# In[10]:


classification.compare_models()


# ## En İyi Model Cross Validation

# Cross validation ile en iyi modeli kullanarak kendimize bir model oluşturuyoruz.

# In[11]:


classification_catboost = classification.create_model('catboost')


# ## En İyi Model İçin İnce Ayar (Fine Tuning)

# In[12]:


tuned_catboost_model = classification.tune_model(classification_catboost)


# ## En İyi Modelin Parametreleri

# In[13]:


print(tuned_catboost_model.get_all_params())


# ## Performance ROC Curve (ROC Preformans Eğrisi)

# In[14]:


classification.plot_model(tuned_catboost_model)


# ## Öğrenme Eğrisi 

# In[15]:


classification.plot_model(estimator = tuned_catboost_model, plot = 'learning')


# ## Hata Matrisi

# In[16]:


# plot confusion matrix
classification.plot_model(tuned_catboost_model, plot = 'confusion_matrix')


# In[17]:


classification.plot_model(tuned_catboost_model, plot = 'error')


# ## Özelliklerin Önemleri

# In[18]:


classification.plot_model(tuned_catboost_model, plot = 'feature')


# ## Modelin Değerlendirilmesi

# In[19]:


classification.evaluate_model(tuned_catboost_model)


# ## Pycaret ile Çoklu Model Uygulama

# In[20]:


# Modelleri Tanımlama

gbc = classification.create_model('gbc');
lgbm = classification.create_model('lightgbm');
rf = classification.create_model('rf')

# Modelleri Karıştırma

blend = classification.blend_models(estimator_list=[tuned_catboost_model,gbc,lgbm,rf])


# ## En İyi Modele İnce Ayar Yapma (Fine Tuning)

# In[21]:


# Sonuç değişmediği için kullanılmadı
# tuned_blend = classification.tune_model(blend) 


# In[22]:


predictions = classification.predict_model(tuned_catboost_model, data=test)
predictions.head()


# In[23]:


classification.save_model(tuned_catboost_model, 'titanic_model')


# In[24]:


sub['Survived'] = round(predictions['prediction_label']).astype(int)
sub.to_csv('submission.csv',index=False)
sub.head(10)


# In[ ]:




