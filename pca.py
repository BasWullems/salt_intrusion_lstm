# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 14:45:53 2022

@author: wullems
"""
from sklearn.decomposition import PCA

pca = PCA(n_components = 30)

pca.fit(trainIn)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
principal_components = pca.fit_transform(trainIn)
principalDF = pd.DataFrame(data = principal_components, columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12','PC13','PC14','PC15','PC16','PC17','PC18','PC19','PC20','PC21','PC22','PC23','PC24','PC25','PC26','PC27','PC28','PC29','PC30'])
print('Explained variation per principal component:{}'.format(pca.explained_variance_ratio_))
A= pca.components_
AllVars =[]
Timesteps = []

for j in range(1,n_past+1):
    for i in range(0,30):    
        AllVars.append(Vars[i])
        Timesteps.append(j-n_past)
for j in range(n_past+1,n_past+2):
    for i in range(15,30):    
        AllVars.append(Vars[i])
        Timesteps.append(j-n_past)


FeatureImportance = pd.DataFrame(data={'Variable':AllVars, 'Timestep':Timesteps, 'PC1':abs(A[0,:]),'PC2':abs(A[1,:]),'PC3':abs(A[2,:]),'PC4':abs(A[3,:])})
PC1 = []

PC2 = []

PC3 = []

PC4= []

for i in range(len(Vars)):
    val1=0
    val2=0
    val3=0
    val4=0
    var = Vars[i]
    for j in range(len(BundledImportance['Variable'])):
        if FeatureImportance['Variable'][j]==var:
            val1 += FeatureImportance['PC1'][j]
            val2 += FeatureImportance['PC2'][j]
            val3 += FeatureImportance['PC3'][j]
            val4 += FeatureImportance['PC4'][j]
    PC1.append(val1)
    PC2.append(val2)
    PC3.append(val3)
    PC4.append(val4)
    
PC_tot = []
    
for i in range(len(BundledImportance['Variable'])):
    tot = PC1[i]+PC2[i]+PC3[i]+PC4[i]
    PC_tot.append(tot)
    
PC_max = []

for i in range(len(BundledImportance['Variable'])):
    maxi = max([PC1[i],PC2[i],PC3[i],PC4[i]])
    PC_max.append(maxi)
    
BundledImportance = pd.DataFrame(data={'Variable':Vars, 'PC1':PC1,'PC2':PC2,'PC3':PC3,'PC4':PC4,'PC_tot':PC_tot,'PC_max':PC_max})