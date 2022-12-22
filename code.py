import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns 
plt.style.use("ggplot") 
import warnings
warnings.filterwarnings("ignore")
data = pd.read_csv("https://homepage.boku.ac.at/leisch/MSA/datasets/mcdonalds.csv")
print(data.shape)
data.head()


print(data.columns)


data.info()


# Age of the customers

plt.figure(figsize=(20,6))
plt.subplot(1,2,1)
sns.histplot(data.Age)

plt.subplot(1,2,2)
sns.distplot(data.Age);


sns.boxplot(data.Age)


plt.rcParams['figure.figsize'] = (20, 7)
dis = sns.countplot(x=data['Age'])
dis.bar_label(dis.containers[0])
plt.title('Age distribution of the customers')
plt.show()



#Reviews of the customers

lis=[]
for i in data.columns:
    if "Yes" in data[i].values:
        lis.append(i)


ser=pd.Series([(data[i].value_counts()["Yes"]/data.shape[0])*100 for i in lis] , index=lis)
ser
ser.sort_values(ascending=False ,ignore_index=False, inplace=True)
plt.figure(figsize=(10,6))
sns.scatterplot(ser.values, ser.index, marker="o", s=100)
for i in range(ser.shape[0]):
    plt.annotate(round(ser[i],3),(ser.values[i], ser.index[i]))
    
plt.title("Reviews of the customers")

plt.xlim(0,100)




# gender Distribution

size = data["Gender"].value_counts()
plt.pie(size, colors = ["pink","blue"], labels=["female", "male"], shadow = True,explode = [0, 0.1])
plt.legend()

plt.title("Gender Distribution", fontsize=20)




# Customers response over the product

print(data["Like"].unique())
data["Like"]=data["Like"].replace({"I hate it!-5":"-5", "I love it!+5":"+5"})
print(data["Like"].unique())


sns.catplot(x="Like", y="Age",data=data,height=6, aspect=2,kind="swarm")
plt.title('Customers response w.r.t Age');


sns.catplot(x="Age", y="VisitFrequency",data=data,height=6, aspect=2,kind="swarm")
plt.title('Customers visit frequently w.r.t age');
plt.xlim(15,75)


sns.catplot(x="Age", y="VisitFrequency",data=data,height=6, aspect=2,kind="swarm")
plt.title('Customers visit frequently w.r.t age');
plt.xlim(15,75)



#Principal component analysis

x = data.loc[:,lis].values

from sklearn.decomposition import PCA

pca = PCA(n_components=11)
pc = pca.fit_transform(x)
names = ['pca1','pca2','pca3','pca4','pca5','pca6','pca7','pca8','pca9','pca10','pca11']
pca_df = pd.DataFrame(data = pc, columns = names)
pca_df.head()


#Proportion of Variance (from PC1 to PC11)
pca.explained_variance_ratio_


import numpy as np
np.cumsum(pca.explained_variance_ratio_)



# correlation coefficient between original variables and the component

loadings = pca.components_
num_pc = pca.n_features_
pc_list = ["PCA"+str(i) for i in list(range(1, num_pc+1))]
loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
loadings_df['variable'] = data.iloc[:,0:11].columns.values
loadings_df = loadings_df.set_index('variable')
loadings_df


#correlation heatmap

plt.rcParams['figure.figsize'] = (10,7)
ax = sns.heatmap(loadings_df, annot=True, cmap="bwr")



from bioinfokit.visuz import cluster
cluster.screeplot(obj=[pc_list, pca.explained_variance_ratio_],show=True,dim=(8,4))



# get PC scores
pca_scores = PCA().fit_transform(x)
pca_scores



# get 2D biplot
cluster.biplot(cscore=pca_scores, loadings=loadings, labels=data.columns.values, var1=round(pca.explained_variance_ratio_[0]*100, 2),
    var2=round(pca.explained_variance_ratio_[1]*100, 2),show=True,dim=(12,6))



#Extracting segments

plt.rcParams['figure.figsize'] = [10,5]

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0).fit(data.iloc[:,0:11])
data['clust_no'] = kmeans.labels_ #adding to df
print (kmeans.labels_, "\n") #Label assigned 
print (kmeans.inertia_, "\n") 
print(kmeans.n_iter_, "\n") 
print(kmeans.cluster_centers_)



#sizing of the each cluster
from collections import Counter
Counter(kmeans.labels_)



#Visulazing clusters
sns.scatterplot(data=pca_df, x="pca1", y="pca2", hue=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
            marker="X", c="r", s=80, label="centroids")
plt.legend()



