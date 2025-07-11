import pickle 
import streamlit as st 
import pandas as pd 
import matplotlib.pylab as plt
import seaborn as sns
import sklearn.decomposition 
import sklearn  as PCA
from sklearn.cluster import KMeans

# membaca model
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))

#judul web 
st.title('Data Mining Pridiksi Diabetes')
st.header("isi dataset")

#membaca kolom
col1, col2 = st.columns(2)
with col1:
    Pregnancies = st.text_input('input nilai Pregnancies')
    
with col2 :
    Glucose = st.text_input('input nilai Glucose')
    
with col1 :
    BloodPressuere = st.text_input('input nilai Blood Pressuere')
    
with col2 : 
    SkinThinckness = st.text_input('input nilai Skin Thinckness')
    
with col1 :
    Insulin = st.text_input('input nilai Insulin')

with col2 :
    BMI = st.text_input('input nilai BMI')
    
with col1 :
    DiabetesPedigreeFunction = st.text_input('input nilai Diabetes Pedigree Function ')
    
with col2 :
    Age = st.text_input('input nilai Age')
  
# code untuk prediksi
diab_diagnosis = ''

#membuat tombol untuk prediksi 
if st.button("Test Prediksi Diabetes"):
    diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressuere, SkinThinckness,  Insulin,  BMI,DiabetesPedigreeFunction, Age]])\
        
    if (diab_prediction[0]==1):
        diab_diagnosis = 'Pasien terkena Diabetes'
    else:
        diab_diagnosis = 'Pasien tidak terkena Diabetes'
        
    st.success(diab_diagnosis)
    
    # menampilkan panah elbow
    cluster=[]
    for i in range(1,11):
        km =KMeans(n_clusters=i).fit(x)
        cluster.append(km.inertia_)
        
    fig, ax = plt.subplots(figsize=(12,8))
    sns.lineplot(x=list(range(1,11)), y=cluster, ax=ax)
    ax.set_title('mencari elbow')
    ax.set_xlabel('clusters')
    ax.set_ylabel('inertia')
    ax.annotate('possible elbow point', xy=(3,140000), xytext=(3,50000), xycoords='data',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue',lw=2))

    ax.annotate('possible elbow point', xy=(5,80000), xytext=(5,150000), xycoords='data',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue',lw=2))
    
    st.set_option('deprecation.showPyplotGlobalUe', False)
    elbow_plot = st.pyplot()
    
    st.sidebar.subheader("Nilai jumlah K")
    clust = st.sidebar.slider("Pilih jumlah cluster :", 2,10,3,1)
         
    def k_means(n_clust):
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)
        
        kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans_labels = KMeans.fit_predict(scaled_data)

    plt.figure(figsize=(8, 5))
    plt.scatter(pca_data[:, 0], pca_data[:, 1], c=kmeans_labels, cmap='viridis')
    plt.title("KMeans Clustering (n_clusters=5)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.colorbar()
    plt.show()
