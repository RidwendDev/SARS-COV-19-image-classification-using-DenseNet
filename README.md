# SARS-COV-19-image-classification-using-DenseNet
## **Domain Proyek**
### **Latar Belakang** 
<image src='https://github.com/RidwendDev/SARS-COV-19-image-classification-using-DenseNet/blob/main/image/3670887.jpg'>
Coronavirus disease merupakan suatu kelompok virus yang dapat 
menyebabkan penyakit pada hewan dan manusia. Beberapa jenis coronavirus
diketahui menyebabkan infeksi saluran nafas pada manusia mulai dari batuk pilek 
hingga yang lebih serius seperti Middle East Respiratory Syndrome (MERS) dan 
Severe Acute Respiratory Syndrome (SARS). Covid-19 merupakan penyakit 
menular yang disebabkan oleh jenis coronavirus yang ditemukan pada akhir 2019
(World Health Organization, 2020). Covid-19 yang saat ini mewabah secara global 
di ratusan negara di dunia menjadi salah satu pandemi yang paling banyak menyita 
perhatian dunia. Di masa pandemi saat ini, secara teknis, teknologi Artificial Intelligence (AI) atau kecerdasan buatan sangatlah dibutuhkan, beberapa dirancang untuk mengidentifikasi pola pergerakan penyebaran virus Covid-19 (forecasting). Namun banyak hal lain yang juga dapat dilakukan salah satu diantaranya adalah untuk membantu dalam hal diagnosa, dengan menggunakan perhitungan alogaritma serta melakukan analisa lewat CT Scan.

### **Goals**
Tujuan dibuatnya proyek ini adalah mencoba mengembangkan model deep learning dengan memanfaatkan transfer learning DenseNet untuk pengklasifikasian tiga kelas hasil gambar rontgen dada yang memiliki Covid-19, Viral Pneumonia, atau Normal. Dataset ini tersedia dalam link kaggle berikut ini: `https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset/code?datasetId=627146&sortBy=voteCount`


## **Data Understanding**
Didalam data terdapat 132 gambar yang dibagi menjadi 3 class Covid-19, Viral Pneumonia, atau Normal dimana masing-masing memiliki jumlah sebesar
- Covid-19 : 52
- Viral Pneumonia : 40 
- Normal : 40 <br><br>
Sampel dari gambar hasil rontgen: <br> 
<image src='https://github.com/RidwendDev/SARS-COV-19-image-classification-using-DenseNet/blob/main/image/ct.png'>
   

## Steps to solve problems
- So yang pertama akan dilakukan adalah dengan menginisiasikan jumlah height dan width dengan dimensi 224, 224.
- Berikutnya membangun base model dengan memanfaatkan model pretrain DenseNet, disini weights yang digunakan adalah imagenet. Untuk arsitekturnya seperti berikut ini<br>
   <img src='https://github.com/RidwendDev/SARS-COV-19-image-classification-using-DenseNet/blob/main/image/demse.png'>
- Berikutnya akan didefinisikan sebuah model sequential yang akan menjadi model final untuk DenseNet yang dibangun, setelah ditambahkan pada base model yang saya buat sebelumnya, tambahkan juga layers Average pooling.
- Langkah keempat ialah melakukan Flattening
- Berikutnya akan ditambahkan 2 layer dengan dimensi 256 dan memanfaatkan fungsi aktivasi ReLu
- Disini saya juga menambahkan layer dropout sebesar 0.5, hal ini saya lakukan untuk meregularize yang mana dapat mencegah terjadinya overfitting
   
   * Dapat dilihat dari grafik customer yang churn(pindah) dengan tingkat presentase tertinggi adalah yang menggunakan Elektronik check sebagai Metode Pembayaran.
   * Pelanggan yang memilih mailed check,bank transfer, credit card(automatic) sebagai Metode Pembayaran cenderung tidak pindah.
   
   <img src='https://github.com/RidwendDev/Customer-Churn-Classification/blob/main/Visualizations/depen.png?raw=true'>

   <img src='https://github.com/RidwendDev/Customer-Churn-Classification/blob/main/Visualizations/partner.png?raw=true'>
   
   * Pelanggan yang tidak memiliki dependents(tanggungan) cenderungnya akan churn
   * Pelanggan yang tidak memiliki partners(pasangan) cenderungnya akan churn
   
   <img src='https://github.com/RidwendDev/Customer-Churn-Classification/blob/main/Visualizations/monthlykde.png?raw=true'>
   
   * Terlihat dari grafik bahwa customer dengan monthly charges yang lebih tinggi akan cenderung lebih churn
   
   <img src='https://github.com/RidwendDev/Customer-Churn-Classification/blob/main/Visualizations/tenurekde.png?raw=true'>
   
   * Terlihat dari grafik bahwa customer dengan tenure yang lebih tinggi akan lebih loyal dan sebaliknya pula customer baru akan cenderung lebih churn
   
    
## **Data Preparation**
Berikut merupakan tahapan dalam mempersiapkan data untuk keperluan pelatihan model:
### **Split dataset**
Membagi dataset menjadi data latih (_train_) dan data uji (_test_) merupakan hal yang harus kita lakukan sebelum membuat model.Data latih adalah sekumpulan data yang akan digunakan oleh model untuk melakukan pelatihan. Sedangkan, data uji adalah sekumpulan data yang akan digunakan untuk memvalidasi kinerja pada model yang telah dilatih. Proporsi pembagian yang saya lakukan pada dataset ini menggunakan proporsi pembagian 80:20 yang berarti sebanyak 80% merupakan data latih dan 20% persen merupakan data uji.
   
### **Normalisasi data**
Melakukan transformasi pada data fitur fitur yang akan dipelajari oleh model menggunakan _library_ _StandardScaler_. _StandardScaler_ mentransformasikan fitur dengan menskalakan setiap fitur ke rentang tertentu. _Library_ ini menskalakan dan mentransformasikan setiap fitur secara individual sehingga berada dalam rentang yang diberikan pada set pelatihan, _StandardScaler()_ akan menormalkan fitur-fitur (setiap kolom X) sehingga setiap kolom/fitur/variabel akan memiliki mean = 0 dan standard deviation = 1. Dengan merenapkan teknik normalisasi data, model akan dengan lebih mudah mengenali pola-pola yang terdapat pada data sehingga akan menghasilkan prediksi yang lebih baik daripada tidak menggunakan teknik normalisasi.

## **Modeling**
Algoritma _machine learning_ yang digunakan pada proyek ini yaitu _Support Vector Classifier, K-Nearest Neighbours, Random Forest Classifier_.
### **Support Vector Classifier**
_Support Vector Classifier_ (SVC)berusaha mencari ‘jalan’ terbesar yang bisa memisahkan sampel-sampel dari kelas berbeda, maka pada kasus regresi SVR berusaha mencari jalan yang dapat menampung sebanyak mungkin sampel di ‘jalan’.

Pada tahap ini model akan melakukan pelatihan terhadap data latih untuk mendapatkan error seminimal mungkin, kemudian setelah pelatihan model melakukan prediksi terhadap data yang belum pernah di lihat sebelumnya menggunakan data uji. Adapun algoritma ini memiliki keunggulan dan kekurangan.
Keunggulannya seperti SVC efektif pada data berdimensi tinggi (data dengan jumlah fitur atau atribut yang sangat banyak).Adapun kelemahan dari  Support Vector Classifier: Sulit dipakai dalam problem berskala besar. Skala besar dalam hal ini dimaksudkan dengan jumlah sample yang diolah.
   
### **K-Nearest Neighbours**
K-nearest neighbor adalah salah satu algoritma machine learning dengan pendekatan supervised learning yang bekerja dengan mengkelaskan data baru menggunakan kemiripan antara data baru dengan sejumlah data (k) pada lokasi yang terdekat yang telah tersedia. Algoritma ini menerapkan lazy learning” atau “instant based learning” dan merupakan algoritma non parametrik. Algoritma KNN digunakan untuk klasifikasi dan regresi. Pada pembuatan model ini akan menggunaka modul KNN yang terlah di sediakan oleh library _scikit-learn_ .Pada model ini hanya akan menggunakan 1 parameter yaitu `n_neighbours` (Jumlah tetangga). Jumlah _neighbours_ yang di gunakan yaitu sejumlah 11 neighbours. Kemudian, untuk menentukan titik mana dalam data yang paling mirip dengan input baru, KNN menggunakan perhitungan ukuran jarak. Metrik ukuran jarak yang digunakan secara default pada library sklearn adalah Minkowski distance. Setelah menentukan nilai-nilai pada parameter model melakukan pelatihan menggunakan data latih setelah itu model akan melakukan prediksi terhadap data yang belum pernah dilihat dengan menggunakan data uji. Namun algoritma ini memiliki keunggulan dan kekurangan.

Berikut keunggulan K-Nearest Neighbours:

* Sangat sederhana dan mudah dipahami
* Sangat mudah diterapkan
* Dapat digunakan dalam proses klasifikasi maupun regresi.
* Sangat mudah jika akan dilakukan penambahan data
* Parameter yang diperlukan sedikit, yaitu hanya jumlah tetangga yang dipertimbangkan (K), dan metode perhitungan jaraknya (distance metrik)

Berikut kelemahan K-Nearest Neighbours:

* Perlu menentukan nilai K yang tepat.
* Computation cost yang tinggi
* Waktu pemrosesan yang lama jika datasetnya sangat besar.
* Tidak cukup bagus jika diterapkan pada high dimensional data
* Sangat sensitif pada data yang memiliki banyak noise (noisy data), banyak data yang hilang (missing data), dan pencilan (outliers).

### **Random Forest Classifier**
Algoritma ini merupakan sekumpulan algoritma decision tree. Konsep dasar decision tree adalah mengubah data menjadi aturan-aturan keputusan. Kombinasi dari masing–masing decision tree yang baik kemudian dikombinasikan ke dalam satu model. Random Forest bergantung pada sebuah nilai vector random dengan distribusi yang sama pada semua pohon yang masing masing decision tree memiliki kedalaman yang maksimal. Algoritma ini bisa menyelesaikan permasalahan klasifikasi dan regresi. Pada kasus klasifikasi, prediksi akhir diambil dari prediksi terbanyak pada seluruh pohon. Sedangkan, pada kasus regresi, prediksi akhir adalah rata-rata prediksi seluruh pohon. Untuk pembuatan model Random Forest, akan menggunakan beberapa parameter, antara lain:
* `n_estimator`: jumlah trees (pohon) di forest. Di sini saya set `n_estimator`=500.
Setelah itu model akan melakukan pelatihan menggunakan data latih, setelah itu model bisa melakukan prediksi pada data yang belum pernah diliath dengan menggunakan data uji. Namun model ini sama seperti yang lainnya mempunyai plus minusnya.
   
Keunggulan Random Forest:
* Algoritma Random Forest merupakan algoritma dengan pembelajaran paling akurat yang tersedia. Untuk banyak kumpulan data, algoritma ini menghasilkan pengklasifikasi yang sangat akurat.
* Berjalan secara efisien pada data besar.
* Memiliki metode yang efektif untuk memperkirakan data yang hilang dan menjaga akurasi ketika sebagian besar data hilang.
   
Kelemahan Random Forest:
* Algoritma Random Forest overfiting untuk beberapa kumpulan data dengan tugas klasifikasi/regresi yang noise.

Pada proyek ini yang menjadi model dengan solusi terbaik adalah _Random Forest Classifier_. Dimana model ini memiliki nilai Akurasi paling tinggi dari kedua model lainnya.
   
## **Evaluation**
Pada proyek machine learning ini, didapatkan model terbaik dengan akurasi diatas 80% adalah Random Forest Classifier
   <img src='https://github.com/RidwendDev/Customer-Churn-Classification/blob/main/Visualizations/model.png?raw=true'>
   
Untuk klasifikasi sesudah kita mendapatkan akurasi terbaik kita juga bisa memanfaatkan Confusion matrix
   <img src='https://github.com/RidwendDev/Customer-Churn-Classification/blob/main/Visualizations/cf.png?raw=true'>

Adapun penjelasannya sebagai berikut:
   * True Positive (TP) : Jumlah data yang bernilai Positif dan diprediksi benar sebagai Positif.
   * False Positive (FP) : Jumlah data yang bernilai Negatif tetapi diprediksi sebagai Positif.
   * False Negative (FN) : Jumlah data yang bernilai Positif tetapi diprediksi sebagai Negatif.
   * True Negative (TN) : Jumlah data yang bernilai Negatif dan diprediksi benar sebagai Negatif.
   
Confusion matrix testing model terbaik:<br>
<img src='https://github.com/RidwendDev/Customer-Churn-Classification/blob/main/Visualizations/ev%20rf.png?raw=true'>
   <br>Kesimpulan dari evaluasinya:
   Dari data terlihat bahwasannya model mampu memprediksi data dengan menghasilkan akurasi sebesar 80%, dengan detil tebakan churn yang sebenernya benar churn  adalah 185, tebakan tidak churn yang sebenernya tidak churn adalah 948, tebakan tidak churn yang sebenernya benar churn adalah 189 dan tebakan churn yang sebenernya tidak churn adalah 87.
   
## Referensi
   * Kavitha, V., Kumar, G. H., Kumar, S. M., & Harish, M. (2020). Churn prediction of customer in telecom industry using machine learning algorithms. International Journal of Engineering Research and Technology (IJERT), 9(5), 181–184. https://doi.org/10.17577/ijertv9is050022
   
   
   
   
   
