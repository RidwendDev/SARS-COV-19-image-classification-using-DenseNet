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
- Lalu tambahkan layer terakhir dengan dimensi 3 dan fungsi aktivasi softmax untuk melakukan Multiclass
- Berikutnya akan diberikan loss function Sparse Categorical Crossentropy, dan memberikan optimizer Adaptive Momentum(Adam) serta menggunakan acc sebagai metrics

   <img src='https://github.com/RidwendDev/SARS-COV-19-image-classification-using-DenseNet/blob/main/image/ars1.png'>
- Berikut merupakan gambaran arsitektur yang sudah dibangun

   <img src='https://github.com/RidwendDev/SARS-COV-19-image-classification-using-DenseNet/blob/main/image/ars2.png'>
- Langkah berikutnya adalah membuat sebuah checkpoint dari model dengan memanfaatkan callbacks pada keras dan memberikan parameter-parameter seperti file yang akan disave, mode dan lainnya
-  Setelah itu saya memanfaatkan early stopping untuk memonitor nilai acc yang dihasilkan saat proses training berlangsung, disini saya memberikan patience sebesar 10
   <img src='https://github.com/RidwendDev/SARS-COV-19-image-classification-using-DenseNet/blob/main/image/callbak.png'>
-  Didapat Hasil selama 50 epoch adalah sebagai berikut
   
   <img src='https://github.com/RidwendDev/SARS-COV-19-image-classification-using-DenseNet/blob/main/image/epoch.png'>
- Setelah proses training selesai, saya memvisualiasikan plot learning curvenya seperti pada gambar berikut
   <img src='https://github.com/RidwendDev/SARS-COV-19-image-classification-using-DenseNet/blob/main/image/1.png'>
   <img src='https://github.com/RidwendDev/SARS-COV-19-image-classification-using-DenseNet/blob/main/image/2.png'>
   * Hasil loss dan acc untuk masing-masing train dan val
   <img src='https://github.com/RidwendDev/SARS-COV-19-image-classification-using-DenseNet/blob/main/image/trainval.png'>
- Proses terakhir yang dilakukan adalah membuat classification report serta memvisualisasikan confusion matrix
   <img src='https://github.com/RidwendDev/SARS-COV-19-image-classification-using-DenseNet/blob/main/image/creport.png'>
   <img src='https://github.com/RidwendDev/SARS-COV-19-image-classification-using-DenseNet/blob/main/image/confmat.png'>
   <br>Kesimpulan dari hasil evaluasi:
      Dari data terlihat bahwasannya model yang dibangun ini mampu memprediksi data dengan menghasilkan akurasi sebesar 100%, hasil ini tentunya juga dipengaruhi oleh data yang sedikit juga serta proses iterasi dari epoch yang cukup banyak. Jadi meskipun didapat hasil akurasi yang tinggi, model ini tentunya masih memiliki banyak kekurangan alasannya tentu dari data yang digunakan masih sangat sedikit untuk ditrain dalam Neural Network tingkat industri, hal lain tentunya model ini tidak dapat dijadikan sebagai tolok ukur utama untuk mengidentifikasi penyakit covid-19 dengan citra dada hasil rontgen. Harapannya saya dapat mengembangkan model yang lebih baik dengan skala data yang lebih besar lagi di masa yang akan datang. Baik itu saja yang dapat saya sampaikan pada kesempatan kali ini, semoga dapat bermanfaat. Terima kasih 🙏🏻 
   
   
   
   
   
