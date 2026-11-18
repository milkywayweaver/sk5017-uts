# UTS SK2017 Sains dan Pengamanan Data

# 1. Klasifikasi
Pada bagian ini, algoritma klasifikasi _Support Vector Machine_ dan _Random Forest_ digunakan untuk memprediksi jenis kelamin (`sex`) dari abalone. Sebelum data diolah oleh SVM dan RF, data harus dibersihkan dan dipraproses terlebih dahulu.

**Pembersihan Data** <br>
- Pengecekan nilai nan tidak menunjukkan keberadaan nilai nan.
- Pengecekan duplikat tidak menunjukkan keberadaan entri duplikat.
- Pengecekan tipe dataa memberikan tipe data yang tepat.

**Praproses Data** <br>
Sebelum data dipraproses, pencilan pada data dihapus. Penghapusan pencilan ini dilakukan dengan metode _clustering_ DBSCAN yang dapat memberikan nilai _cluster_ -1 untuk titik-titik data yang dianggap sebagai _noise_/pencilan. Setelah data pencilan dihilangkan, data dibagi menjadi data latih dan data uji. Praproses data dilakukan dengan `Pipeline` yang mempermudah proses latih. `Pipeline` tersebut diiisi dengan beberapa proses.
- **Untuk kolom numerik**
  - Transformasi log berdasarkan [1], dengan implementasi dari https://github.com/milkywayweaver/autologtransform. Hal ini dilakukan karena distribusi fitur pada data berbentuk _skewed_.
  - Normalisasi dengan `StandardScaler` untuk membuat rentang tiap fitur berada pada orde yang serupa.
  - Reduksi dimensi dengan `PCA`. Selain untuk reduksi dimensi, PCA juga berfungsi untuk membuat separasi antarkelas lebih terlihat pada beberapa PC saja.
- **Untuk kolom kategorikal**
  - _Encoding_ data dengan `OrdinalEncoder`. `OrdinalEncoder` digunakan dibanding `LabelEncoder` agar dapat diintegrasikan pada `Pipeline`.

**Proses Latih** <br>
Baik model SVM dan RF dilatih dengan menggunakan Optuna sebagai alat bantu untuk _hyperparameter tuning_, dengan parameter yang dicari berasal dari bagian praproses dan bagian model.

| Bagian | _Hyperparameter_ | Rentang Nilai |
| ---   | ---              | ---           |
| Praproses | `ASLT beta` | [-50, 50] |
| Praproses | `PCA n_components` | [2, Jumlah fitru numerik data] |
| SVC | `C` | [0.01, 10] |
| SVC | `kernel` | [`linear`, `poly`, `rbf`] |
| RF | `n_estimators` | [10, 500] |
| RF | `criterion` | [`entropy`, `gini`, `log_loss` |

**Hasil** <br>

| SVM | Male |	Female | Infant |	accuracy | macro avg |	weighted avg |
| --- | --- | --- | --- | --- | --- | --- |
|precision|	0.494792|	0.503226|	0.744108|	0.584928|	0.580708|	0.577061|
|recall|	0.618893|	0.297710|	0.827715|	0.584928|	0.581439|	0.584928|
|f1-score|	0.549928|	0.374101|	0.783688|	0.584928|	0.569239|	0.569482|
|support|	307.000000|	262.000000|	267.000000|	0.584928|	836.000000|	836.000000|


| RF | Male |	Female | Infant |	accuracy | macro avg |	weighted avg |
| --- | --- | --- | --- | --- | --- | --- |
|precision|	0.461988|	0.457547|	0.762411|	0.562201|	0.560649|	0.556545|
|recall|	0.514658|	0.370229|	0.805243|	0.562201|	0.563377|	0.562201|
|f1-score|	0.486903|	0.409283|	0.783242|	0.562201|	0.559809|	0.557221|
|support|	307.000000|	262.000000|	267.000000|	0.562201|	836.000000|	836.000000|

Hasil mungkin tidak sama persis dengan yang ada pada file notebook karena perbedaan _random seed_ dari setiap run.
<img width="1071" height="490" alt="image" src="https://github.com/user-attachments/assets/20df2efe-83f0-412a-ba47-6b163c735506" />
Gambar 1.1. _Confusion Matrix_ kedua model. Angka dalam tanda kurung menandakan akurasi.

Dari tabel dan gambar di atas, dapat dilihat bahwa kedua model memiliki kesulitan untuk memprediksi jenis kelamin abalone, baik untuk _male_ atau _female_. Kelas _male_ tampak memiliki _recall_ yang lebih baik dibanding _female_, dengan _precision_ yang serupa. Dari _confusion matrix_ di atas, kelas _female_ terlihat banyak misklasifikasi dengan kelas _male_. Di sisi lain, kelas _infant_ tampak memiliki performa yang paling baik. Hal ini cukup masuk akal jika dsitribusi setiap kelas dilihat. Gambar berikut menunjukkan distribusi tersebut, dimana kelas _male_ dan _female_ memiliki distribusi yang saling tumpang-tindih, sedangkan _infant_ memiliki separasi yang cukup jelas. Hal ini menandakan, fitur-fitur yang ada pada dataset ini masih belum cukup untuk memisahkan kelas _male_ dan _female_ dengan efektif, dan butuh lebih banyak _domain knowledge_ untuk memperbaikinya.
<img width="1189" height="611" alt="image" src="https://github.com/user-attachments/assets/1fd3284e-7657-4b9d-9643-f5b8faa3aeff" />

Pada dataset ini, kedua model tampak memiliki performa yang cukup setara. Keunggulan RF yaitu tidak begitu terdampak oleh data pencilan dan skala fitur tidak terlalu terlihat karena data telah dibersihkan dan dinormalisasi. 

# **Regresi**
Pada bagian ini, dua model regresi `LinearRegression` dan `Lasso` akan digunakan untuk memprediksi jumlah cincin (`rings`) pada abalone. Tahapan pembersihan dan praproses data sama dengan bagian klasifikasi. Untuk pelatihan model, Optuna kembali digunakan dengan ruang cari sebagai berikut.

| Bagian | _Hyperparameter_ | Rentang Nilai |
| ---   | ---              | ---           |
| Praproses | `ASLT beta` | [-50, 50] |
| Praproses | `PCA n_components` | [2, Jumlah fitru numerik data] |
| Lasso | `alpha` | [0, 10] |
| Lasso | `selection` | [`cyclic`, `random`] |

**Hasil**
| Model | RMSE | MSE | MAPE | R2 |
| --- | --- | --- | --- | --- |
|Linear Regression|	2.328315|	5.421053|	0.152454|	0.499220|
|LASSO Regression|	2.517166|	6.336124|	0.159768|	0.414688|

Dari hasil tersebut, dapat dilihat bahwa regresi linear biasa tampak lebih baik dibanding model Lasso dari seluruh metrik. Meski begitu, karena jumlah cincin merupakan bilangan bulat, kedua model memiliki galat yang cukup serupa, sekitar 2 untuk regresi linear dan 3 untuk lasso (dari pembulatan RMSE). Gambar 2.1 menunjukkan perbedaan hasil dari kedua model jika dibandingkan dengan jumlah cincin sebenarnya. Dari gambar kiri, lasso tampak memiliki distribusi yang lebih ramping dibanding regresi linear biasa, tetapi dengan rata-rata yang cukup bergeser dari rata-rata sebenarnya. Distribusi jumlah cincin sebenarnya juga memiliki ekor yang cukup panjang di sebelah kanan. Gambar di sebelah kanan menunjukkan distribusi kumulatif perbedaan antara hasil prediksi model dengan nilai sebenarnya. Dari gambar tersebut, dapat dilihat bahwa sekitar 70% hasil prediksi memiliki 2 atau kurang, dan 90% hasil prediksi memiliki galat 4 atau kurang.

<img width="1189" height="509" alt="image" src="https://github.com/user-attachments/assets/d96a6111-6f2a-4fa0-8a1f-bf10749d68f3" />
Gambar 2.1. Distribusi prediksi cincin abalone.

Jika _correlation map_ data dilihat (gambar 2.2), dapat diperhatikan bahwa fitur `sex` memiliki korelasi yang mendekati 0 dengan `rings`. Jika fitur `sex` dibuang, galat model yang diberikan naik untuk regresi linear biasa dan turun untuk Lasso. Hal ini mengindikasikan regresi linear yang dapat memanfaatkan fitur `sex` untuk prediksinya. Model lasso tampak kurang cocok pada dataset ini yang memiliki jumlah fitur yang relatif sedikit. Kemungkinan, sedikitnya jumlah fitur ini membuat regresi lasso tidak bisa memanfaatkan keunggulannya dalam memberikan bobot pada setiap koefisien.

<img width="590" height="490" alt="image" src="https://github.com/user-attachments/assets/0d7a7355-2e0b-4acb-a83b-017fb7b3780e" />

Gambar 2.2. _Correlation map_ data abalone.

| Model |	RMSE | MSE | MAPE | R2 |
| --- | --- | --- | --- | --- |
|Linear Regression|	2.357671|	5.558612|	0.152759|	0.486512|
|LASSO Regression|	2.370825|	5.620813|	0.146680|	0.480766|

# **_Clustering_**
Pada bagian ini, tida metode _clustering_ digunakan untuk mengelompokkan data abalone:  _k-means clustering_, _spectral clustering_, dan _hierarchical clustering_. Terdapat dua sub-bagian pada bagian ini: _clustering_ pada jenis kelamin (`sex`) dan pada jumlah cincin (`ring`). _Hierarchical clustering_ dilakukan secara _bottom-up_ dengan modul `AgglomerativeClustering`. Sama seperti bagian-bagian sebelumnya, proses pembersihan dan praproses data dilakukan dengan metode serupa. Jumlah cincin yang berupa bilangan bulat dikelompokkan menjadi 6 kelompok dalam rentang `[0, 5, 10, 15, 20, 25, 30]`. Kemudian, Optuna kembali digunakan untuk melakukan _hyperparameter tuning_ dengan ruang cari berikut.

| Bagian | _Hyperparameter_ | Rentang Nilai |
| ---   | ---              | ---           |
| Praproses | `ASLT beta` | [-50, 50] |
| Praproses | `PCA n_components` | [2, Jumlah fitru numerik data] |
| K-Means | `n_clusters` | [2, 11] |
| _Spectral clustering_ | `affinity` | [`rbf`, `nearest_neighbors`] |
| _Spectral clustering_ | `n_clusters` | [2, 11] |
| _Hierarchical clustering_ | `metric` | [`euclidean`, `l1`, `l2`, `cosine`] |
| _Hierarchical clustering_ | `linkage` | [`ward`, `complete`, `average`, `single`] |
| _Hierarchical clustering_ | `n_clusters` | [2, 11] |

Berikut adalah hasil untuk pengecekan jenis kelamin dengan _clustering_.
|Model|	ARI|	Homogenity|	Completeness|	V Measure|	Silhouette|
| --- | --- | --- | --- | --- | --- |
|KMeans (`n_clusters=2`)|	0.191320|	0.143200|	0.232551|	0.177252|	0.248919|
|SpectralClustering (`n_clusters=3`)|	0.166365|	0.153057|	0.199754|	0.173315|	0.121965|
|AgglomerativeClustering (`n_clusters=2`)|	0.211069|	0.164154|	0.272834|	0.204980|	0.328393|

Dari tabel tersebut, dapat dilihat bahwa hasil _clustering_ dari metode _hierarchical clustering_ memiliki performa yang terbaik, dan _spectral clustering_ memiliki performa yang terburuk. Kemungkinan, hal ini disebabkan karena jumlah _cluster_ terbaik yang diperoleh, dimana K-Means dan _hierarchical clustering_ sama-sama memiliki 2 _cluster_ sedangkan _spectral clustering_ memiliki 3 _cluster_. Jika dilihat scatterplot_ dari hasil _hierarchical clustering_, dapat dilihat bahwa kedua _decision boundary_ kedua _cluster_ tampak memisahkan _infant_ dan bukan _infant_, tetapi tidak bisa membedakan _male_ dan _female_. Hal ini cukup wajar karena titik-titik data _male_ dan _female_ yang saling tumpah tindih.

<img width="2044" height="1965" alt="image" src="https://github.com/user-attachments/assets/1ccda5ad-c8b7-4185-9ff4-3269f58474b9" />
Gambar 3.1. _Pairplot_ prediksi _hierarchical clustering_ untuk pengelompokan jenis kelamin.

# **Referensi**
[1] Feng, Q., Hannig, J., & Marron, J. S. (2016). A note on automatic data transformation. Stat, 5(1), 82-87.
