
# **Laporan Proyek Machine Learning - Fikri Zulfialdi**

## **Project Overview**


Bermain game tidak hanya memberikan hiburan, tetapi juga menawarkan berbagai manfaat psikologis dan kognitif. Penelitian menunjukkan bahwa game dapat meningkatkan kemampuan pemecahan masalah, kreativitas, serta koordinasi tangan-mata (Granic, Lobel, & Engels, 2014). Selain itu, beberapa game membantu pemain mengelola stres, seperti yang ditemukan dalam tinjauan sistematis tentang manfaat game terhadap kesehatan mental (Primack et al., 2012). Game juga mendukung pembentukan komunitas online yang memperkuat koneksi sosial (Kowert & Quandt, 2016) dan memberikan peluang pembelajaran melalui mekanisme permainan yang melatih keterampilan berpikir kritis (Gee, 2003). Bahkan, beberapa jenis game dapat meningkatkan aktivitas fisik dan kesehatan, seperti yang dijelaskan dalam studi tentang exergames (Staiano & Calvert, 2011). Dengan manfaat ini, menemukan game yang sesuai dengan minat dan kebutuhan pemain menjadi semakin penting untuk memaksimalkan pengalaman bermain.

Proyek pengembangan sistem rekomendasi game untuk platform Steam bertujuan memberikan pengalaman pengguna yang lebih personal dengan menyarankan game yang relevan berdasarkan preferensi dan aktivitas mereka. Sistem ini sangat dibutuhkan karena jumlah game yang tersedia sangat banyak, yang dapat membuat pengguna kewalahan dalam menemukan game yang sesuai dengan minat mereka (Valve Corporation, 2023). Selain membantu pengguna, sistem ini juga berdampak signifikan pada platform, seperti mengurangi tingkat ketidakaktifan pengguna (churn rate) dan meningkatkan pendapatan melalui penjualan yang lebih terarah (McKinsey & Company, 2021). Rekomendasi yang tepat mampu menciptakan pengalaman bermain yang lebih memuaskan, mempermudah pengguna menemukan konten baru, dan meningkatkan keterlibatan pengguna secara keseluruhan, sebagaimana dicatat dalam penelitian tentang perilaku pengguna di platform digital (Resnick & Varian, 1997).

Efektivitas sistem rekomendasi telah terbukti melalui berbagai penelitian. Menurut McKinsey & Company (2021), sekitar 35% penjualan Amazon berasal dari sistem rekomendasi, menunjukkan potensi besar model serupa untuk diterapkan pada Steam. Yann LeCun, seorang ahli kecerdasan buatan, menjelaskan bahwa teknologi seperti deep learning mampu membuat rekomendasi lebih personal dan kontekstual, sehingga meningkatkan relevansi rekomendasi bagi pengguna (LeCun, 2018). Dalam bukunya The Long Tail, Chris Anderson (2006) menyoroti bagaimana personalisasi membantu menjangkau pasar "ekor panjang," yang memungkinkan game indie atau kurang populer menemukan audiens yang sesuai. Penelitian Resnick dan Varian (1997) juga menunjukkan bahwa sistem rekomendasi berperan penting dalam meningkatkan keterlibatan dan loyalitas pengguna di platform digital. Valve, pengembang Steam, melaporkan bahwa fitur seperti Steam Discovery Queue mampu meningkatkan konversi penjualan game indie hingga 20-30%, memperkuat dampak positif sistem rekomendasi pada kesuksesan platform game digital (Valve Corporation, 2023).

Oleh karena itu sangat penting hadirnya sistem rekomendasi yang mumpuni yang bisa merekomendasikan dengan baik sesuai preferensi *user* yang diberi rekomendasi. Menjawab kepentingan tersebut, akan dibuat model sistem rekomendasi dengan menggunakan Content-Based Filtering dan Collaborative Filtering untuk menemukan sistem rekomendasi game terbaik menggunakan data dari platform penjualan game Steam.

## **Business Understanding**



### **Problem Statement**
1. Bagaimana cara meningkatkan pengalaman pengguna dalam memilih game di
platform Steam?
2. Game apa yang paling populer berdasarkan total durasi bermain, jumlah review, jumlah rekomendasi?
3. Apakah platform atau sistem operasi yang didukung (Windows, Mac, Linux) mempengaruhi preferensi pengguna?
4. Bagaimana distribusi harga game memengaruhi tingkat ulasan positif dari pengguna?
5. Apakah ada hubungan antara rating game dan jumlah waktu yang dihabiskan pengguna untuk memainkannya?


### **Goals**  
1. Mengembangkan model rekomendasi game yang mempermudah pengguna memilih game berdasarkan karakteristik game dan karakteristik user yang serupa.  
2. Menganalisis game dengan populasi yang diwakilkan oleh agregat total durasi bermain, jumlah review, dan jumlah rekomendasi.  
3. Mengeksplorasi pengaruh sistem operasi terhadap preferensi game pengguna.  
4. Menganalisis hubungan antara distribusi harga game dan tingkat ulasan positif.  
5. Mengidentifikasi hubungan antara rating game dan waktu yang dihabiskan pengguna untuk memainkannya.  



### **Solution**  
1. Menggunakan dua pendekatan rekomendasi utama:  
   - Content-Based Filtering: Menganalisis genre, tag, harga, dan fitur game lainnya untuk merekomendasikan game yang serupa dengan preferensi pengguna menggunakan *cosine similarity*.  
   - Collaborative Filtering: Menggunakan data ulasan dan interaksi pengguna untuk merekomendasikan game yang disukai oleh pengguna dengan preferensi serupa menggunakan algoritma deep learning *RecommenderNet*.  

2. Menggunakan Exploratory Data Analysis (EDA) untuk memahami tren dan pola pengguna:  
   - Menganalisis agregat data seperti total durasi bermain, jumlah ulasan, dan jumlah rekomendasi untuk mengidentifikasi game yang paling populer.  
   - Memahami hubungan antara distribusi harga game dengan tingkat ulasan positif.  
   - Mengeksplorasi preferensi genre berdasarkan jumlah ulasan, durasi bermain, dan feedback positif.  
   - Menganalisis dampak platform (Windows, Mac, Linux) terhadap preferensi pengguna untuk meningkatkan rekomendasi lintas platform.  
   - Menyelidiki hubungan antara rating game dengan waktu bermain untuk memberikan wawasan terhadap tingkat keterlibatan pengguna.  



## **Data Understanding**



## **Data Loading**

Dataset ini berasal dari "https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam" repositori Kaggle milik Anton Kozyriev, kemungkinan besar menggunakan data dari Steam API atau metode scraping untuk tujuan analisis rekomendasi game. Dengan 71K dilihat dan jumlah unduhan (10.9K entri) serta usability 10.00/10.00, dataset ini cukup populer dan dapat diandalkan.
Terdapat 4 file csv namun disini hanya digunakan 3 file yaitu `games.csv`, `games_metadata.json`, dan `recommendations.csv` ke dalam 2 DataFrame berbeda. Hal ini dilakukakan karena terdapat perbedaan struktur data.

Dataframe `games_metadata`
 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>app_id</th>
      <th>description</th>
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13500</td>
      <td>Enter the dark underworld of Prince of Persia ...</td>
      <td>[Action, Adventure, Parkour, Third Person, Gre...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22364</td>
      <td></td>
      <td>[Action]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>113020</td>
      <td>Monaco: What's Yours Is Mine is a single playe...</td>
      <td>[Co-op, Stealth, Indie, Heist, Local Co-Op, St...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>226560</td>
      <td>Escape Dead Island is a Survival-Mystery adven...</td>
      <td>[Zombies, Adventure, Survival, Action, Third P...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>249050</td>
      <td>Dungeon of the Endless is a Rogue-Like Dungeon...</td>
      <td>[Roguelike, Strategy, Tower Defense, Pixel Gra...</td>
    </tr>
  </tbody>
</table>

Dataframe `games`    

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>app_id</th>
      <th>title</th>
      <th>date_release</th>
      <th>win</th>
      <th>mac</th>
      <th>linux</th>
      <th>rating</th>
      <th>positive_ratio</th>
      <th>user_reviews</th>
      <th>price_final</th>
      <th>price_original</th>
      <th>discount</th>
      <th>steam_deck</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13500</td>
      <td>Prince of Persia: Warrior Within™</td>
      <td>2008-11-21</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>Very Positive</td>
      <td>84</td>
      <td>2199</td>
      <td>9.99</td>
      <td>9.99</td>
      <td>0.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22364</td>
      <td>BRINK: Agents of Change</td>
      <td>2011-08-03</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>Positive</td>
      <td>85</td>
      <td>21</td>
      <td>2.99</td>
      <td>2.99</td>
      <td>0.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>113020</td>
      <td>Monaco: What's Yours Is Mine</td>
      <td>2013-04-24</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>Very Positive</td>
      <td>92</td>
      <td>3722</td>
      <td>14.99</td>
      <td>14.99</td>
      <td>0.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>226560</td>
      <td>Escape Dead Island</td>
      <td>2014-11-18</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>Mixed</td>
      <td>61</td>
      <td>873</td>
      <td>14.99</td>
      <td>14.99</td>
      <td>0.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>249050</td>
      <td>Dungeon of the ENDLESS™</td>
      <td>2014-10-27</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>Very Positive</td>
      <td>88</td>
      <td>8784</td>
      <td>11.99</td>
      <td>11.99</td>
      <td>0.0</td>
      <td>True</td>
    </tr>
  </tbody>
</table>

File `games.csv` dan `games_metadata.json` digabung kedalam 1 dataframe yaitu `games_data`.

Dataframe `games_data`
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>app_id</th>
      <th>description</th>
      <th>tags</th>
      <th>title</th>
      <th>date_release</th>
      <th>win</th>
      <th>mac</th>
      <th>linux</th>
      <th>rating</th>
      <th>positive_ratio</th>
      <th>user_reviews</th>
      <th>price_final</th>
      <th>price_original</th>
      <th>discount</th>
      <th>steam_deck</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13500</td>
      <td>Enter the dark underworld of Prince of Persia ...</td>
      <td>[Action, Adventure, Parkour, Third Person, Gre...</td>
      <td>Prince of Persia: Warrior Within™</td>
      <td>2008-11-21</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>Very Positive</td>
      <td>84</td>
      <td>2199</td>
      <td>9.99</td>
      <td>9.99</td>
      <td>0.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22364</td>
      <td></td>
      <td>[Action]</td>
      <td>BRINK: Agents of Change</td>
      <td>2011-08-03</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>Positive</td>
      <td>85</td>
      <td>21</td>
      <td>2.99</td>
      <td>2.99</td>
      <td>0.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>113020</td>
      <td>Monaco: What's Yours Is Mine is a single playe...</td>
      <td>[Co-op, Stealth, Indie, Heist, Local Co-Op, St...</td>
      <td>Monaco: What's Yours Is Mine</td>
      <td>2013-04-24</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>Very Positive</td>
      <td>92</td>
      <td>3722</td>
      <td>14.99</td>
      <td>14.99</td>
      <td>0.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>226560</td>
      <td>Escape Dead Island is a Survival-Mystery adven...</td>
      <td>[Zombies, Adventure, Survival, Action, Third P...</td>
      <td>Escape Dead Island</td>
      <td>2014-11-18</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>Mixed</td>
      <td>61</td>
      <td>873</td>
      <td>14.99</td>
      <td>14.99</td>
      <td>0.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>249050</td>
      <td>Dungeon of the Endless is a Rogue-Like Dungeon...</td>
      <td>[Roguelike, Strategy, Tower Defense, Pixel Gra...</td>
      <td>Dungeon of the ENDLESS™</td>
      <td>2014-10-27</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>Very Positive</td>
      <td>88</td>
      <td>8784</td>
      <td>11.99</td>
      <td>11.99</td>
      <td>0.0</td>
      <td>True</td>
    </tr>
  </tbody>
</table>

Data `recommendations.csv` dibuat dataframe terpisah yaitu `recommendations` karena memiliki dimensi data yang berbeda.

Dataframe `recommendations`
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>app_id</th>
      <th>helpful</th>
      <th>funny</th>
      <th>date</th>
      <th>is_recommended</th>
      <th>hours</th>
      <th>user_id</th>
      <th>review_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>975370</td>
      <td>0</td>
      <td>0</td>
      <td>2022-12-12</td>
      <td>True</td>
      <td>36.3</td>
      <td>51580</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>304390</td>
      <td>4</td>
      <td>0</td>
      <td>2017-02-17</td>
      <td>False</td>
      <td>11.5</td>
      <td>2586</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1085660</td>
      <td>2</td>
      <td>0</td>
      <td>2019-11-17</td>
      <td>True</td>
      <td>336.5</td>
      <td>253880</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>703080</td>
      <td>0</td>
      <td>0</td>
      <td>2022-09-23</td>
      <td>True</td>
      <td>27.4</td>
      <td>259432</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>526870</td>
      <td>0</td>
      <td>0</td>
      <td>2021-01-10</td>
      <td>True</td>
      <td>7.9</td>
      <td>23869</td>
      <td>4</td>
    </tr>
  </tbody>
</table>

	Total rows of games_data: 50872
	Total columns of games_data: 15
	Total rows of recommendations: 41154794
	Total columns of recommendations: 8



`games_data`  memiliki 50872 entri data dengan 15 kolom informasi sedangkan  `recommendation`  memiliki 41154794 entri data dengan 8 kolom informasi.

## **Variable Description**

penjelasan variabel pada dataframe `games_data`


| **No** | **Variabel**      | **Tipe Data** | **Penjelasan**                                                                 |
|--------|-------------------|---------------|---------------------------------------------------------------------------------|
| 1      | `app_id`          | `int64`       | ID unik untuk setiap aplikasi atau game di Steam.                              |
| 2      | `description`     | `object`      | Deskripsi singkat tentang game, biasanya mencakup fitur utama atau cerita.     |
| 3      | `tags`            | `object`      | Kumpulan tag atau kategori yang menggambarkan genre dan fitur game.            |
| 4      | `title`           | `object`      | Nama atau judul game di Steam.                                                 |
| 5      | `date_release`    | `object`      | Tanggal rilis game dalam format string.                                        |
| 6      | `win`             | `bool`        | Menunjukkan apakah game tersedia untuk platform Windows (`True`/`False`).      |
| 7      | `mac`             | `bool`        | Menunjukkan apakah game tersedia untuk platform MacOS (`True`/`False`).        |
| 8      | `linux`           | `bool`        | Menunjukkan apakah game tersedia untuk platform Linux (`True`/`False`).        |
| 9      | `rating`          | `object`      | Kategori rating game berdasarkan ulasan, seperti "Mostly Positive".            |
| 10     | `positive_ratio`  | `int64`       | Rasio ulasan positif dalam bentuk persentase (%).                              |
| 11     | `user_reviews`    | `int64`       | Jumlah total ulasan pengguna untuk game tersebut.                              |
| 12     | `price_final`     | `float64`     | Harga akhir game setelah diskon (dalam satuan mata uang tertentu).             |
| 13     | `price_original`  | `float64`     | Harga asli game sebelum diskon (dalam satuan mata uang tertentu).              |
| 14     | `discount`        | `float64`     | Persentase diskon yang diberikan pada game (dalam %).                          |
| 15     | `steam_deck`      | `bool`        | Menunjukkan apakah game kompatibel dengan Steam Deck (`True`/`False`).         |


penjelasan variabel pada dataframe `recommendations`

| **No** | **Variabel**      | **Tipe Data** | **Penjelasan**                                                               |
|--------|-------------------|---------------|-------------------------------------------------------------------------------|
| 1      | `app_id`          | `int64`       | ID unik untuk setiap aplikasi atau game di Steam yang terkait ulasan ini.    |
| 2      | `helpful`         | `int64`       | Jumlah reaksi "helpful" yang diberikan pengguna lain untuk ulasan ini.        |
| 3      | `funny`           | `int64`       | Jumlah reaksi "funny" yang diberikan pengguna lain untuk ulasan ini.          |
| 4      | `date`            | `object`      | Tanggal ulasan diberikan, biasanya dalam format string.                      |
| 5      | `is_recommended`  | `bool`        | Menunjukkan apakah ulasan merekomendasikan game (`True`/`False`).            |
| 6      | `hours`           | `float64`     | Jumlah jam yang dihabiskan pengguna bermain game sebelum menulis ulasan.     |
| 7      | `user_id`         | `int64`       | ID unik pengguna yang menulis ulasan.                                        |
| 8      | `review_id`       | `int64`       | ID unik untuk setiap ulasan dalam dataset.                                   |


## **Statistic Data**

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 50872 entries, 0 to 50871
    Data columns (total 15 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   app_id          50872 non-null  int64  
     1   description     50872 non-null  object 
     2   tags            50872 non-null  object 
     3   title           50872 non-null  object 
     4   date_release    50872 non-null  object 
     5   win             50872 non-null  bool   
     6   mac             50872 non-null  bool   
     7   linux           50872 non-null  bool   
     8   rating          50872 non-null  object 
     9   positive_ratio  50872 non-null  int64  
     10  user_reviews    50872 non-null  int64  
     11  price_final     50872 non-null  float64
     12  price_original  50872 non-null  float64
     13  discount        50872 non-null  float64
     14  steam_deck      50872 non-null  bool   
    dtypes: bool(4), float64(3), int64(3), object(5)
    memory usage: 4.5+ MB

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>app_id</th>
      <th>positive_ratio</th>
      <th>user_reviews</th>
      <th>price_final</th>
      <th>price_original</th>
      <th>discount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5.087200e+04</td>
      <td>50872.000000</td>
      <td>5.087200e+04</td>
      <td>50872.000000</td>
      <td>50872.000000</td>
      <td>50872.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.055224e+06</td>
      <td>77.052033</td>
      <td>1.824425e+03</td>
      <td>8.620325</td>
      <td>8.726788</td>
      <td>5.592212</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.103249e+05</td>
      <td>18.253592</td>
      <td>4.007352e+04</td>
      <td>11.514164</td>
      <td>11.507021</td>
      <td>18.606679</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000e+01</td>
      <td>0.000000</td>
      <td>1.000000e+01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.287375e+05</td>
      <td>67.000000</td>
      <td>1.900000e+01</td>
      <td>0.990000</td>
      <td>0.990000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>9.860850e+05</td>
      <td>81.000000</td>
      <td>4.900000e+01</td>
      <td>4.990000</td>
      <td>4.990000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.524895e+06</td>
      <td>91.000000</td>
      <td>2.060000e+02</td>
      <td>10.990000</td>
      <td>11.990000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.599300e+06</td>
      <td>100.000000</td>
      <td>7.494460e+06</td>
      <td>299.990000</td>
      <td>299.990000</td>
      <td>90.000000</td>
    </tr>
  </tbody>
</table>

   

Dataframe games_data memiliki 50872 entri rekod dan 15 kolom dengan 5 kolom numerik.

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 41154794 entries, 0 to 41154793
    Data columns (total 8 columns):
     #   Column          Dtype  
    ---  ------          -----  
     0   app_id          int64  
     1   helpful         int64  
     2   funny           int64  
     3   date            object 
     4   is_recommended  bool   
     5   hours           float64
     6   user_id         int64  
     7   review_id       int64  
    dtypes: bool(1), float64(1), int64(5), object(1)
    memory usage: 2.2+ GB

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>app_id</th>
      <th>helpful</th>
      <th>funny</th>
      <th>hours</th>
      <th>user_id</th>
      <th>review_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4.115479e+07</td>
      <td>4.115479e+07</td>
      <td>4.115479e+07</td>
      <td>4.115479e+07</td>
      <td>4.115479e+07</td>
      <td>4.115479e+07</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>6.032724e+05</td>
      <td>3.202567e+00</td>
      <td>1.058071e+00</td>
      <td>1.006022e+02</td>
      <td>7.450576e+06</td>
      <td>2.057740e+07</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4.729233e+05</td>
      <td>4.693649e+01</td>
      <td>2.867060e+01</td>
      <td>1.761675e+02</td>
      <td>4.010685e+06</td>
      <td>1.188037e+07</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000e+01</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.539400e+05</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>7.800000e+00</td>
      <td>4.287256e+06</td>
      <td>1.028870e+07</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.351500e+05</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>2.730000e+01</td>
      <td>7.546446e+06</td>
      <td>2.057740e+07</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>9.331100e+05</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>9.920000e+01</td>
      <td>1.096877e+07</td>
      <td>3.086609e+07</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.253290e+06</td>
      <td>3.621200e+04</td>
      <td>2.810900e+04</td>
      <td>1.000000e+03</td>
      <td>1.430606e+07</td>
      <td>4.115479e+07</td>
    </tr>
  </tbody>
</table>


Dataframe recommendations memiliki 41154794  entri data dan 8 kolom dengan 3 kolom numerik.

## **Exploratory Data Analysis**

### **Unvariate Data Analysis**

#### **Price**

    
![png](gambar_files/gambar_51_0.png)
    


	Mean Price: 8.620324933165593
	Median Price: 4.99
	Mode Price: 0.0


Distribusi harga dari sebuah game ternyata sangat terjal ke kanan. Artinya sebagian besar data berada di bawah quartil bawah.

#### **Rating**


    
![png](gambar_files/gambar_54_0.png)
    

![png](gambar_files/gambar_187_0.png)

Sebagian besar `rating` berada ada kategori positive dengan berbagai derajatnya kecuali `overwhelmingly positive`.

#### **Positive Ratio**





    
![png](gambar_files/gambar_57_0.png)
    


	Top-rated games (positive_ratio > 90%):
	                                           title  positive_ratio
	2                   Monaco: What's Yours Is Mine              92
	8              Hyperdimension Neptunia Re;Birth1              94
	16                                          FORM              91
	21      Sniper Elite 3 - Camouflage Weapons Pack              95
	27                   Take Command - 2nd Manassas              93
	...                                          ...             ...
	50856                                   STANDBOX              91
	50858                              Fortune's Run              94
	50860                             Kill The Crows              96
	50867  I Expect You To Die 3: Cog in the Machine              96
	50870                        Forgive Me Father 2              95
	
	[13405 rows x 2 columns]


Distribusi `Rasio Ulasan Positif` adalah normal dengan miring ke kiri. Sebagian besar data berada di atas rata-rata. Dengan Nilai di kisaran 90 memiliki frekuensi terbanyak.

#### **User Reviews**





    
![png](gambar_files/gambar_60_0.png)
    


	Top-rated games (positive_ratio > 90%):
	                                  title  user_reviews
	14398  Counter-Strike: Global Offensive       7494460
	47770               PUBG: BATTLEGROUNDS       2217226
	13176                            Dota 2       2045628
	12717                Grand Theft Auto V       1484122
	14535   Tom Clancy's Rainbow Six® Siege        993312
	...                                 ...           ...
	47793                          Factorio        134384
	11720              World of Tanks Blitz        131334
	480                          Far Cry® 5        129943
	15926                 World of Warships        129335
	48601                Sons Of The Forest        128626
	
	[100 rows x 2 columns]

Jumlah review seluruh game diurutkan dari yang terbesar dan ternyata distribusi jumlah review setiap game sangat timpang dengan hanya beberapa game saja yang memiliki jumlah review yang sangat besar.

#### **Date Release**





    
![png](gambar_files/gambar_63_0.png)
    


	date_release
	1997       2
	1998       1
	1999       3
	2000       2
	2001       2
	2002       1
	2003       2
	2004       4
	2005       3
	2006      56
	2007      82
	2008     146
	2009     322
	2010     284
	2011     376
	2012     565
	2013     822
	2014    1921
	2015    2963
	2016    4209
	2017    4989
	2018    5461
	2019    5057
	2020    6135
	2021    6774
	2022    7265
	2023    3425
	Name: count, dtype: int64


Setiap tahun jumlah game yang dirilis cenderung selalu meningkat kecuali di tahun 2018 dan 2023.

#### **Top Games**


    
![png](gambar_files/gambar_66_1.png)
    


Game yang memiliki Total Waktu Dimainkan tertinggi adalah **Team Fortress 2**.



    
![png](gambar_files/gambar_68_1.png)
    


Game yang paling banyak direkomendasikan adalah **Team Fortress 2**.


    
![png](gambar_files/gambar_70_1.png)
    


Game yang paling banyak di-review adalah **Counter-Strike: Global Offensive**.

### **Multivariate Data Analysis**

#### **Average Price vs Positive Feedback Ratio**





    
![png](gambar_files/gambar_74_0.png)
    


 Rasio *feedback* positif yang rendah (10-50), terlihat fluktuasi harga yang cukup besar dengan kecenderungan sedikit meningkat. Namun, seiring dengan meningkatnya rasio, fluktuasi mulai berkurang, dan grafik menjadi lebih stabil. Yaitu pada rentang rasio *feedback* positif 50-100, harga rata-rata cenderung stabil dengan variasi yang minim. Ini menunjukkan bahwa rasio *feedback* positif tidak memiliki kaitan signifikan dengan harga, hanya ada sedikit korelasi positif pada kisaran yang rendah.

#### **Median Positive Feedback Ratio by Price Intervals**

    
![png](gambar_files/gambar_77_1.png)
    


Sejauh ini tidak signifikan pengaruh harga game terhadap Rasio Ulasan Positif. Hanya ada sedikit penurunan rasio ulasan positif pada rentang harga yang rendah dan sedikit peningkatan pada rentang harga menengah ke tinggi.

#### **Average Positive Ratio by Tag**

    
![png](gambar_files/gambar_80_0.png)
    


	Top 10 Tags by Average Positive Ratio:
	Indie           42.558795
	Singleplayer    35.419602
	Action          32.794347
	Adventure       30.567424
	Casual          27.528994
	2D              18.713850
	Simulation      18.241567
	Strategy        16.236043
	RPG             15.084840
	Atmospheric     13.605638
	dtype: float64
	
	Other tags aggregated average positive ratio: 1.51


Tags yang memiliki Rasio Ulasan Positif paling tinggi adalah Indie. Game indie banyak memiliki ulasan positif karena banyak dimaklumi oleh pemain juga harganya yang biasanya sangat murah.

#### **Average Price by Tag**





    
![png](gambar_files/gambar_83_0.png)
    


	Top 10 and Bottom 10 Genres:
	Musou               47.115417
	Baseball            28.531081
	Rugby               27.240625
	BMX                 25.712222
	Audio Production    24.915827
	Medical Sim         24.242135
	Photo Editing       22.773750
	Cycling             22.528500
	Web Publishing      21.339808
	Video Production    20.850899
	Other                9.947339
	Abstract             4.193056
	Short                4.132869
	Mod                  4.104306
	Minimalist           3.948958
	Spelling             3.698846
	Idler                3.383874
	Clicker              3.328889
	Documentary          2.906122
	Free to Play         1.898432
	Tile-Matching        1.592000
	dtype: float64


Tag game yang memiliki Harga rata-rata tertinggi adalah Musou. Game Musou banyak diminati penggemar fanatik yang rela keluar uang lebih banyak.

#### **Distribution of Total Hours Played per Rating Category**





    
![png](gambar_files/gambar_86_0.png)
    


Game yang memiliki ulasan rating yang positif dimainkan lebih banyak daripada yang negatif. Menunjukkan bahwa rating berpengaruh terhadap minat bermain.

#### **Percentage of Games per OS**





    
![png](gambar_files/gambar_89_0.png)
    


Jumlah Game yang rilis di platform windows mendominasi dengan 69.4%.

#### **Game Popularity per OS**





    
![png](gambar_files/gambar_92_0.png)
    


Terlihat ternyata game yang rilis di platform windows jauh lebih populer dari os lainnya. Namun ini bisa disebabkan juga karena lebih banyak game yang rilis di platform windows.





    
![png](gambar_files/gambar_94_0.png)
    


Jika memperhitungkan jumlah game yang dirilis di setiap platform, malah linux yang memiliki popularitas leih tinggi.

## **Data Preparation**

### **Data Cleaning**

#### **Missing Value & Duplicate**

Menilik dari nilai data yang ada, untuk mengetahui data yang kosong perlu dilakukan pre-processing terlebih dahulu karena missing value tidak berupa `None`, `Null` atau `NaN` tapi berupa empty string `''` , empty list `[]`, zero-value atau non-numerical value pada kolom numerik.

Jumlah _missing value_ empty string `''`  dan empty list `[]` yang ada di dataframe `games_data` adalah sebagai berikut

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Nilai yang Kosong</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>app_id</th>
      <td>0</td>
    </tr>
    <tr>
      <th>description</th>
      <td>10373</td>
    </tr>
    <tr>
      <th>tags</th>
      <td>1244</td>
    </tr>
    <tr>
      <th>title</th>
      <td>0</td>
    </tr>
    <tr>
      <th>date_release</th>
      <td>0</td>
    </tr>
    <tr>
      <th>win</th>
      <td>0</td>
    </tr>
    <tr>
      <th>mac</th>
      <td>0</td>
    </tr>
    <tr>
      <th>linux</th>
      <td>0</td>
    </tr>
    <tr>
      <th>rating</th>
      <td>0</td>
    </tr>
    <tr>
      <th>positive_ratio</th>
      <td>0</td>
    </tr>
    <tr>
      <th>user_reviews</th>
      <td>0</td>
    </tr>
    <tr>
      <th>price_final</th>
      <td>0</td>
    </tr>
    <tr>
      <th>price_original</th>
      <td>0</td>
    </tr>
    <tr>
      <th>discount</th>
      <td>0</td>
    </tr>
    <tr>
      <th>steam_deck</th>
      <td>0</td>
    </tr>
  </tbody>
</table>

Dilakukan drop pada baris yang memiliki _missing value_ sehingga menghasilkan jumlah baris sebagai berikut.
    
    Total of rows: 40484
    Total of column: 15

selanjutnya diperiksa nilai numerik yang menghasilkan nilai 0 pada dataframe `games_data`.

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Nilai yang bernilai 0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>app_id</th>
      <td>0</td>
    </tr>
    <tr>
      <th>description</th>
      <td>0</td>
    </tr>
    <tr>
      <th>tags</th>
      <td>0</td>
    </tr>
    <tr>
      <th>title</th>
      <td>0</td>
    </tr>
    <tr>
      <th>date_release</th>
      <td>0</td>
    </tr>
    <tr>
      <th>win</th>
      <td>0</td>
    </tr>
    <tr>
      <th>mac</th>
      <td>0</td>
    </tr>
    <tr>
      <th>linux</th>
      <td>0</td>
    </tr>
    <tr>
      <th>rating</th>
      <td>0</td>
    </tr>
    <tr>
      <th>positive_ratio</th>
      <td>0</td>
    </tr>
    <tr>
      <th>user_reviews</th>
      <td>0</td>
    </tr>
    <tr>
      <th>price_final</th>
      <td>7799</td>
    </tr>
    <tr>
      <th>price_original</th>
      <td>7846</td>
    </tr>
    <tr>
      <th>discount</th>
      <td>36261</td>
    </tr>
    <tr>
      <th>steam_deck</th>
      <td>0</td>
    </tr>
  </tbody>
</table>

Terdapat nilai 0 di dalam kolom yang memiliki kepentingan yaitu kolom `price_final`. Maka baris yang memiliki nilai 0 tersebut dihilangkan, sedangkan nilai 0 di baris lainnya dibiarkan karena tidak relevan dan malah akan menghilangkan data penting ketika dihilangkan. 
    
    Total of rows: 32685
    Total of column: 15
    
Dilakukan deteksi adanya data non-numerik di kolom-kolom numerik. Data yang terdeteksi akan diubah menjadi NaN untuk dihilangkan menggunakan dropna().

    Total of rows: 32685
    Total of column: 15

Diperiksa juga missing value dan data invalid di kolom numerik pada dataframe recommendations

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Nilai yang Kosong</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>app_id</th>
      <td>0</td>
    </tr>
    <tr>
      <th>helpful</th>
      <td>0</td>
    </tr>
    <tr>
      <th>funny</th>
      <td>0</td>
    </tr>
    <tr>
      <th>date</th>
      <td>0</td>
    </tr>
    <tr>
      <th>is_recommended</th>
      <td>0</td>
    </tr>
    <tr>
      <th>hours</th>
      <td>0</td>
    </tr>
    <tr>
      <th>user_id</th>
      <td>0</td>
    </tr>
    <tr>
      <th>review_id</th>
      <td>0</td>
    </tr>
  </tbody>
</table>

tidak ada _missing value_ yang harus dihilangkan

Penghilangan nilai 0 tidak dilakukan pada dataframe `recommendations` karena kolom penting `hours` berisi banyak data nilai 0 yang berarti user tidak memiliki cukup waktu bermain game yang user tersebut ulas.

Selanjutnya diperiksa data duplikat pada dataframe `games_data`    

    Number of duplicates (excluding 'tags' column): 0


Diperiksa juga data duplikat pada dataframe `recommendations`

    Number of duplicates : 0

Tidak terdeteksi data duplikat pada kedua dataframe sehingga tidak perlu dihilangkan.

	Total rows of games_data: 32685
	Total columns of games_data: 15
	Total rows of recommendations: 41154794
	Total columns of recommendations: 8

Jumlah data di games_data setelah cleaning adalah 32685 dan di recommendations adalah 41154794

#### **Data Reduction**

Dilakukan filter game yang dianggap relevan di dataframe `recommendations` dengan menggunakan data dari dataframe `games_data` yang telah dihilangkan *missing value*-nya. Hasilnya sebagai berikut

    Total of rows: 16337800
    Total of column: 8
    <class 'pandas.core.frame.DataFrame'>
    Index: 16337800 entries, 66 to 41154792
    Data columns (total 8 columns):
     #   Column          Dtype  
    ---  ------          -----  
     0   app_id          int64  
     1   helpful         int64  
     2   funny           int64  
     3   date            object 
     4   is_recommended  bool   
     5   hours           float64
     6   user_id         int64  
     7   review_id       int64  
    dtypes: bool(1), float64(1), int64(5), object(1)
    memory usage: 1012.8+ MB


Karena ukuran data yang masih terlalu besar yaitu 16337800 data maka dilakukan sampling dengan kriteria seperti berikut: data tidak lebih lama dari tahun 2020, setiap interval dari total waktu dimainkan dari setiap game akan diambil 200 game relevan, setiap user relevan memiliki minimal 5 review game.

Setelah dilakukan filter melalui data game setelah tahun 2020.

    Number of rows after date filtering: 9024102

Disampling 200 game pada setiap kelas data dari kolom 'hours' yang diagreasi dari setiap game lalu diurutkan dan dibuat 10 kelas interval.

    Number of sampled games: 2000
    Number of rows in the filtered dataset: 675275

Dilakukan filter user dengan minimal 5 review.

    Number of user with min reviews: 4235
    Number of rows in the filtered dataset: 34586


Jumlah entri data `games_data` setelah dibersihkan adalah 32685 dan jumlah baris setelah reduksi dari data `recommendations` adalah 34586.





### **1. Content Based Filtering**

Dataframe dari `games_data` dianalisis kolomnya untuk ditemukan kolom yang sesuai dalam penghitungan `cosine similarity`.

<table class="dataframe" border="1">
<thead>
<tr>
<th>app_id</th>
<th>description</th>
<th>tags</th>
<th>title</th>
<th>date_release</th>
<th>win</th>
<th>mac</th>
<th>linux</th>
<th>rating</th>
<th>positive_ratio</th>
<th>user_reviews</th>
<th>price_final</th>
<th>price_original</th>
<th>discount</th>
<th>steam_deck</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>13500</td>
<td>Enter the dark underworld of Prince of Persia ...</td>
<td>[Action, Adventure, Parkour, Third Person, Gre...</td>
<td>Prince of Persia: Warrior Within&trade;</td>
<td>2008-11-21</td>
<td>True</td>
<td>False</td>
<td>False</td>
<td>Very Positive</td>
<td>84</td>
<td>2199</td>
<td>9.99</td>
<td>9.99</td>
<td>0.0</td>
<td>True</td>
</tr>
<tr>
<th>1</th>
<td>113020</td>
<td>Monaco: What's Yours Is Mine is a single playe...</td>
<td>[Co-op, Stealth, Indie, Heist, Local Co-Op, St...</td>
<td>Monaco: What's Yours Is Mine</td>
<td>2013-04-24</td>
<td>True</td>
<td>True</td>
<td>True</td>
<td>Very Positive</td>
<td>92</td>
<td>3722</td>
<td>14.99</td>
<td>14.99</td>
<td>0.0</td>
<td>True</td>
</tr>
<tr>
<th>2</th>
<td>226560</td>
<td>Escape Dead Island is a Survival-Mystery adven...</td>
<td>[Zombies, Adventure, Survival, Action, Third P...</td>
<td>Escape Dead Island</td>
<td>2014-11-18</td>
<td>True</td>
<td>False</td>
<td>False</td>
<td>Mixed</td>
<td>61</td>
<td>873</td>
<td>14.99</td>
<td>14.99</td>
<td>0.0</td>
<td>True</td>
</tr>
<tr>
<th>3</th>
<td>249050</td>
<td>Dungeon of the Endless is a Rogue-Like Dungeon...</td>
<td>[Roguelike, Strategy, Tower Defense, Pixel Gra...</td>
<td>Dungeon of the ENDLESS&trade;</td>
<td>2014-10-27</td>
<td>True</td>
<td>True</td>
<td>False</td>
<td>Very Positive</td>
<td>88</td>
<td>8784</td>
<td>11.99</td>
<td>11.99</td>
<td>0.0</td>
<td>True</td>
</tr>
<tr>
<th>4</th>
<td>250180</td>
<td>&ldquo;METAL SLUG 3&rdquo;, the masterpiece in SNK&rsquo;s emble...</td>
<td>[Arcade, Classic, Action, Co-op, Side Scroller...</td>
<td>METAL SLUG 3</td>
<td>2015-09-14</td>
<td>True</td>
<td>False</td>
<td>False</td>
<td>Very Positive</td>
<td>90</td>
<td>5579</td>
<td>7.99</td>
<td>7.99</td>
<td>0.0</td>
<td>True</td>
</tr>
</tbody>
</table>
Ditentukan bahwa kolom informasi yang digunakan yaitu 'description', 'tags', 'title', 'rating', 'positive_ratio', 'user_reviews', 'price_final'. Kolom `descriptions` dan `tags` akan digunakan *feature*nya dari konversi TF-IDF sedangkan kolom `rating`, `positive_ratio`, `user_reviews`, `price_final` akan digunakan sebagai *feature* nilai numeriknya. Kolom lain yang tidak termasuk akan di-drop dari dataframe.

Kolom tag yang berisi list tag apa saja yang ada di sebuah game, dipecah menjadi satu kumpulan string dipisahkan oleh spasi yang bisa diterima sebagai corpus oleh TfidfVectorizer. Namun sebelum dipecah beberapa tag yang berisi lebih dari satu kata seperti `Local Co-Op`, `Third Person`, `Tower Defense` disatukan menjadi satu kata dengan mengganti spasi dengan underscore '_' agar menjadi seperti ini `Local_Co-Op`, `Third_Person`, `Tower_Defense`.

Rating di- agar dapat dimengerti oleh cosine similarity dengan map seperti berikut:


```python
# Convert the rating column (text to numeric)
rating_mapping = {
    'Overwhelmingly Positive': 4,
    'Very Positive': 3,
    'Positive': 2,
    'Mostly Positive': 1,
    'Mixed': 0,
    'Mostly Negative': -1,
    'Negative': -2,
    'Very Negative': -3,
    'Overwhelmingly Negative': -4
}
```





#### **Hyper-Parameter Tuning**

**GridSearch for TF-IDF**

Digunakan GridSearch untuk mengoptimalkan proses vektorisasi TF-IDF dari deskripsi game yang memiliki deretan string yang panjang. Yaitu dengan optimasi parameter sebagai berikut: max_features yaitu berapa variasi kata yang akan digunakan model, ngram yaitu jenis urutan kata yang digunakan apakah unigrams (1 kata) atau bigrams (2 kata), max_df adalah jumlah maksimal persentase sebuah kata muncul yaitu jika sebuah kata muncul terlalu banyak maka maknanya hilang, min_df yaitu sebaliknya jika sebuah kata hanya muncul dalam sedikit dokumen maka tidak relevan maka nilai ini adalah jumlah dokumen minimal sebuah kata muncul. Parameter pengujian digunakan KNN karena algoritma tersebut adalah clustering yang sesuai dengan tujuan vektorisasi TF-IDF.




    Fitting 3 folds for each of 864 candidates, totalling 2592 fit


    Best parameters found:  {'kmeans__init': 'random', 'kmeans__max_iter': 300, 'kmeans__n_clusters': 4, 'tfidf__max_df': 1.0, 'tfidf__max_features': 100, 'tfidf__min_df': 5, 'tfidf__ngram_range': (1, 2)}

Parameter paling optimal adalah 'max_df' = 1.0, 'max_features'= 100, 'min_df'= 5, 'ngram_range'= (1, 2) atau bigrams.

#### **Vectorizer**

Digunakan TF-IDF dengan nilai dari 0 hingga 1 untuk data berupa teks yaitu deskripsi dan tags. Khusus untuk data deskripsi digunakan parameter hasil parameter tuning sedangkan untuk data tags tidak menggunakan parameter tuning karena data tags masing-masing berdiri sendiri tanpa konteks yang berkaitan. Data numerikal di-vektorisasi menggunakan min-max scaler yang menghasilkan nilai dari 0 hingga 1.


#### **Feature Engineering**

Tiga vektor yang telah dibuat digabungkan seluruh kolomnya menghasilkan data vektor gabungan tf-idf dan numerikal yaitu Combined Features. Hal ini dilakukan untuk melihat keterkaitan yang lebih kompleks dari data yang ada. Metode yang digunakan adalah concatenation array 
```
Vector Combined Features = Vector Tf-idf Descriptions  + Vactor Tf-idf Tags + Vector Numerical Features
```

#### **Data Vector**

**Vector TF-IDF Descriptions**

<table class="dataframe" border="1">
<thead>
<tr>
<th>&nbsp;</th>
<th>2d</th>
<th>3d</th>
<th>action</th>
<th>adventure</th>
<th>adventure game</th>
<th>arcade</th>
<th>based</th>
<th>battle</th>
<th>beautiful</th>
<th>best</th>
<th>...</th>
<th>turn</th>
<th>turn based</th>
<th>unique</th>
<th>use</th>
<th>using</th>
<th>vr</th>
<th>war</th>
<th>way</th>
<th>weapons</th>
<th>world</th>
</tr>
<tr>
<th>title</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
</tr>
</thead>
<tbody>
<tr>
<th>Prince of Persia: Warrior Within&trade;</th>
<td>0.000000</td>
<td>0.000000</td>
<td>0.000000</td>
<td>0.00000</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>...</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>0.000000</td>
<td>0.000000</td>
</tr>
<tr>
<th>Monaco: What's Yours Is Mine</th>
<td>0.000000</td>
<td>0.000000</td>
<td>0.000000</td>
<td>0.00000</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>...</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>0.000000</td>
<td>0.000000</td>
</tr>
<tr>
<th>Escape Dead Island</th>
<td>0.000000</td>
<td>0.000000</td>
<td>0.000000</td>
<td>0.39422</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>...</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>0.000000</td>
<td>0.000000</td>
</tr>
<tr>
<th>Dungeon of the ENDLESS&trade;</th>
<td>0.000000</td>
<td>0.000000</td>
<td>0.000000</td>
<td>0.00000</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>...</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.334867</td>
<td>0.000000</td>
<td>0.000000</td>
</tr>
<tr>
<th>METAL SLUG 3</th>
<td>0.533932</td>
<td>0.000000</td>
<td>0.412735</td>
<td>0.00000</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>...</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>0.000000</td>
<td>0.000000</td>
</tr>
<tr>
<th>...</th>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
</tr>
<tr>
<th>Welcome to Kowloon</th>
<td>0.000000</td>
<td>0.000000</td>
<td>0.000000</td>
<td>0.00000</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>...</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>0.000000</td>
<td>0.000000</td>
</tr>
<tr>
<th>Taboo Trial</th>
<td>0.000000</td>
<td>0.427918</td>
<td>0.320664</td>
<td>0.00000</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>...</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.390671</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>0.000000</td>
<td>0.257031</td>
</tr>
<tr>
<th>Hometopia</th>
<td>0.000000</td>
<td>0.000000</td>
<td>0.000000</td>
<td>0.00000</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.579992</td>
<td>...</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>0.000000</td>
<td>0.337227</td>
</tr>
<tr>
<th>Fading Afternoon</th>
<td>0.000000</td>
<td>0.000000</td>
<td>0.000000</td>
<td>0.00000</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>...</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>0.000000</td>
<td>0.000000</td>
</tr>
<tr>
<th>Forgive Me Father 2</th>
<td>0.000000</td>
<td>0.000000</td>
<td>0.379612</td>
<td>0.00000</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>...</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.403275</td>
<td>0.497654</td>
<td>0.000000</td>
</tr>
</tbody>
</table>
<p>32685 rows &times; 100 columns</p>
 
**Vector TF-IDF Tags**

<table class="dataframe" border="1">
<thead>
<tr>
<th>&nbsp;</th>
<th>1980s</th>
<th>1990</th>
<th>2d</th>
<th>2d_fighter</th>
<th>2d_platformer</th>
<th>360_video</th>
<th>3d</th>
<th>3d_fighter</th>
<th>3d_platformer</th>
<th>3d_vision</th>
<th>...</th>
<th>well</th>
<th>werewolves</th>
<th>western</th>
<th>wholesome</th>
<th>word_game</th>
<th>world_war_i</th>
<th>world_war_ii</th>
<th>wrestling</th>
<th>written</th>
<th>zombies</th>
</tr>
<tr>
<th>title</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
</tr>
</thead>
<tbody>
<tr>
<th>Prince of Persia: Warrior Within&trade;</th>
<td>0.000000</td>
<td>0.0</td>
<td>0.000000</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>0.0</td>
<td>0.000000</td>
<td>0.0</td>
<td>...</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
</tr>
<tr>
<th>Monaco: What's Yours Is Mine</th>
<td>0.000000</td>
<td>0.0</td>
<td>0.113716</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>0.0</td>
<td>0.000000</td>
<td>0.0</td>
<td>...</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
</tr>
<tr>
<th>Escape Dead Island</th>
<td>0.000000</td>
<td>0.0</td>
<td>0.000000</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>0.0</td>
<td>0.000000</td>
<td>0.0</td>
<td>...</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.272423</td>
</tr>
<tr>
<th>Dungeon of the ENDLESS&trade;</th>
<td>0.000000</td>
<td>0.0</td>
<td>0.137786</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>0.0</td>
<td>0.000000</td>
<td>0.0</td>
<td>...</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
</tr>
<tr>
<th>METAL SLUG 3</th>
<td>0.000000</td>
<td>0.0</td>
<td>0.114235</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>0.0</td>
<td>0.000000</td>
<td>0.0</td>
<td>...</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
</tr>
<tr>
<th>...</th>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
</tr>
<tr>
<th>Welcome to Kowloon</th>
<td>0.000000</td>
<td>0.0</td>
<td>0.000000</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.206379</td>
<td>0.0</td>
<td>0.000000</td>
<td>0.0</td>
<td>...</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
</tr>
<tr>
<th>Taboo Trial</th>
<td>0.000000</td>
<td>0.0</td>
<td>0.000000</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.186225</td>
<td>0.0</td>
<td>0.278652</td>
<td>0.0</td>
<td>...</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
</tr>
<tr>
<th>Hometopia</th>
<td>0.000000</td>
<td>0.0</td>
<td>0.000000</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.160494</td>
<td>0.0</td>
<td>0.000000</td>
<td>0.0</td>
<td>...</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
</tr>
<tr>
<th>Fading Afternoon</th>
<td>0.283493</td>
<td>0.0</td>
<td>0.130473</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>0.0</td>
<td>0.000000</td>
<td>0.0</td>
<td>...</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
</tr>
<tr>
<th>Forgive Me Father 2</th>
<td>0.000000</td>
<td>0.0</td>
<td>0.000000</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.185446</td>
<td>0.0</td>
<td>0.000000</td>
<td>0.0</td>
<td>...</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
</tr>
</tbody>
</table>
<p>32685 rows &times; 464 columns</p>


**Vector Numerical Features**

<table class="dataframe" border="1">
<thead>
<tr>
<th>&nbsp;</th>
<th>rating</th>
<th>positive_ratio</th>
<th>user_reviews</th>
<th>price_final</th>
</tr>
<tr>
<th>title</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
</tr>
</thead>
<tbody>
<tr>
<th>Prince of Persia: Warrior Within&trade;</th>
<td>3.0</td>
<td>84.0</td>
<td>2199.0</td>
<td>9.99</td>
</tr>
<tr>
<th>Monaco: What's Yours Is Mine</th>
<td>3.0</td>
<td>92.0</td>
<td>3722.0</td>
<td>14.99</td>
</tr>
<tr>
<th>Escape Dead Island</th>
<td>0.0</td>
<td>61.0</td>
<td>873.0</td>
<td>14.99</td>
</tr>
<tr>
<th>Dungeon of the ENDLESS&trade;</th>
<td>3.0</td>
<td>88.0</td>
<td>8784.0</td>
<td>11.99</td>
</tr>
<tr>
<th>METAL SLUG 3</th>
<td>3.0</td>
<td>90.0</td>
<td>5579.0</td>
<td>7.99</td>
</tr>
<tr>
<th>...</th>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
</tr>
<tr>
<th>Welcome to Kowloon</th>
<td>1.0</td>
<td>78.0</td>
<td>499.0</td>
<td>7.00</td>
</tr>
<tr>
<th>Taboo Trial</th>
<td>3.0</td>
<td>94.0</td>
<td>494.0</td>
<td>12.00</td>
</tr>
<tr>
<th>Hometopia</th>
<td>0.0</td>
<td>61.0</td>
<td>248.0</td>
<td>17.00</td>
</tr>
<tr>
<th>Fading Afternoon</th>
<td>1.0</td>
<td>79.0</td>
<td>358.0</td>
<td>20.00</td>
</tr>
<tr>
<th>Forgive Me Father 2</th>
<td>3.0</td>
<td>95.0</td>
<td>82.0</td>
<td>17.00</td>
</tr>
</tbody>
</table>
<p>32685 rows &times; 4 columns</p>

**Vector Combined Features**


<table class="dataframe" border="1">
<thead>
<tr>
<th>&nbsp;</th>
<th>2d</th>
<th>3d</th>
<th>action</th>
<th>adventure</th>
<th>adventure game</th>
<th>arcade</th>
<th>based</th>
<th>battle</th>
<th>beautiful</th>
<th>best</th>
<th>...</th>
<th>word_game</th>
<th>world_war_i</th>
<th>world_war_ii</th>
<th>wrestling</th>
<th>written</th>
<th>zombies</th>
<th>rating</th>
<th>positive_ratio</th>
<th>user_reviews</th>
<th>price_final</th>
</tr>
<tr>
<th>title</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
</tr>
</thead>
<tbody>
<tr>
<th>Prince of Persia: Warrior Within&trade;</th>
<td>0.000000</td>
<td>0.000000</td>
<td>0.000000</td>
<td>0.00000</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>...</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>3.0</td>
<td>84.0</td>
<td>2199.0</td>
<td>9.99</td>
</tr>
<tr>
<th>Monaco: What's Yours Is Mine</th>
<td>0.000000</td>
<td>0.000000</td>
<td>0.000000</td>
<td>0.00000</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>...</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>3.0</td>
<td>92.0</td>
<td>3722.0</td>
<td>14.99</td>
</tr>
<tr>
<th>Escape Dead Island</th>
<td>0.000000</td>
<td>0.000000</td>
<td>0.000000</td>
<td>0.39422</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>...</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.272423</td>
<td>0.0</td>
<td>61.0</td>
<td>873.0</td>
<td>14.99</td>
</tr>
<tr>
<th>Dungeon of the ENDLESS&trade;</th>
<td>0.000000</td>
<td>0.000000</td>
<td>0.000000</td>
<td>0.00000</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>...</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>3.0</td>
<td>88.0</td>
<td>8784.0</td>
<td>11.99</td>
</tr>
<tr>
<th>METAL SLUG 3</th>
<td>0.533932</td>
<td>0.000000</td>
<td>0.412735</td>
<td>0.00000</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>...</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>3.0</td>
<td>90.0</td>
<td>5579.0</td>
<td>7.99</td>
</tr>
<tr>
<th>...</th>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
</tr>
<tr>
<th>Welcome to Kowloon</th>
<td>0.000000</td>
<td>0.000000</td>
<td>0.000000</td>
<td>0.00000</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>...</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>1.0</td>
<td>78.0</td>
<td>499.0</td>
<td>7.00</td>
</tr>
<tr>
<th>Taboo Trial</th>
<td>0.000000</td>
<td>0.427918</td>
<td>0.320664</td>
<td>0.00000</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>...</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>3.0</td>
<td>94.0</td>
<td>494.0</td>
<td>12.00</td>
</tr>
<tr>
<th>Hometopia</th>
<td>0.000000</td>
<td>0.000000</td>
<td>0.000000</td>
<td>0.00000</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.579992</td>
<td>...</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>0.0</td>
<td>61.0</td>
<td>248.0</td>
<td>17.00</td>
</tr>
<tr>
<th>Fading Afternoon</th>
<td>0.000000</td>
<td>0.000000</td>
<td>0.000000</td>
<td>0.00000</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>...</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>1.0</td>
<td>79.0</td>
<td>358.0</td>
<td>20.00</td>
</tr>
<tr>
<th>Forgive Me Father 2</th>
<td>0.000000</td>
<td>0.000000</td>
<td>0.379612</td>
<td>0.00000</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>...</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.000000</td>
<td>3.0</td>
<td>95.0</td>
<td>82.0</td>
<td>17.00</td>
</tr>
</tbody>
</table>
<p>32685 rows &times; 568 columns</p>


### **2. Collaborative Filtering**

Berikut ini dataframe dari `recommendations` yang dianalisis kolomnya untuk ditemukan data mana yang dapat digunakan untuk melatih model deep learning `RecommenderNet`.

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>app_id</th>
      <th>helpful</th>
      <th>funny</th>
      <th>date</th>
      <th>is_recommended</th>
      <th>hours</th>
      <th>user_id</th>
      <th>review_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>601840</td>
      <td>0</td>
      <td>0</td>
      <td>2020-06-18</td>
      <td>True</td>
      <td>51.3</td>
      <td>4591253</td>
      <td>6126607</td>
    </tr>
    <tr>
      <th>1</th>
      <td>999220</td>
      <td>2</td>
      <td>0</td>
      <td>2022-04-05</td>
      <td>True</td>
      <td>8.8</td>
      <td>9089111</td>
      <td>6128126</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1090630</td>
      <td>0</td>
      <td>0</td>
      <td>2022-06-06</td>
      <td>False</td>
      <td>10.1</td>
      <td>6138998</td>
      <td>6129816</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1812090</td>
      <td>2</td>
      <td>0</td>
      <td>2022-07-14</td>
      <td>True</td>
      <td>0.7</td>
      <td>13271495</td>
      <td>6136413</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1090630</td>
      <td>52</td>
      <td>0</td>
      <td>2021-12-14</td>
      <td>True</td>
      <td>241.4</td>
      <td>12053923</td>
      <td>6140311</td>
    </tr>
  </tbody>
</table>
  
Digunakan data `is_recommended` dan `hours` sebagai parameter untuk model deep learning. Kolom lain yang tidak dibutuhkan didrop.

#### **Encoding**

Dilakukan pembuatan map encoding terhadap user_id dan app_id menjadi nilai integer ordinal untuk menyederhanakan data. Map untuk mengkonversi ulang nilai encoding ke semula juga dibuat untuk melihat hasil rekomendasi. Lalu sebagian output dari map-nya ditampilkan sebagai berikut

	app_id ke ordinal: [(601840, 0), (999220, 1), (1090630, 2), (1812090, 3), (363440, 4), (976590, 5), (586200, 6), (1147560, 7), (874390, 8), (851890, 9), (577690, 10)] 
	user_id ke ordinal: [(4591253, 0), (9089111, 1), (6138998, 2), (13271495, 3), (12053923, 4), (13219396, 5), (2622846, 6), (3366677, 7), (14088822, 8), (11451103, 9), (13112500, 10)]
	ordinal app_id ke semula : [(0, 601840), (1, 999220), (2, 1090630), (3, 1812090), (4, 363440), (5, 976590), (6, 586200), (7, 1147560), (8, 874390), (9, 851890), (10, 577690)] 
	ordinal user_id ke semula: [(0, 4591253), (1, 9089111), (2, 6138998), (3, 13271495), (4, 12053923), (5, 13219396), (6, 2622846), (7, 3366677), (8, 14088822), (9, 11451103), (10, 13112500)]

Data user_id dan app_id di-encode sebagai dataframe baru berisi nilai-nilai representasi integer dari mapping yang telah dibuat sebelumnya. Output hasil mapping ini yang ditraining melalu model deep learning `RecommenderNet`. Sebagian data ditampilkan sebagai berikut

<table class="dataframe" border="1">
<thead>
<tr>
<th>&nbsp;</th>
<th>user_encoded</th>
<th>app_encoded</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>0</td>
<td>0</td>
</tr>
<tr>
<th>1</th>
<td>1</td>
<td>1</td>
</tr>
<tr>
<th>2</th>
<td>2</td>
<td>2</td>
</tr>
<tr>
<th>3</th>
<td>3</td>
<td>3</td>
</tr>
<tr>
<th>4</th>
<td>4</td>
<td>2</td>
</tr>
<tr>
<th>5</th>
<td>5</td>
<td>2</td>
</tr>
<tr>
<th>6</th>
<td>6</td>
<td>4</td>
</tr>
<tr>
<th>7</th>
<td>7</td>
<td>2</td>
</tr>
<tr>
<th>8</th>
<td>8</td>
<td>2</td>
</tr>
<tr>
<th>9</th>
<td>9</td>
<td>5</td>
</tr>
</tbody>
</table>
<p>&nbsp;</p>

#### **Feature Engineering**

Data "hours" disesuaikan berdasarkan data game direkomendasikan atau tidak. Jika game direkomendasikan (is_recommended bernilai True), maka nilai  "hours" dengan dikalikan 1.25, dan jika tidak direkomendasikan, maka dikalikan dengan 0.75. Setelah itu dibuat kolom baru dalam dataframe sebagai `adjusted_hours`.


```math
\mathrm{adjusted\_hours} =
\begin{cases} 
\mathrm{hours} \times 1.25 & \text{jika } \mathrm{is\_recommended} = \text{True} \\
\mathrm{hours} \times 0.75 & \text{jika } \mathrm{is\_recommended} = \text{False}
\end{cases}
```


#### **Data Normalization**

Data `hours` dan `adjusted_hours` dinormalisasi dengan MinMaxScaler() agar data berada di rentang 0 hingga 1 sehingga model jadi lebih sederhana dan metrik evaluasi lebih mudah untuk dibandingkan. Data tersebut disimpan masing-masing dalam list baru agar data aslinya dapat digunakan dalam mengurutkan rekomendasi.

	+------------------------------+------------------------------+
	| hours                        | adjusted_hours               |
	+------------------------------+------------------------------+
	| 0.05135649214135549          | 0.05135649214135549          |
	+------------------------------+------------------------------+
	| 0.0088096906597257           | 0.008809690659725698         |
	+------------------------------+------------------------------+
	| 0.010111122234457903         | 0.006066673340674742         |
	+------------------------------+------------------------------+
	| 0.000700770847932726         | 0.000700770847932726         |
	+------------------------------+------------------------------+
	| 0.24166583241565723          | 0.24166583241565723          |
	+------------------------------+------------------------------+
	| 0.022925217739513466         | 0.022925217739513466         |
	+------------------------------+------------------------------+
	| 0.013414756231855042         | 0.01341475623185504          |
	+------------------------------+------------------------------+
	| 0.008008809690659726         | 0.004805285814395835         |
	+------------------------------+------------------------------+
	| 0.0032035238762638907        | 0.0019221143257583345        |
	+------------------------------+------------------------------+
	| 0.020222244468915806         | 0.020222244468915806         |
	+------------------------------+------------------------------+

#### **Train Test Split**

Dibuat 3 set data train-test dengan komposisi 8:2 untuk 3 model berbeda. Model pertama menggunakan data `hours` yang telah dinormalisasi minmaxscaler, lalu data kedua menggunakan data `is_recommended` yang bernilai boolean, data kedua menggunakan data adjusted_hours yang menggabungkan parameter `hours` dan `is_recommended`.

## **Modeling**

### **1. Content Based Filtering**

Model Content-Based Filtering dengan Cosine Similarity menggunakan informasi yang ada pada item itu sendiri, seperti deskripsi, tag, dan fitur numerik, untuk memberikan rekomendasi berdasarkan kemiripan dengan item lain. Pada model berbasis deskripsi, kemiripan dihitung berdasarkan teks deskripsi game, yang memberikan rekomendasi berdasarkan konten game itu sendiri. Model berbasis tag menggunakan kategori atau genre game, sedangkan model berbasis fitur numerik memanfaatkan data seperti rating, harga, dan waktu dimainkan untuk menentukan kemiripan. Gabungan ketiga model ini memberikan pendekatan yang lebih komprehensif dan akurat, karena menggabungkan berbagai jenis informasi. Kelebihan utama dari model content-based adalah kemampuannya memberikan rekomendasi untuk item baru tanpa membutuhkan data interaksi pengguna, namun kelemahannya adalah tidak mempertimbangkan preferensi pengguna secara langsung. Dibandingkan dengan Collaborative Filtering, yang lebih bergantung pada interaksi pengguna dan lebih personal, model content-based tidak mengalami masalah pada item baru namun bisa kesulitan memahami preferensi yang lebih kompleks dan subjektif.

#### **Cosine Similarity**

Content-based filtering menggunakan cosine similarity sebagai algoritma untuk membangun sistem rekomendasi berbasis konten. Cosine similarity mengukur kesamaan antara dua vektor dan menentukan sejauh mana kedua vektor tersebut mengarah ke arah yang sama. Ini dihitung dengan melihat sudut cosinus antara dua vektor, di mana semakin kecil sudutnya, semakin besar nilai cosine similarity. Rumusnya adalah sebagai berikut:

```math
\begin{aligned}
\text{Cos}(\theta) &= \frac{\sum_{i=1}^n a_i b_i}{\sqrt{\sum_{i=1}^n a_i^2} \sqrt{\sum_{i=1}^n b_i^2}} \\
a_i &\text{ adalah elemen ke-} i \text{ dari vektor } a \\
b_i &\text{ adalah elemen ke-} i \text{ dari vektor } b \\
n   &\text{ adalah jumlah elemen dalam vektor } a \text{ dan } b
\end{aligned}
```


 Cosine similarity memiliki beberapa kelebihan, seperti output yang ternormalisasi dalam rentang -1 hingga 1, sehingga memudahkan interpretasi. Selain itu, metode ini sederhana dan efisien untuk menangani data sparse berdimensi tinggi, seperti yang dihasilkan oleh TF-IDF. Namun, terdapat juga kelemahan, seperti asumsi bahwa semua faktor atau parameter dianggap sama penting, sensitivitas terhadap perubahan kecil pada 'sudut vektor', serta kurang cocok untuk data yang mengandung nilai negatif. Setelah sistem rekomendasi ini dibangun menggunakan deskripsi, tag, dan fitur numerik dari game, serta diujicobakan untuk menampilkan 10 rekomendasi teratas berdasarkan interaksi pengguna dengan game, hasil yang diperoleh akan memberikan gambaran tentang efektivitas model ini dalam memberikan rekomendasi.




	    Cosine Similarity (Description):
	 [[1.         0.         0.         ... 0.         0.         0.        ]
	 [0.         1.         0.         ... 0.0742182  0.         0.        ]
	 [0.         0.         1.         ... 0.24239257 0.         0.        ]
	 ...
	 [0.         0.0742182  0.24239257 ... 1.         0.         0.        ]
	 [0.         0.         0.         ... 0.         1.         0.        ]
	 [0.         0.         0.         ... 0.         0.         1.        ]]
	
	Cosine Similarity (Tags):
	 [[1.         0.09845734 0.26812377 ... 0.07997012 0.1104218  0.10809628]
	 [0.09845734 1.         0.27347163 ... 0.00967708 0.14930489 0.02757299]
	 [0.26812377 0.27347163 1.         ... 0.0837894  0.12352083 0.21100701]
	 ...
	 [0.07997012 0.00967708 0.0837894  ... 1.         0.15551319 0.12310468]
	 [0.1104218  0.14930489 0.12352083 ... 0.15551319 1.         0.10857035]
	 [0.10809628 0.02757299 0.21100701 ... 0.12310468 0.10857035 1.        ]]
	
	Cosine Similarity (Numerical Features):
	 [[1.         0.99990903 0.99942132 ... 0.97759408 0.98279593 0.67610525]
	 [0.99990903 1.         0.99889954 ... 0.97476498 0.98028576 0.66620428]
	 [0.99942132 0.99889954 1.         ... 0.98416122 0.98846778 0.70021227]
	 ...
	 [0.97759408 0.97476498 0.98416122 ... 1.         0.999638   0.81483341]
	 [0.98279593 0.98028576 0.98846778 ... 0.999638   1.         0.79990442]
	 [0.67610525 0.66620428 0.70021227 ... 0.81483341 0.79990442 1.        ]]
	
	Cosine Similarity (Combined):
	 [[1.         0.99990876 0.99941995 ... 0.9775791  0.98278858 0.67606337]
	 [0.99990876 1.         0.99889824 ... 0.97475012 0.98027853 0.66616278]
	 [0.99941995 0.99889824 1.         ... 0.98414637 0.98845954 0.70016962]
	 ...
	 [0.9775791  0.97475012 0.98414637 ... 1.         0.99961698 0.814774  ]
	 [0.98278858 0.98027853 0.98845954 ... 0.99961698 1.         0.79985098]
	 [0.67606337 0.66616278 0.70016962 ... 0.814774   0.79985098 1.        ]]


#### **Result**

Rekomendasi ditemukan dengan cara mengambil baris dari judul game yang diprediksi. Hasilnya adalah list dari similarity score game tersebut terhadap game lain. Lalu list tersebut diurutkan dari yang terbesar dengan tidak mengikutkan kolom game yang diprediksi dalam list tersebut. Rekomendasi diambil 10 teratas dari list tersebut.

Hasil prediksi dari model dilihat dengan cara pemilihan judul game secara acak untuk diprediksi rekomendasinya.

    Randomly selected Game Title: Striving for Light: Survival
    App ID for the selected game: 2286450

**Model 1:** Cosine Similarity (Description)

    Randomly selected Game Title: Striving for Light: Survival
    App ID for the selected game: 2286450
    
    Recommendations based on Description:

<div class="stream output-id-1">
<table class="dataframe" border="1">
<thead>
<tr>
<th>&nbsp;</th>
<th>app_id</th>
<th>title</th>
<th>description</th>
<th>similarity_score</th>
</tr>
</thead>
<tbody>
<tr>
<th>11842</th>
<td>1082680</td>
<td>The Walking Dead Onslaught</td>
<td>There&rsquo;s no rest when survival is on the line. ...</td>
<td>0.703221</td>
</tr>
<tr>
<th>31784</th>
<td>2153780</td>
<td>Survival Nation</td>
<td>Survival Nation is an open-world online RPG su...</td>
<td>0.694917</td>
</tr>
<tr>
<th>12648</th>
<td>361670</td>
<td>STAR WARS&trade; - X-Wing Alliance&trade;</td>
<td>A neutral family fights for its business - and...</td>
<td>0.669261</td>
</tr>
<tr>
<th>12764</th>
<td>1900</td>
<td>Earth 2160</td>
<td>After the destruction of the EARTH in 2150, th...</td>
<td>0.669261</td>
</tr>
<tr>
<th>18118</th>
<td>322980</td>
<td>Gods vs Humans</td>
<td>Humans are building a tower to reach the Kingd...</td>
<td>0.669261</td>
</tr>
<tr>
<th>26481</th>
<td>453730</td>
<td>Borstal</td>
<td>Survival roguelike novellas with meaningful ch...</td>
<td>0.669261</td>
</tr>
<tr>
<th>30470</th>
<td>2020460</td>
<td>Bring It On!</td>
<td>Bring It On! is a single-player auto-attacking...</td>
<td>0.664698</td>
</tr>
<tr>
<th>13023</th>
<td>653940</td>
<td>Zafehouse Diaries 2</td>
<td>Zafehouse Diaries 2 is a game of survival, exp...</td>
<td>0.657635</td>
</tr>
<tr>
<th>25702</th>
<td>1234240</td>
<td>Last Farewell</td>
<td>Last Farewell is a Survival Co-op game where y...</td>
<td>0.652663</td>
</tr>
<tr>
<th>27517</th>
<td>1665490</td>
<td>Dead Survival</td>
<td>Dead Survival is a tactical survival FPS, in a...</td>
<td>0.638971</td>
</tr>
</tbody>
</table>
</div>

**Model 2:** Cosine Similarity (Tags)

    Randomly selected Game Title: Striving for Light: Survival
    App ID for the selected game: 2286450
    
    Recommendations based on Tags:

<div class="stream output-id-1">
<div class="stream output-id-1">
<div class="output_subarea output_text">
<table class="dataframe" border="1">
<thead>
<tr>
<th>&nbsp;</th>
<th>app_id</th>
<th>title</th>
<th>description</th>
<th>similarity_score</th>
</tr>
</thead>
<tbody>
<tr>
<th>6519</th>
<td>2068280</td>
<td>Nordic Ashes: Survivors of Ragnarok</td>
<td>Nordic Ashes is a Norse inspired action-roguel...</td>
<td>0.729871</td>
</tr>
<tr>
<th>14941</th>
<td>2172190</td>
<td>Stickman's Arena</td>
<td>Stickman's Arena is a top-down arena shooter r...</td>
<td>0.727932</td>
</tr>
<tr>
<th>32614</th>
<td>2250250</td>
<td>Mighty Mage</td>
<td>Mighty Mage is a top-down arena shooter rogue-...</td>
<td>0.686939</td>
</tr>
<tr>
<th>30709</th>
<td>2331710</td>
<td>Sky Survivors</td>
<td>Sky Survivors is an arena shooter roguelite. C...</td>
<td>0.676116</td>
</tr>
<tr>
<th>20897</th>
<td>2154890</td>
<td>Survival Academy</td>
<td>A dangerous class to survive begins! A 30-minu...</td>
<td>0.658408</td>
</tr>
<tr>
<th>504</th>
<td>2077590</td>
<td>Sidestep Legends</td>
<td>Practise your sidestep skills as you dodge spe...</td>
<td>0.609145</td>
</tr>
<tr>
<th>3300</th>
<td>1646790</td>
<td>Striving for Light</td>
<td>Striving for Light is a rogue-lite ARPG where ...</td>
<td>0.596842</td>
</tr>
<tr>
<th>30451</th>
<td>1290330</td>
<td>Time Wasters</td>
<td>Time Wasters is a space shooter bullet heaven ...</td>
<td>0.589939</td>
</tr>
<tr>
<th>26879</th>
<td>1355620</td>
<td>Mage Rage</td>
<td>Join the battle, use the fury of fire, unleash...</td>
<td>0.585083</td>
</tr>
<tr>
<th>8141</th>
<td>2218140</td>
<td>Alien Slayers</td>
<td>Alien Slayers is a roguelite time survival gam...</td>
<td>0.575405</td>
</tr>
</tbody>
</table>
</div>
</div>
</div>


**Model 3:** Cosine Similarity (Numerical Features)

    Randomly selected Game Title: Striving for Light: Survival
    App ID for the selected game: 2286450
    
    Recommendations based on Numerical Features:


<div class="stream output-id-1">
<div class="output_subarea output_text">
<table class="dataframe" border="1">
<thead>
<tr>
<th>&nbsp;</th>
<th>app_id</th>
<th>title</th>
<th>description</th>
<th>similarity_score</th>
</tr>
</thead>
<tbody>
<tr>
<th>15176</th>
<td>1860370</td>
<td>Weapons Simulator</td>
<td>A Realistic Simulation with Manual Bolt Operat...</td>
<td>1.000000</td>
</tr>
<tr>
<th>29463</th>
<td>1567400</td>
<td>Yakyosho - Terror and escape at school</td>
<td>A simple Japanese horror game, with horrible c...</td>
<td>0.999999</td>
</tr>
<tr>
<th>19461</th>
<td>695570</td>
<td>PyroMind</td>
<td>Arguably the most explosive and unforgiving li...</td>
<td>0.999998</td>
</tr>
<tr>
<th>22389</th>
<td>1367230</td>
<td>Neon Cyborg Cat Club</td>
<td>A relaxing and peaceful experience set in a po...</td>
<td>0.999998</td>
</tr>
<tr>
<th>23991</th>
<td>1392130</td>
<td>Game Of Puzzles: Slavic Mythology</td>
<td>A puzzle game where you need to assemble a com...</td>
<td>0.999998</td>
</tr>
<tr>
<th>25140</th>
<td>1200780</td>
<td>Mini Island: Night</td>
<td>Mini Island: Night is a one small hold &amp; Gun, ...</td>
<td>0.999998</td>
</tr>
<tr>
<th>27007</th>
<td>593680</td>
<td>Rocking Pilot</td>
<td>Shoot, blast and slash through hordes of enemi...</td>
<td>0.999998</td>
</tr>
<tr>
<th>29945</th>
<td>1631280</td>
<td>The Jean-Paul Software Screen Explosion</td>
<td>Modern, multi-monitor, customisable screensave...</td>
<td>0.999998</td>
</tr>
<tr>
<th>25556</th>
<td>1894760</td>
<td>SnakeGame</td>
<td>Snake Game is an arena shooter about a huge sn...</td>
<td>0.999998</td>
</tr>
<tr>
<th>6509</th>
<td>1498940</td>
<td>Binky's Trash Service</td>
<td>Storm evil lairs... and take out the trash! Pl...</td>
<td>0.999995</td>
</tr>
</tbody>
</table>
</div>
</div>
  
**Model 4:** Cosine Similarity (Combined)

    Randomly selected Game Title: Striving for Light: Survival
    App ID for the selected game: 2286450
    
    Recommendations based on Combined Features:


<div class="stream output-id-1">
<div class="output_subarea output_text">
<div class="stream output-id-1">
<div class="output_subarea output_text">
<table class="dataframe" border="1">
<thead>
<tr>
<th>&nbsp;</th>
<th>app_id</th>
<th>title</th>
<th>description</th>
<th>similarity_score</th>
</tr>
</thead>
<tbody>
<tr>
<th>20944</th>
<td>1013820</td>
<td>Stars and Snowdrops</td>
<td>Spend a day in a rainy castle and befriend, or...</td>
<td>0.999815</td>
</tr>
<tr>
<th>23991</th>
<td>1392130</td>
<td>Game Of Puzzles: Slavic Mythology</td>
<td>A puzzle game where you need to assemble a com...</td>
<td>0.999813</td>
</tr>
<tr>
<th>24501</th>
<td>1556160</td>
<td>Techno Tanks</td>
<td>Techno Tanks is an intense, fast-paced arcade-...</td>
<td>0.999812</td>
</tr>
<tr>
<th>27007</th>
<td>593680</td>
<td>Rocking Pilot</td>
<td>Shoot, blast and slash through hordes of enemi...</td>
<td>0.999812</td>
</tr>
<tr>
<th>25140</th>
<td>1200780</td>
<td>Mini Island: Night</td>
<td>Mini Island: Night is a one small hold &amp; Gun, ...</td>
<td>0.999809</td>
</tr>
<tr>
<th>17247</th>
<td>42950</td>
<td>Elven Legacy: Ranger</td>
<td>First addon for Elven Legacy wargame.</td>
<td>0.999809</td>
</tr>
<tr>
<th>23950</th>
<td>600330</td>
<td>CONTRACTED</td>
<td>The infection is spreading. Cases of outbreak ...</td>
<td>0.999809</td>
</tr>
<tr>
<th>19461</th>
<td>695570</td>
<td>PyroMind</td>
<td>Arguably the most explosive and unforgiving li...</td>
<td>0.999806</td>
</tr>
<tr>
<th>30416</th>
<td>2251540</td>
<td>IBIS AM</td>
<td>All I want you to do is catch fish and search ...</td>
<td>0.999805</td>
</tr>
<tr>
<th>28782</th>
<td>464120</td>
<td>Xcinerator</td>
<td>Introducing Xcinerator, the specialized privac...</td>
<td>0.999805</td>
</tr>
</tbody>
</table>
</div>
</div>
</div>
</div>

 
#### **Best Model**

Digunakan threshold similarity score >= 0.5 sebagai nilai True Prediction. Semua model dapat memprediksi rekomendasi game dengan baik dengan nilai similarity score diatas 0.5 mencapai 100% prediksi. Namun Model terbaik dipilih Model 3: Cosine Similarity (Numerical Features) karena similarity score mencapai 1 pada rekomendasi teratas. Penjelasan mengenai metrik akan dijelaskan pada rubrik Evaluation.


### **2. Collaborative Filtering**


Collaborative Filtering dapat diterapkan menggunakan deep learning dengan memanfaatkan `embedding layer` untuk membangun model rekomendasi. `Embedding layer` adalah tipe layer dalam deep learning yang berfungsi untuk mengubah data kategorikal menjadi vektor bernilai kontinu, yang kemudian digunakan untuk merepresentasikan data secara lebih padat dan bermakna. Di Python, kita dapat menggunakan `tensorflow.keras.layers.Embedding` untuk membangun embedding layer ini.

Dalam implementasi ini, tiga model rekomendasi dibangun menggunakan berbagai fitur, yaitu `hours`, `is_recommended`, dan `adjusted hours`. Fitur `adjusted hours` diperoleh dengan menyesuaikan nilai `hours` berdasarkan apakah game direkomendasikan atau tidak, menggunakan bobot tertentu.

Data embedding antara user dan game terhadap fitur dilatih menggunakan **RecommenderNet** yang melakukan **Matrix Factorization** terhadap matrix user-game(item) $R$ yang direpresentasikan oleh `embedding layer` menjadi dua matriks kecil $P$ dan $Q$ untuk memprediksi rating yang belum diketahui. Rumus utama adalah:

$$
\hat{R} = P \times Q^T
$$

Di mana:
- $P$ adalah matriks faktor pengguna (berukuran $m \times k$),
- $Q$ adalah matriks faktor game(item) (berukuran $n \times k$),
- $\hat{R}$ adalah prediksi rating.

**RecommenderNet** melakukan operasi perhitungan ini dengan menggunakan dot product `embedding layer` vector user terhadap fitur (user_vector) sebagai $P$ dan `embedding layer` vector game terhadap fitur (app_vector) sebagai $Q$. Hasilnya adalah matriks $\hat{R}$ berisi hasil prediksi.

Untuk memperbarui $P$ dan $Q$, digunakan metode **gradient descent**, yang bertujuan meminimalkan kesalahan prediksi. Pembaruan dilakukan dengan rumus:

$$
P_i \leftarrow P_i - \eta \frac{\partial L}{\partial P_i}
$$

$$
Q_j \leftarrow Q_j - \eta \frac{\partial L}{\partial Q_j}
$$

Di mana $\eta$ adalah laju pembelajaran (learning rate), dan $\frac{\partial L}{\partial P_i}$ dan $\frac{\partial L}{\partial Q_j}$ adalah turunan dari fungsi kerugian terhadap $P_i$ dan  $Q_j$, yang mengukur perubahan yang diperlukan untuk memperbaiki kesalahan prediksi yang diimplementasikan dengan l2 regularizer pada model. Dengan iterasi ini, model akan semakin akurat dalam memprediksi nilai fitur yang belum diketahui.

Algoritma **Matrix Factorization** menggunakan metode yang disebut "collaborative filtering",  yang berasumsi bahwa jika user 1 memiliki pendapat yang sama dengan user 2 tentang suatu hal, maka user 1 lebih mungkin memiliki pandangan yang sama dengan user 2 tentang hal lain.

Contohnya, jika user 1 dan user 2 memiliki waktu bermain yang serupa terhadap game tertentu, maka user 2 lebih mungkin untuk menikmati game yang telah dimainkan oleh user 1 dalam waktu yang lama.


**Matrix Factorization** memiliki beberapa kelebihan, seperti mampu mengurangi kompleksitas model, fleksibel untuk digunakan dalam berbagai algoritma deep learning, dan efektif dalam menangkap hubungan semantik antara data. Namun, *matrix factorization* juga memiliki kelemahan, seperti membutuhkan data dalam jumlah besar untuk menghasilkan representasi yang baik, sensitivitas terhadap hyperparameter, serta rentan terhadap masalah *cold start*.

#### **RecommenderNet Model**

Model dibuat dengan diwariskan dari class  `RecommenderNet`  dari  `keras`. Model dioptimasi dengan Adam dengan learning rate 0.001 untuk  `model hours`  dan  `model adjusted_hours`  dan 0.0001 untuk  `model is_recommended`. Model  `is_recomended`  agak sulit untuk konvergen pada tingkat kesalahan yang kecil sehingga dilakukan sedikit fine-tune. Digunakan  `l2 regularizer`  sebesar 0.01 yaitu nilai default dari  `keras`  dengan loss function Binary Crossentropy. Metrik yang digunakan untuk memonitor model adalah RMSE. Tidak digunakan callback pada model ini.


#### **Training**

##### **Model 1:** Hours-Based

    Epoch 198/200
    865/865 ━━━━━━━━━━━━━━━━━━━━ 10s 6ms/step - loss: 0.0678 - root_mean_squared_error: 0.0289 - val_loss: 0.0764 - val_root_mean_squared_error: 0.0458
    Epoch 199/200
    865/865 ━━━━━━━━━━━━━━━━━━━━ 12s 8ms/step - loss: 0.0664 - root_mean_squared_error: 0.0262 - val_loss: 0.0764 - val_root_mean_squared_error: 0.0457
    Epoch 200/200
    865/865 ━━━━━━━━━━━━━━━━━━━━ 6s 7ms/step - loss: 0.0679 - root_mean_squared_error: 0.0296 - val_loss: 0.0764 - val_root_mean_squared_error: 0.0457

##### **Model 2:** User's Recommendation-Based

    Epoch 198/200
    865/865 ━━━━━━━━━━━━━━━━━━━━ 8s 6ms/step - loss: 0.3197 - root_mean_squared_error: 0.3109 - val_loss: 0.4018 - val_root_mean_squared_error: 0.3558
    Epoch 199/200
    865/865 ━━━━━━━━━━━━━━━━━━━━ 10s 6ms/step - loss: 0.3153 - root_mean_squared_error: 0.3078 - val_loss: 0.4017 - val_root_mean_squared_error: 0.3558
    Epoch 200/200
    865/865 ━━━━━━━━━━━━━━━━━━━━ 5s 6ms/step - loss: 0.3162 - root_mean_squared_error: 0.3088 - val_loss: 0.4016 - val_root_mean_squared_error: 0.3558

##### **Model 3:** Adjusted Hours-Based

    Epoch 198/200 
    865/865  ━━━━━━━━━━━━━━━━━━━━  9s 6ms/step - loss: 0.0645 - root_mean_squared_error: 0.0279 - val_loss: 0.0733 - val_root_mean_squared_error: 0.0446 
    Epoch 199/200 
    865/865  ━━━━━━━━━━━━━━━━━━━━  6s 7ms/step - loss: 0.0647 - root_mean_squared_error: 0.0272 - val_loss: 0.0733 - val_root_mean_squared_error: 0.0446 
    Epoch 200/200 
    865/865  ━━━━━━━━━━━━━━━━━━━━  11s 8ms/step - loss: 0.0648 - root_mean_squared_error: 0.0269 - val_loss: 0.0733 - val_root_mean_squared_error: 0.0446

#### **Result**

**Model 1:** Hours-Based

    Selected User ID: 7193034
    55/55 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step
    Showing recommendations for user: 7193034
    ========================================
    Games with high hours played by the user
    ╒═════════════════════════════════════════════════════╤═════════╤════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╕
    │ title                                               │   hours │ tags                                                                                                                                                                                                                                                                       │
    ╞═════════════════════════════════════════════════════╪═════════╪════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╡
    │ Paint the Town Red                                  │     5.1 │ ['Gore', 'Action', 'Blood', 'Fighting', 'Violent', 'Multiplayer', 'First-Person', 'Roguelike', 'Physics', 'Funny', 'Roguelite', 'Voxel', "Beat 'em up", 'Singleplayer', 'Indie', 'FPS', 'Mature', 'Adventure', 'Difficult', 'Arcade']                                      │
    ├─────────────────────────────────────────────────────┼─────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │ Kane & Lynch 2: Dog Days                            │     0.6 │ ['Action', 'Co-op', 'Crime', 'Third-Person Shooter', 'Shooter', 'Violent', 'Third Person', 'Atmospheric', 'Mature', 'Multiplayer', 'Singleplayer', 'Short', 'Heist', 'Nudity', 'Story Rich', 'Open World', 'Adventure', 'Local Co-Op', 'Great Soundtrack', 'Controller']   │
    ├─────────────────────────────────────────────────────┼─────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │ The Settlers® : Heritage of Kings - History Edition │     0.3 │ ['Strategy', 'Colony Sim', 'City Builder', 'RTS', 'Medieval']                                                                                                                                                                                                              │
    ├─────────────────────────────────────────────────────┼─────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │ Gothic® 3                                           │     0.2 │ ['RPG', 'Open World', 'Fantasy', 'Action', 'Singleplayer', 'Atmospheric', 'Third Person', 'Medieval', 'Gothic', 'Adventure', 'Great Soundtrack', 'Story Rich', 'Magic', 'Action RPG', 'Sandbox', 'Classic', 'First-Person', 'Dark Fantasy', 'Replay Value', 'Exploration'] │
    ├─────────────────────────────────────────────────────┼─────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │ Company of Heroes                                   │     0.1 │ ['Strategy', 'World War II', 'RTS', 'War', 'Action', 'Multiplayer', 'Singleplayer', 'Tactical', 'Military', 'Violent', 'Historical', 'Base Building', 'Co-op', 'Classic', 'Real Time Tactics', 'Mod', 'Moddable', 'Story Rich', 'Mature', 'Great Soundtrack']              │
    ╘═════════════════════════════════════════════════════╧═════════╧════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╛

    Top 10 game recommendations
    ╒═══════════════════════════════════════════════════════╤═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╕
    │ title                                                 │ tags                                                                                                                                                                                                                                                                                        │
    ╞═══════════════════════════════════════════════════════╪═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╡
    │ Cricket 22                                            │ ['Simulation', 'Sports', '3D', 'Realistic', 'Family Friendly', 'Singleplayer', 'Controller', 'Co-op', 'Multiplayer', 'Baseball', 'Cricket', 'Online Co-Op', 'Management', 'Artificial Intelligence', 'Early Access', 'Immersive', 'Open World', 'Gambling', 'NSFW', 'Violent']              │
    ├───────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │ Deadlock II: Shrine Wars                              │ ['Strategy', 'Classic', 'Turn-Based', 'Retro']                                                                                                                                                                                                                                              │
    ├───────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │ SAO Utils: Beta                                       │ ['Utilities', 'Anime', 'Early Access', 'Indie', 'Software']                                                                                                                                                                                                                                 │
    ├───────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │ I don't think I've walked this stretch of road before │ ['Adventure', 'Walking Simulator', 'Minimalist', 'Story Rich', '3D', 'Exploration', 'Point & Click', 'Atmospheric', 'Emotional', 'Psychological', 'Supernatural', 'Mystery', 'Third Person', 'Pixel Graphics', "1990's", 'Linear', 'Singleplayer', 'Indie']                                 │
    ├───────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │ Cherry Tree High Girls' Fight                         │ ['Card Battler', 'Anime', 'Indie', 'Simulation', 'Sports', 'Life Sim', 'Card Game', 'Multiple Endings', 'Female Protagonist', 'Nudity']                                                                                                                                                     │
    ├───────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │ VERSUS: The Deathscapes                               │ ['Action', 'Adventure', 'Casual', 'RPG', 'Interactive Fiction', 'Choose Your Own Adventure', 'Text-Based', 'Sci-fi', 'Indie', 'Superhero', 'Singleplayer']                                                                                                                                  │
    ├───────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │ Of Ships & Scoundrels                                 │ ['Action RTS', 'Real Time Tactics', 'Action', 'Pirates', 'Naval', 'Multiplayer', 'RTS', 'Crafting', 'Strategy', 'Procedural Generation', 'Indie', 'Early Access', 'Real-Time with Pause', 'Exploration', '3D', '3D Vision', 'Realistic', 'Top-Down', 'Sailing', 'Tactical']                 │
    ├───────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │ BIOS                                                  │ ['Action', 'Shooter', 'FPS', 'Zombies', 'Competitive', 'First-Person', 'Multiplayer', 'Co-op', 'Indie', 'Early Access']                                                                                                                                                                     │
    ├───────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │ Slashers                                              │ ['Psychological Horror', 'Horror', 'Action', 'Adventure', 'Casual', 'Gore', 'Early Access', 'Realistic', 'Indie', 'Strategy', 'Violent', 'Thriller', 'Dark', 'Online Co-Op', 'Multiplayer', 'Co-op', 'Action-Adventure', 'Hidden Object', 'First-Person', 'Stealth']                        │
    ├───────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │ Eco                                                   │ ['Open World Survival Craft', 'Survival', 'Multiplayer', 'Building', 'Open World', 'Crafting', 'Sandbox', 'Simulation', 'Co-op', 'Economy', 'Adventure', 'Base Building', 'Indie', 'Exploration', 'Online Co-Op', 'Early Access', 'Realistic', 'Education', 'Singleplayer', 'First-Person'] │
    ╘═══════════════════════════════════════════════════════╧═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╛

**Model 2:** User's Recommendation-Based

    Selected User ID: 7193034
    55/55 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step
    Showing recommendations for user: 7193034
    ========================================
    Games with high recommendations played by the user
    ╒═════════════════════════════════════════════════════╤══════════════════╤══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╕
    │ title                                               │ is_recommended   │ tags                                                                                                                                                                                                                                                                                                     │
    ╞═════════════════════════════════════════════════════╪══════════════════╪══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╡
    │ Paint the Town Red                                  │ True             │ ['Gore', 'Action', 'Blood', 'Fighting', 'Violent', 'Multiplayer', 'First-Person', 'Roguelike', 'Physics', 'Funny', 'Roguelite', 'Voxel', "Beat 'em up", 'Singleplayer', 'Indie', 'FPS', 'Mature', 'Adventure', 'Difficult', 'Arcade']                                                                    │
    ├─────────────────────────────────────────────────────┼──────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │ Company of Heroes                                   │ True             │ ['Strategy', 'World War II', 'RTS', 'War', 'Action', 'Multiplayer', 'Singleplayer', 'Tactical', 'Military', 'Violent', 'Historical', 'Base Building', 'Co-op', 'Classic', 'Real Time Tactics', 'Mod', 'Moddable', 'Story Rich', 'Mature', 'Great Soundtrack']                                            │
    ├─────────────────────────────────────────────────────┼──────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │ BloodRayne 2: Terminal Cut                          │ True             │ ['Action', 'Character Action Game', 'Female Protagonist', 'Vampire', 'Third-Person Shooter', 'Hack and Slash', 'Gore', 'Horror', 'Spectacle fighter', 'Blood', 'Sexual Content', 'Mature', 'Action-Adventure', 'Nudity', 'Platformer', 'Sequel', 'Third Person', 'Singleplayer', 'Adventure', 'Violent'] │
    ├─────────────────────────────────────────────────────┼──────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │ The Settlers® : Heritage of Kings - History Edition │ True             │ ['Strategy', 'Colony Sim', 'City Builder', 'RTS', 'Medieval']                                                                                                                                                                                                                                            │
    ├─────────────────────────────────────────────────────┼──────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │ Kane & Lynch 2: Dog Days                            │ True             │ ['Action', 'Co-op', 'Crime', 'Third-Person Shooter', 'Shooter', 'Violent', 'Third Person', 'Atmospheric', 'Mature', 'Multiplayer', 'Singleplayer', 'Short', 'Heist', 'Nudity', 'Story Rich', 'Open World', 'Adventure', 'Local Co-Op', 'Great Soundtrack', 'Controller']                                 │
    ╘═════════════════════════════════════════════════════╧══════════════════╧══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╛

    Top 10 game recommendations
    ╒═════════════════════════════════════════╤═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╕
    │ title                                   │ tags                                                                                                                                                                                                                                                                                        │
    ╞═════════════════════════════════════════╪═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╡
    │ The Infectious Madness of Doctor Dekker │ ['FMV', 'Lovecraftian', 'Atmospheric', 'Dark', 'Indie', 'Story Rich', 'Detective', 'Mystery', 'Psychological Horror', 'Psychological', 'Singleplayer', 'Adventure', 'Simulation', 'Horror', 'Multiple Endings', 'Drama', 'Interactive Fiction', 'Choices Matter', 'Casual', 'Visual Novel'] │
    ├─────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │ The Black Heart                         │ ['Combat', 'Fighting', 'Arcade', '2D Fighter', 'Narration', '2D', 'Dark Fantasy', 'Horror', 'Old School', 'Dark', 'Pixel Graphics', 'Story Rich', 'Atmospheric', 'Action', 'Gothic', 'Retro', 'Singleplayer', 'Local Multiplayer', 'Adventure', 'Cartoon']                                  │
    ├─────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │ The Tarnishing of Juxtia                │ ['Action', 'RPG', 'Dark Fantasy', 'Souls-like', 'Combat', 'Pixel Graphics', '2D', 'Singleplayer', 'Platformer', '2D Platformer', 'Story Rich', 'Lore-Rich', 'Dark', 'Metroidvania', 'Atmospheric', 'Character Customization', 'Multiple Endings', 'Indie', 'Violent', 'Gore']               │
    ├─────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │ A Wolf in Autumn                        │ ['Adventure', 'Indie', 'Horror', 'Psychological Horror', 'Exploration', 'Story Rich', 'Atmospheric', 'First-Person', 'Female Protagonist', 'Singleplayer', 'Dark', 'Blood', 'Surreal', 'Great Soundtrack', 'Short', 'Puzzle', 'Walking Simulator', 'Point & Click', 'Investigation', '3D']  │
    ├─────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │ Siege of Treboulain                     │ ['RPG', 'Interactive Fiction', 'Choose Your Own Adventure', 'Action RPG', 'Text-Based', 'First-Person', 'Fantasy', 'Medieval', 'War', 'Casual', 'Indie', 'Character Customization', 'Choices Matter', 'Adventure', 'Multiple Endings', 'Action', 'Story Rich', 'Singleplayer']              │
    ├─────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │ Colorado Cocoa Club                     │ ['Sexual Content', 'Visual Novel', 'LGBTQ+', 'Nudity', 'Anime', 'Dating Sim', 'NSFW', 'Mature', 'Memes', 'Cartoon', 'Simulation', 'Casual', 'Romance', 'Choose Your Own Adventure', 'Choices Matter', '2D', 'Cartoony', 'Cute', 'America', 'Comedy']                                        │
    ├─────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │ 10mg: Always Down                       │ ['Adventure', 'Experimental', 'Exploration', 'Interactive Fiction', 'Platformer', '2D Platformer', 'Puzzle Platformer', 'Metroidvania', 'Side Scroller', '2D', 'Abstract', 'Minimalist', 'Indie', 'Pixel Graphics', 'Stylized', 'Character Customization', 'Singleplayer']                  │
    ├─────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │ Autocraft                               │ ['Sandbox', 'Simulation', 'Building', 'Indie', 'Physics', 'Casual', 'Space', 'Crafting', 'Singleplayer', 'Early Access', 'Multiplayer', 'Space Sim', 'Open World']                                                                                                                          │
    ├─────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │ Robo Do It                              │ ['Casual', 'Strategy', 'Indie', 'Simulation', 'Adventure', 'Puzzle', 'Level Editor', 'Puzzle Platformer', 'Arcade', 'Programming']                                                                                                                                                          │
    ├─────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │ There Is No Game: Wrong Dimension       │ ['Indie', 'Adventure', 'Casual', 'Point & Click', 'Pixel Graphics', 'Comedy', 'Funny', 'Puzzle', 'Story Rich', 'Singleplayer', '2D', 'Narration', 'Parody', 'Romance', 'Interactive Fiction', 'Great Soundtrack', 'Simulation', 'Remake', 'RPG', 'Action']                                  │
    ╘═════════════════════════════════════════╧═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╛




**Model 3:** Adjusted Hours-Based

    Selected User ID: 7193034
    55/55 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step
    Showing recommendations for user: 7193034
    ========================================
    Games with high adjusted hours played by the user
    ╒═════════════════════════════════════════════════════╤══════════════════╤════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╕
    │ title                                               │   adjusted_hours │ tags                                                                                                                                                                                                                                                                       │
    ╞═════════════════════════════════════════════════════╪══════════════════╪════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╡
    │ Paint the Town Red                                  │            6.375 │ ['Gore', 'Action', 'Blood', 'Fighting', 'Violent', 'Multiplayer', 'First-Person', 'Roguelike', 'Physics', 'Funny', 'Roguelite', 'Voxel', "Beat 'em up", 'Singleplayer', 'Indie', 'FPS', 'Mature', 'Adventure', 'Difficult', 'Arcade']                                      │
    ├─────────────────────────────────────────────────────┼──────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │ Kane & Lynch 2: Dog Days                            │            0.75  │ ['Action', 'Co-op', 'Crime', 'Third-Person Shooter', 'Shooter', 'Violent', 'Third Person', 'Atmospheric', 'Mature', 'Multiplayer', 'Singleplayer', 'Short', 'Heist', 'Nudity', 'Story Rich', 'Open World', 'Adventure', 'Local Co-Op', 'Great Soundtrack', 'Controller']   │
    ├─────────────────────────────────────────────────────┼──────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │ The Settlers® : Heritage of Kings - History Edition │            0.375 │ ['Strategy', 'Colony Sim', 'City Builder', 'RTS', 'Medieval']                                                                                                                                                                                                              │
    ├─────────────────────────────────────────────────────┼──────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │ Gothic® 3                                           │            0.15  │ ['RPG', 'Open World', 'Fantasy', 'Action', 'Singleplayer', 'Atmospheric', 'Third Person', 'Medieval', 'Gothic', 'Adventure', 'Great Soundtrack', 'Story Rich', 'Magic', 'Action RPG', 'Sandbox', 'Classic', 'First-Person', 'Dark Fantasy', 'Replay Value', 'Exploration'] │
    ├─────────────────────────────────────────────────────┼──────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │ Company of Heroes                                   │            0.125 │ ['Strategy', 'World War II', 'RTS', 'War', 'Action', 'Multiplayer', 'Singleplayer', 'Tactical', 'Military', 'Violent', 'Historical', 'Base Building', 'Co-op', 'Classic', 'Real Time Tactics', 'Mod', 'Moddable', 'Story Rich', 'Mature', 'Great Soundtrack']              │
    ╘═════════════════════════════════════════════════════╧══════════════════╧════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╛

    Top 10 game recommendations
    ╒════════════════════════════════════════╤═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╕
    │ title                                  │ tags                                                                                                                                                                                                                                                                                                  │
    ╞════════════════════════════════════════╪═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╡
    │ Dreamin' Her - 僕は、彼女の夢を見る。- │ ['Simulation', 'Adventure', 'Visual Novel', 'Dating Sim', 'Choose Your Own Adventure', 'Cute', '2D', 'Emotional', 'Drama', 'Sexual Content', 'Romance', 'Story Rich', 'Violent', 'Choices Matter', 'Singleplayer']                                                                                    │
    ├────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │ Find Yourself                          │ ['Simulation', 'Horror', 'Exploration', 'Walking Simulator', '3D Platformer', 'Cinematic', 'First-Person', 'Psychedelic', 'Realistic', 'Psychological Horror', 'Indie', 'Violent', 'Thriller', 'Story Rich', 'Singleplayer', 'Dark Humor', 'Nudity', 'Dark', 'Short', 'Adventure']                    │
    ├────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │ Sudocats                               │ ['Casual', 'Cats', 'Puzzle', 'Relaxing', 'Cute', 'Logic', 'Hand-drawn', 'Wholesome', '2D', 'Cozy', 'Education', 'Tabletop', 'Cartoony', 'Colorful', 'Minimalist', 'Family Friendly', 'Singleplayer', 'Mouse only', 'Stylized', 'Indie']                                                               │
    ├────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │ The Cyclist: Tactics                   │ ['Sports', 'Strategy', 'Cycling', 'Management', 'Turn-Based Tactics', 'Tactical', '2D', 'Hand-drawn', 'Top-Down', 'Turn-Based Strategy', 'Singleplayer']                                                                                                                                              │
    ├────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │ Joe Danger 2: The Movie                │ ['Racing', 'Indie', 'Action', 'Casual', 'Controller', 'Score Attack', 'Local Multiplayer', 'Platformer', 'Funny', 'Arcade', 'Local Co-Op']                                                                                                                                                            │
    ├────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │ The Note of Red Evil                   │ ['RPG', 'Indie', 'Casual', 'Strategy', 'Adventure', 'Nudity']                                                                                                                                                                                                                                         │
    ├────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │ Daemonic Runner                        │ ['Action', 'Indie', 'Violent', 'Retro', 'Gore', 'Fast-Paced', 'First-Person', 'Gothic', 'Shooter', 'FPS', 'Classic', 'Singleplayer', 'Demons']                                                                                                                                                        │
    ├────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │ Rabbit Island                          │ ['Strategy', 'Indie', 'Tower Defense']                                                                                                                                                                                                                                                                │
    ├────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │ Onset                                  │ ['RPG', 'Massively Multiplayer', 'Simulation', 'Action', 'Indie', 'Open World', 'Multiplayer', 'Sandbox', 'Early Access']                                                                                                                                                                             │
    ├────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │ Fight Ascending                        │ ['Female Protagonist', 'Martial Arts', '3D Fighter', 'Procedural Generation', "Beat 'em up", 'Spectacle fighter', 'Sexual Content', 'Nudity', 'Mature', 'NSFW', 'Arcade', 'Casual', 'Fast-Paced', 'Fighting', 'Action', 'Fantasy', 'Combat', 'Singleplayer', 'Character Action Game', 'Third Person'] │
    ╘════════════════════════════════════════╧═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╛


#### **Best Model**


Dipilih  **Model adjusted hours based**  sebagai model terbaik karena mampu memprediksi dengan tingkat kesalahan paling minimal yaitu  **RMSE**  sebesar 0.0269 dari data pelatihan dan sebesar 0.0446 dalam pengujian. Selain itu model ini pula adalah gabungan dari dua fitur yang diuji sehingga didapat korelasi yang lebih kompleks antar game dan antar user di dalam model ini. Penjelasan lebih detail mengenai evaluasi akan dijelaskan di rubrik selanjutnya.

## **Evaluation**

### **1. Content Based Filtering**

Metrik yang digunakan dalam evaluasi model content-based filtering meliputi Precision@k, Recall@k, F1@k, dan MRR@k. Sebelum membahas hasil evaluasi, berikut adalah penjelasan tentang cara menghitung masing-masing metrik serta penggunaan confusion matrix untuk mengukur performa model.

Sekilas tentang `Confusion Matrix` dan Metrik Evaluasi

`Confusion Matrix` adalah tabel yang digunakan untuk mengevaluasi performa model klasifikasi dengan mengukur jumlah prediksi yang benar dan salah berdasarkan label aktual dan prediksi. Setiap baris dalam `confusion matrix` mewakili nilai sebenarnya (`actual`), sedangkan setiap kolom mewakili nilai prediksi (`predicted`). Komponen utama dari `confusion matrix` adalah sebagai berikut:
- **True Positive (TP)**: Jumlah data positif yang diprediksi benar.
- **True Negative (TN)**: Jumlah data negatif yang diprediksi benar.
- **False Positive (FP)**: Jumlah data negatif yang salah diprediksi sebagai positif (*Error Tipe 1*).
- **False Negative (FN)**: Jumlah data positif yang salah diprediksi sebagai negatif (*Error Tipe 2*).

![png](gambar_files/gambar_188_0.png)


  

**Metrik Evaluasi @k**

  

1. **Precision@k**

$Precision@k$ mengukur seberapa banyak rekomendasi yang relevan dalam $top-k$ rekomendasi. Ini dihitung dengan rumus:

  

$$
\text{Precision@k} = \frac{\text{Jumlah item relevan dalam top-k}}{k}
$$

  

Di mana:

- $k$ adalah jumlah rekomendasi teratas yang diberikan oleh model.

  

2. **Recall@k**

$Recall@k$ mengukur seberapa baik model dalam menemukan semua item relevan dalam $top-k$ rekomendasi dibandingkan dengan total item relevan yang ada. Rumusnya adalah:

  

$$
\text{Recall@k} = \frac{\text{Jumlah item relevan dalam top-k}}{\text{Total item relevan yang tersedia}}
$$

  

3. **F1@k**

$F1@k$ adalah rata-rata harmonik antara $Precision@k$ dan $Recall@k$, yang memberikan keseimbangan antara keduanya. Rumusnya adalah:

  

$$
F1@k = 2 \cdot \frac{\text{Precision@k} \cdot \text{Recall@k}}{\text{Precision@k} + \text{Recall@k}}
$$

  

4. **Accuracy@k**

$Accuracy@k$ adalah metrik yang digunakan untuk mengukur seberapa akurat model dalam memberikan rekomendasi terbaik. Metrik ini mengukur proporsi item relevan yang muncul dalam $top-k$ rekomendasi dibandingkan dengan seluruh rekomendasi yang diprediksi oleh model. $Accuracy@k$ memberi gambaran tentang seberapa sering item relevan muncul dalam daftar teratas rekomendasi.

  

Rumus untuk menghitung $Accuracy@k$ adalah:

  

$$
\text{Accuracy@k} = \frac{\text{Jumlah item relevan dalam top-k}}{k}
$$

  

Di mana:

- $k$ adalah jumlah rekomendasi teratas yang diberikan oleh model.

- Jumlah item relevan dalam $top-k$ adalah jumlah item yang benar-benar relevan dan ada di dalam urutan rekomendasi teratas.

  

5. **MRR@k (Mean Reciprocal Rank)**

$MRR@k$ mengukur kualitas urutan rekomendasi berdasarkan posisi item relevan pertama yang ditemukan dalam $top-k$ rekomendasi. Metrik ini sangat berguna ketika urutan rekomendasi memiliki peran penting, dan kita hanya tertarik pada posisi pertama dari item relevan yang ditemukan oleh model.

Rumus $MRR@k$ adalah:

  

$$
\text{MRR@k} = \frac{1}{Q} \sum_{i=1}^Q \frac{1}{\text{Rank}_i}
$$

  

Di mana:

- $Q$ adalah jumlah total query atau pengguna.

- $Rank_i$ adalah posisi relevan pertama untuk pengguna ke-i.

  

$Reciprocal Rank (RR)$ dihitung dengan rumus:

  

$$
\text{RR} = \frac{1}{\text{Rank of first relevant item}}
$$

  

Di dunia nyata, pengguna jarang melihat semua rekomendasi, biasanya hanya $top-k$ (misalnya, 5 atau 10 teratas). Dengan menggunakan $@k$, fokus evaluasi dapat diarahkan pada rekomendasi terbaik yang diberikan model. Nilai $k$ juga dapat disesuaikan dengan jumlah rekomendasi yang relevan untuk aplikasi tertentu. Selain itu, $MRR@k$ memberikan perhatian khusus pada posisi item relevan pertama dalam urutan rekomendasi, yang penting dalam sistem yang mengutamakan urutan penyajian rekomendasi kepada pengguna.

**Model 1:** Cosine Similarity (descriptions)
    
![png](gambar_files/gambar_176_0.png)


    Evaluation for Striving for Light: Survival:
    Precision@5: 1.00
    Recall@5: 1.00
    Mean Reciprocal Rank (MRR): 1.00
    Accuracy: 1.00
    
    Confusion Matrix:
    [[ 0  0]
     [ 0 10]]
    
    Classification Report:
                  precision    recall  f1-score   support
    
      Irrelevant       0.00      0.00      0.00         0
        Relevant       1.00      1.00      1.00        10
    
        accuracy                           1.00        10
       macro avg       0.50      0.50      0.50        10
    weighted avg       1.00      1.00      1.00        10
    




**Model 2:** Cosine Similarity (tags)

    
![png](gambar_files/gambar_177_0.png)
    


    Evaluation for Striving for Light: Survival:
    Precision@5: 1.00
    Recall@5: 1.00
    Mean Reciprocal Rank (MRR): 1.00
    Accuracy: 1.00
    
    Confusion Matrix:
    [[ 0  0]
     [ 0 10]]
    
    Classification Report:
                  precision    recall  f1-score   support
    
      Irrelevant       0.00      0.00      0.00         0
        Relevant       1.00      1.00      1.00        10
    
        accuracy                           1.00        10
       macro avg       0.50      0.50      0.50        10
    weighted avg       1.00      1.00      1.00        10
    




**Model 3:** Cosine Similarity (numerical features)

    
![png](gambar_files/gambar_178_0.png)
    


    Evaluation for Striving for Light: Survival:
    Precision@5: 1.00
    Recall@5: 1.00
    Mean Reciprocal Rank (MRR): 1.00
    Accuracy: 1.00
    
    Confusion Matrix:
    [[ 0  0]
     [ 0 10]]
    
    Classification Report:
                  precision    recall  f1-score   support
    
      Irrelevant       0.00      0.00      0.00         0
        Relevant       1.00      1.00      1.00        10
    
        accuracy                           1.00        10
       macro avg       0.50      0.50      0.50        10
    weighted avg       1.00      1.00      1.00        10
    




**Model 4:** Cosine Similarity (combined features)

    
![png](gambar_files/gambar_179_0.png)
    


    Evaluation for Striving for Light: Survival:
    Precision@5: 1.00
    Recall@5: 1.00
    Mean Reciprocal Rank (MRR): 1.00
    Accuracy: 1.00
    
    Confusion Matrix:
    [[ 0  0]
     [ 0 10]]
    
    Classification Report:
                  precision    recall  f1-score   support
    
      Irrelevant       0.00      0.00      0.00         0
        Relevant       1.00      1.00      1.00        10
    
        accuracy                           1.00        10
       macro avg       0.50      0.50      0.50        10
    weighted avg       1.00      1.00      1.00        10
    


Empat model content-based filtering diuji dengan menggunakan berbagai fitur untuk membuat rekomendasi. Model pertama, yang menggunakan *Cosine Similarity* berbasis deskripsi, mengevaluasi kesamaan antara item berdasarkan teks deskripsi mereka. Model kedua, yang mengandalkan fitur tag, mengukur kesamaan antar item berdasarkan tag yang relevan yang dimiliki setiap item. Model ketiga menggunakan fitur numerik, yang memanfaatkan data seperti harga atau rating untuk menghitung kesamaan antar item. Model keempat, yang merupakan gabungan dari ketiga model sebelumnya, mengintegrasikan deskripsi, tag, dan fitur numerik untuk memberikan rekomendasi yang lebih komprehensif dan akurat.

Model diuji menggunakan berbagai metrik evaluasi, seperti Precision@k, Recall@k, F1@k, MRR@k, dan confusion matrix. Hasil simulasi menunjukkan bahwa semua model berhasil mencapai 100% akurasi dalam prediksi.

Dengan nilai Precision@k yang mencapai 100%, model berhasil memberikan rekomendasi yang relevan secara konsisten dalam top-k yang diprediksi. Artinya, setiap rekomendasi yang diberikan kepada pengguna adalah item yang relevan, menunjukkan bahwa model sangat akurat dalam menyeleksi item yang sesuai dengan preferensi pengguna. Nilai Recall@k yang juga mencapai 100% menunjukkan bahwa model dapat menemukan semua item relevan yang tersedia dalam daftar rekomendasi top-k. Ini menunjukkan bahwa tidak ada item relevan yang terlewat dalam rekomendasi.

F1@k, yang menggabungkan Precision@k dan Recall@k, juga memperoleh nilai 100%. Ini menunjukkan bahwa model tidak hanya memberikan rekomendasi yang relevan, tetapi juga berhasil menangkap seluruh item relevan yang ada, tanpa mengorbankan keseimbangan antara akurasi dan kelengkapan. Sementara itu, MRR@k menunjukkan bahwa model selalu menempatkan item relevan pertama pada urutan yang tepat atau sangat dekat dengan urutan pertama dalam rekomendasi. Hal ini menunjukkan bahwa posisi item relevan dalam daftar rekomendasi sangat baik, yang penting untuk meningkatkan pengalaman pengguna.

Confusion matrix menunjukkan bahwa semua prediksi yang dihasilkan oleh model adalah True Positive (TP), yaitu semua item relevan diprediksi dengan benar sebagai relevan, dan tidak ada kesalahan prediksi (False Positive atau False Negative). Ini mengindikasikan bahwa model bekerja dengan sangat tepat dalam mengklasifikasikan item relevan.

Secara keseluruhan, hasil evaluasi yang menunjukkan nilai 100% untuk semua metrik utama ini menunjukkan bahwa model content-based filtering yang digunakan sangat efektif dalam memberikan rekomendasi yang relevan dan akurat. Metrik evaluasi yang baik, termasuk MRR@k, menunjukkan bahwa model ini tidak hanya memberikan rekomendasi yang tepat, tetapi juga memperhatikan kualitas urutan rekomendasi, yang penting dalam konteks sistem rekomendasi berbasis urutan.




### **2. Collaborative Filtering**

Model *collaborative filtering* ini, metrik evaluasi yang digunakan adalah **Root Mean Squared Error (RMSE)**.

  **Sekilas tentang RMSE**

**Root Mean Squared Error (RMSE)** adalah salah satu metode untuk mengukur kesalahan pada pelatihan model dengan menghitung jarak rata-rata antara nilai prediksi dan nilai aktual. RMSE dapat dihitung dengan rumus berikut:

$$
\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

Keterangan:
- $y_i$: Nilai aktual pada observasi ke-`i`
- $\hat{y}_i$: Nilai prediksi pada observasi ke-`i`
- $N$: Jumlah observasi

Jika nilai prediksi mendekati nilai sesungguhnya, maka selisih antara $(y_i - \hat{y}_i)$ akan semakin kecil. Artinya, semakin kecil nilai RMSE atau semakin mendekati nol, maka model yang digunakan semakin akurat dan baik.

  **Penerapan Evaluasi Model dengan RMSE**

Pada *collaborative filtering*, setelah melatih model selama 200 epoch, diperoleh nilai RMSE sebesar 0.0315 untuk data pelatihan dan 0.1886 untuk data pengujian. Berikut adalah nilai RMSE untuk tiga model yang diuji:

- **RMSE Model hours based**: 0.0296 (data pelatihan) dan 0.0457 (data pengujian)
- **RMSE Model user's recommendation**: 0.3088 (data pelatihan) dan 0.3558(data pengujian)
- **RMSE Model adjusted hours based**: 0.0269 (data pelatihan) dan 0.0446 (data pengujian)

Jika dilihat melalui grafik, hasilnya dapat dilihat pada plot berikut.



**RMSE historical graph: Model hours based**

    
![png](gambar_files/gambar_184_0.png)
    


**RMSE historical graph: Model user's recommendation**



    
![png](gambar_files/gambar_185_0.png)
    



**RMSE historical graph: Model adjusted hours based**


    
![png](gambar_files/gambar_186_0.png)
    


Plot tersebut menunjukkan, bahwa nilai RMSE pada data pelatihan dan pengujian terus menurun tajam, tetapi setelah 10 epoch, nilai RMSE mulai stagnan untuk model hours-based dan adjusted-hours-based. Meskipun RMSE pada data pengujian lebih besar dibandingkan dengan data pelatihan, keduanya memiliki nilai yang sangat mendekati 0. Oleh karena itu, model ini dapat dianggap baik dan akurat untuk digunakan dalam sistem rekomendasi. Sedangkan model user's recommendation nilai RMSE pelatihan dan pengujian sama-sama menurun dengan gradient yang stabil. Hanya saja metrik pengujian memiliki slope yang lebih landai dan menuju nilai 0.35 pada RMSE. Menunjukkan bahwa model tidak dapat meningkat lagi akurasinya dan diperlukan perlakuan tambahan untuk mengoptimalkan modelnya.


Nilai **Root Mean Squared Error (RMSE)** yang diperoleh pada setiap model dapat diinterpretasikan dalam konteks data yang memiliki rentang antara 0 hingga 1, memberikan gambaran yang lebih jelas tentang kualitas prediksi model. Model dengan fitur `hours`, memiliki nilai RMSE sebesar **0.0457** pada data pengujian berarti bahwa rata-rata kesalahan prediksi model adalah sekitar **4.57%** dari nilai maksimum yaitu 1. Ini menunjukkan bahwa meskipun terdapat sedikit kesalahan dalam prediksi, tingkat kesalahan tersebut sangat kecil jika dibandingkan dengan rentang data yang ada. Dengan demikian, model ini memiliki tingkat akurasi yang sangat baik.

Untuk model dengan fitur `is_recommended`, nilai RMSE yang lebih tinggi yaitu **0.3088** pada data pelatihan dan **0.3558** pada data pengujian menunjukkan bahwa model ini kurang akurat dalam memprediksi rekomendasi yang relevan. Nilai RMSE ini setara dengan **30.88%** dan **35.58%** dari nilai maksimum, yang berarti model ini memiliki kesalahan prediksi yang cukup besar jika dibandingkan dengan model lainnya. Namun, meskipun nilai RMSE lebih tinggi, model ini masih dapat diterima tergantung pada konteks dan aplikasi penggunaan, terutama jika rekomendasi berbasis `is_recommended` memiliki variasi yang lebih kompleks.

Sementara itu, untuk model dengan `adjusted_hours`, nilai RMSE yang diperoleh adalah **0.0269** pada data pelatihan dan **0.0446** pada data pengujian. Nilai RMSE ini setara dengan **2.62%** pada data pelatihan dan **4.46%** pada data pengujian, yang menunjukkan bahwa model ini juga mampu memberikan prediksi yang akurat, dengan kesalahan yang sangat kecil. Seperti model `hours`, model ini memiliki performa yang sangat baik dalam memprediksi data dengan rentang 0 hingga 1, dengan tingkat kesalahan yang relatif rendah.

Secara keseluruhan, meskipun ada perbedaan dalam nilai RMSE antar model, semua nilai RMSE yang diperoleh berada dalam kisaran yang sangat kecil jika dibandingkan dengan skala data 0 hingga 1. Hal ini menunjukkan bahwa model-model ini, meskipun memiliki karakteristik yang berbeda, mampu menghasilkan prediksi yang sangat mendekati nilai aktual, dengan kesalahan yang minimal, yang menjadikannya sangat baik untuk digunakan dalam sistem rekomendasi berbasis data dengan rentang terbatas seperti ini.

## **Kesimpulan**

Berikut adalah kesimpulan dari _goals_ yang telah dicapai secara singkat:

1. Model rekomendasi game yang dikembangkan berhasil dibuat untuk pengguna dalam memilih game berdasarkan kesamaan karakteristik game dan pengguna. Model terbaik content based filtering adalah **Model 3: Cosine Similarity (Numerical Features)** dengan **Precision@k=10** bernilai **1**. Model terbaik collaborative filtering adalah **Model adjusted hours based** dengan nilai **RMSE** pengujian **0.0446**.
2. Analisis game berdasarkan kepopuleran yang diwakili oleh total durasi bermain, jumlah review, dan jumlah rekomendasi memberikan wawasan tentang kepopuleran dan interaksi pengguna. Game teratas dalam kategori **total durasi dimainkan** adalah **Team Fortress 2**, kategori **jumlah rekomendasi** adalah **Team Fortress 2**, dan kategori **jumlah ulasan** adalah **Counter Strike: Global Offensive**.
3. Pengaruh sistem operasi terhadap preferensi game pengguna menunjukkan perbedaan kecenderungan berdasarkan platform yang digunakan. Yaitu **windows** adalah os yang paling populer untuk bermain game karena banyaknya game yang rilis di platform ini. Sedangkan **linux** memiliki kepopuleran tertinggi jika memeperhitungkan jumlah game yang dirilis disana.
4. Hubungan antara distribusi harga game dan tingkat ulasan positif mengungkapkan dampak harga terhadap persepsi kualitas game oleh pengguna. Secara umum **tidak terlalu signifikan** hubungan keduanya namun secara spesifik terdapat sedikit pola hubungan yang menunjukkan game yang lebih tinggi harganya memiliki lebih banyak ulasan positif pada rentang harga menengah.
5. Analisis hubungan antara rating game dan waktu yang dihabiskan pengguna untuk memainkannya memperlihatkan keterkaitan antara kualitas game dan tingkat keterlibatan pengguna. Pengguna lebih banyak menghabiskan waktunya untuk game dengan **rating bernada positif**.

Model rekomendasi game berhasil dikembangkan secara optimal menggunakan content-based filtering dan collaborative filtering. Exploratory Data Analysis juga memberikan wawasan tentang kepopuleran game, pengaruh sistem operasi terhadap preferensi pengguna, dampak harga terhadap ulasan positif, serta keterkaitan rating dengan waktu bermain, yang secara keseluruhan mendukung pemahaman dan pengalaman pengguna dalam pembelian game.

## **Reference**

1. Granic, I., Lobel, A., & Engels, R. C. M. E. (2014). *The Benefits of Playing Video Games*. *American Psychologist*, 69(1), 66-78. Diakses dari [https://doi.org/10.1037/a0034857](https://doi.org/10.1037/a0034857).

2. Primack, B. A., Carroll, M. V., McNamara, M., et al. (2012). *Role of Video Games in Improving Health-Related Outcomes: A Systematic Review*. *American Journal of Preventive Medicine*, 42(6), 630-638. Diakses dari [https://doi.org/10.1016/j.amepre.2012.02.023](https://doi.org/10.1016/j.amepre.2012.02.023).

3. McKinsey & Company. (2021). *Personalization: How to Capture Value*. Diakses dari [https://www.mckinsey.com/business-functions/marketing-and-sales/our-insights/the-future-of-personalization-and-how-to-get-ready-for-it](https://www.mckinsey.com/business-functions/marketing-and-sales/our-insights/the-future-of-personalization-and-how-to-get-ready-for-it).

4. LeCun, Y. (2018). *Deep Learning: A Revolution in AI*. *Nature AI*. Diakses dari [https://www.nature.com/articles/d41586-018-05084-7](https://www.nature.com/articles/d41586-018-05084-7).

5. Anderson, C. (2006). *The Long Tail: Why the Future of Business is Selling Less of More*. Hyperion. Cuplikan dapat diakses dari [https://books.google.com](https://books.google.com).

6. Resnick, P., & Varian, H. R. (1997). *Recommender Systems*. *Communications of the ACM*, 40(3), 56-58. Diakses dari [https://dl.acm.org/doi/10.1145/963770.963772](https://dl.acm.org/doi/10.1145/963770.963772).

7. Valve Corporation. *Steam Discovery Queue Impact Report*. Valve's Developer Blog. Diakses dari [https://store.steampowered.com/news/](https://store.steampowered.com/news/).

8. Gee, J. P. (2003). *What Video Games Have to Teach Us About Learning and Literacy*. Palgrave Macmillan.

9. Kowert, R., & Quandt, T. (2016). *The Video Game Debate: Unravelling the Physical, Social, and Psychological Effects of Video Games*. Routledge.

10. Staiano, A. E., & Calvert, S. L. (2011). *Exergames for Physical Education: Developing and Leading Active Video Game Exercises*. *Games for Health Journal*, 1(1), 35-39.
