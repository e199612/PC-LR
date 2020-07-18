# PC-LR
利用不同複雜度的模型來對Top-N推薦排序做訓練，並分析不同規模電商的資料集對複雜度不同的模型的影響與結果，從中找到使用不同複雜度模型的時機。
此部分只包含阿里巴巴深度學習模型DIN和DIEN模型的訓練過程，其他部分在易哲的github連結中。

以下附上論文連結：
https://drive.google.com/file/d/1b8f5fRo7R-sEQr59R_9GK7zip9Zuu0cG/view?usp=sharing

# 實驗說明
* 使用的模型有：KNN、MLP、DIN、DIEN
* 使用的資料集：阿里巴巴淘寶資料集、電商A資料集

# 資料夾說明
* ALI_DIEN、ALI_DIN、DART_DIEN和DART_DIN是說明如何前處理資料
* DIEN_train和DIN_train是說明如何訓練不同資料集下不同的深度學習模型
