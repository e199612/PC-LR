環境準備
===
**1.** 安裝XDL環境(推薦使用docker方式執行XDL提供的ubuntu鏡像：registry.cn-hangzhou.aliyuncs.com/xdl/xdl:ubuntu-cpu-tf1.12)

**2.** 數據準備
  * 按照data/README.md的連結下載DIEN使用的數據資料，包括以下7個檔案：
    * cat_voc.pkl
    * mid_voc.pkl
    * uid_voc.pkl
    * local_train_splitByUser
    * local_test_splitByUser
    * reviews-info
    * item-info

**3.** 改寫config.json中的data_dir路徑位置，例如：../data

訓練
===
**1.** 在本機上安裝docker

**2.** 鏡入docker鏡像，並將對應的資料目錄位置掛載進docker內：
```
sudo docker run --net=host -v [path_to_dien]:/home/xxx/DIEN -it registry.cn-hangzhou.aliyuncs.com/xdl/xdl:ubuntu-cpu-tf1.12 /bin/bash
```
  * [path_to_dien]是本機中你檔案目錄的位址，比如說是放在桌面的某個資料夾中，則會長成這樣：
  
    `C:/Users/DART/Desktop/x-deeplearning-master/xdl-algorithm-solution/DIEN:/home/xxx/DIEN`

**3.** 在docker中執行以下命令開始訓練：

```
cd /home/xxx/DIEN/script
python train.py --run_model=local --config=config.json
```

測試
===
**1.** 先將data中的要測試的資料方在對應的檔案目錄裡

**2.** 以阿里巴巴的資料為例，在 `ali_test_restore.py` 中：

```
elif job_type == 'test':
    filename = glob.glob('../data/ali_dien_test_data/*.pkl')
    filename.sort()
    print(filename)
    for n in filename[16:]:
        print(n)
        test(test_file=n)
```

  將要測試的測試資料從這邊輸入，最後再將測試資料的預測結果包成一個pkl檔案當作輸出：
  
```
results = list(zip(user, hist, last_hist, target_list, rank))
with open('ali_dien_rank_4days' + test_file[-11:],'wb') as d:
    pkl.dump(results,d)
```
