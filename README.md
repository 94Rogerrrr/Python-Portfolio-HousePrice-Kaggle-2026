# Python-Portfolio-HousePrice_Kaggle_20260305
It's a repository for Python data analysis portfolio about Kaggle Data - House Price Prediction. Here are .ipynb, requirement.txt, README.md in this repository.

The project is trying to evaluate the house price is underestimated by building a house price predicting model according to the Kaggle Dataset(https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset)

first problem, there are 80+ columns, how do I be able to check every column has non, mean, or outlayer?
## 1. 專案標題與一句話總結 (Title & Hook)
- **標題**：House Price Prediction: Advanced Regression Techniques
- **摘要**：利用 XGBoost 與 Lasso 回歸模型預測房價，成功將誤差 (RMSE) 降低至 0.12，並識別出「廚房品質」與「居住面積」為影響房價的最關鍵因素。
## 2. 商業洞察 (Business Insights) —— _這是你最有價值的地方_
### 相關係數：
- 將文字資料先做 target encoding 轉換成數值類型後，計算整份資料之相關係數。
- 房價相關超過 0.5：OverallQual     0.790982/Neighborhood    0.726538/GrLivArea       0.708624...... 詳見代碼。
- 房價最高相關的欄位為 OverallQual 整體品質，其次為 Neighborhood 所屬社區，再者 GrLivArea 地面上居住面積。
- 文字類型共 7 項，數值類型共 10 項。將針對彼此欄位定義制定相對的資料整理邏輯。
- 查看負相關的數值，發現最低相關為 KitchenAbvGr -0.135907 然而其欄位在說明手冊上卻沒有定義。依據欄位名稱拆解，應該是屬於地面上廚房的相關內容，但不確定單位為何。
- 
### 箱型圖解讀
- 從第一行中間的 GrLivArea 及中間三張圖，GarageArea、 TotalBsmtSF、1stFlrSF 的圖片與離峰值的位置，可判斷這些欄位之資料並非常態分佈，明顯有左偏分布的傾向。
- 實際上是否為左偏分布的狀態，應依據 skew 數值結果

#### 散佈圖解讀
- 離峰值判讀：遠離聚集或是遠離趨勢的點。
- 從第一行中間的 GrLivArea 及中間三張圖，GarageArea、 TotalBsmtSF、1stFlrSF 的圖片，離散與密集程度，可判斷這些欄位之資料具有明顯的離峰值。


_(這裡放一張漂亮的 SHAP 或特徵重要性圖表)_
- 發現 1：翻修廚房比增建泳池更能顯著提升房價。
- 發現 2：雖然坪數重要，但當屋齡超過 20 年後，坪數對價格的邊際效應會遞減。
## 3. 資料處理 (Data Methodology) —— _這裡寫剛剛的健檢結果_
- **清洗**：移除了 2 筆極端異常值與邏輯錯誤資料。
- **填補**：
- 針對 object 資料型態，根據有 null 的欄位去比對欄位手冊 txt 檔案，發現 Electical 沒有定義 NA 的項目，顯然這筆資料明顯有缺失
- MasVnrType 定義為房屋工程上的外牆貼磚，要有貼磚 MasVnrType 才會有貼磚面積 MasVnrArea。發現沒有貼磚材質類型卻有貼磚面積的衝突資料，並且採信貼磚面積，判斷是缺少貼磚類型，共 5 筆。針對字串類型資料，這 5 筆採用眾數方式填入。
- MasVnrType 與 MasVnrArea 若兩者皆為 NaN 屬於合理範圍，表示房子沒有貼磚，故無貼磚面積。共 8 筆
## 4. 模型表現 (Model Performance)
- Baseline (Linear): RMSE 0.18
- Final Model (XGBoost): RMSE 0.12 (**提升 33%**)
