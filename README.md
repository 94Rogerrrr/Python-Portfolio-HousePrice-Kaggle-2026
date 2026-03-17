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
- 從第一行的 SalePrice, GrLivArea 及中間三張圖，GarageArea、 TotalBsmtSF、1stFlrSF 的圖片與離峰值的位置，可判斷這些欄位之資料並非常態分佈，明顯有右偏分布的傾向。
- 實際上是否為左偏分布的狀態，應依據 skew 數值結果

### 散佈圖解讀
- 離峰值判讀：遠離聚集或是遠離趨勢的點。
- 從第一行中間的 GrLivArea 及中間三張圖，GarageArea、 TotalBsmtSF、1stFlrSF 的圖片，離散與密集程度，可判斷這些欄位之資料具有明顯的離峰值。

#### 離峰值
- 綜合散佈圖資訊與查詢結果，發現 GrLivArea 的兩筆離峰值，也正好是其他欄位的離峰值項目。故優先刪減這兩筆資料。
- GrLivArea 的相關係數高，若有極端值易影響模型預測結果。
- 其他項目之影響不如 GrLivArea 來得高，且撇除以刪除之離峰值後，剩下有離峰值嫌疑的資料予以保留。

_(這裡放一張漂亮的 SHAP 或特徵重要性圖表)_
- 發現 1：整體品質跟房價有顯著正相關
- 發現 2：坪數與所處的社區 Neighbor 跟房價有高度相關。
- 雖然坪數重要，但當屋齡超過 20 年後，坪數對價格的邊際效應會遞減。
## 3. 資料處理 (Data Methodology) —— _這裡寫剛剛的健檢結果_
- **清洗**：
  - 若對所有特徵的離峰值都進行整筆資料刪除，此行為恐導致資料枯竭，故針對高相關係數之特徵篩除離峰值，以免其極端值帶來的高槓桿效應，嚴重影響模型學習
  - 故針對 GrLivArea 特徵，移除了 2 筆極端異常值。
  - 針對次要相關係數之離峰值，予以保留。首先，保留該觀測值，可使模型能學習到該筆資料於其他正常特徵上的資訊；其次，最高的離峰值也同 GrLivArea 的資料一起刪除；再來，其相關係數影響不如 GrLivArea 高，故衡量資料的模型學習效果與極端值的影響後，決定予以保留。
- **填補**：
  - 釐清高度相關 17 項指標，其缺失值均為文字型態的無設施，因此補 0
  - 文字資料型態：其中 7 項文字型態，分別依照手冊說明定義映射方式與映射值。Neighborhood 套用 Target Encodinge；Foundation 套用 One-Hot Encoding，其餘套用 Label Encoding。

## 4. 模型表現 (Model Performance)
- Baseline (Linear): RMSE 0.18
- Final Model (XGBoost): RMSE 0.12 (**提升 33%**)
