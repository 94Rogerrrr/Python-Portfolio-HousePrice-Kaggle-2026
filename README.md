# Python-Portfolio-HousePrice_Kaggle_20260305
It's a repository for Python data analysis portfolio about Kaggle Data - House Price Prediction. Here are .ipynb, requirement.txt, README.md in this repository.

The project is trying to evaluate the house price is underestimated by building a house price predicting model according to the Kaggle Dataset(https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset)

## 1. 專案摘要 (Title & Hook)
- **標題**：House Price Prediction: Advanced Regression Techniques
- **摘要**：比較傳統 OLS 模型、機器學習之線性模型、數種樹模型與 LassoCV 回歸模型預測房價，成功將誤差 (RMSE) 降低至 0.12，並識別出「居住面積」與「整體品質」為影響房價的最關鍵因素。
## 2. 商業洞察 (Business Insights) 
### 相關係數：
#### 在 OLS 傳統模型的資料檢查中
- 房價相關超過 0.5 之指標：OverallQual     0.80/Neighborhood    0.73/GrLivArea       0.71 ...... 詳見代碼。
- 房價最高相關的欄位為 OverallQual 整體品質，其次為 Neighborhood 所屬社區，再者 GrLivArea 地面上居住面積。
- 查看負相關的數值，發現最低相關為 KitchenAbvGr -0.135907 然而其欄位在說明手冊上卻沒有定義。依據欄位名稱拆解，應該是屬於地面上廚房的相關內容，但不確定單位為何。
**相關係數熱力圖解讀**
_(這裡放一張漂亮的 SHAP 或特徵重要性圖表)_
- 發現 1 : 文字資料經過較精確的 encode 邏輯，且對房價做常態化後，前三項相關係數有提高。因為資料經過清洗轉換可以呈現更精確的相關影響。
- 發現 2 : OverallQual 整體品質、Neighborhood 所處的社 區與 GrLivArea 坪數，與第一次跑相關係數結果相同，均與房價有高度相關。
- 發現 3 : 當地基材質 Foundation 以 One Hot Encoding 處理成不同欄位，其相關係數差異也被凸顯。
    - Foundation_PConc   = 0.53
    - Foundation_infrequent_sklearn  = - 0.30
    - Foundation_CBlock   = -0.33
<br>可知使用 PConc 材質對房價有正相關，而使用其他材質會有負相關。
#### 在 LassoCV 模型的預測結果
- 相關係數最高的特徵為 GrLivArea 地面上居住面積，次高者為 OverallQual 整體品質。顯示這兩項指標會高度影響房價。
- 多種樹模型之預測誤差小但是 RMSE Degradation (Train vs Test) 均明顯大於線性模型，故推論在本次特徵工程下，線性模型表現較佳，可以兼顧最小的均方誤差與過度擬合問題。


## 3. 資料處理 (Data Methodology) 
- #### 針對 OLS 線性模型
  - **離群值篩選解讀**：
  - 箱型圖解讀
    - 從第一行的 SalePrice, GrLivArea 及中間三張圖，GarageArea、 TotalBsmtSF、1stFlrSF 的圖片與離群值的位置，可判斷這些欄位之資料並非常態分佈，明顯有右偏分布的傾向。
    - 實際上是否為左偏分布的狀態，應依據 skew 數值結果

  - 散佈圖解讀
    - 離群值判讀：遠離聚集或是遠離趨勢的點。
    - 從第一行中間的 GrLivArea 及中間三張圖，GarageArea、 TotalBsmtSF、1stFlrSF 的圖片，離散與密集程度，可判斷這些欄位之資料具有明顯的離群值。

  - 離群值    
    - 綜合散佈圖資訊與查詢結果，發現 GrLivArea 的兩筆離群值，也正好是其他欄位的離群值項目。故優先刪減這兩筆資料。
    - GrLivArea 的相關係數高，若有極端值易影響模型預測結果。
    - 其他項目之影響不如 GrLivArea 來得高，且撇除以刪除之離群值後，剩下有離群值嫌疑的資料予以保留。

  - **清洗**：
      - 若對所有特徵的離群值都進行整筆資料刪除，此行為恐導致資料枯竭，故針對高相關係數之特徵篩除離群值，以免其極端值帶來的高槓桿效應，嚴重影響模型學習
      - 故針對 GrLivArea 特徵，移除了 2 筆極端異常值。
      - 針對次要相關係數之離群值，予以保留。首先，保留該觀測值，可使模型能學習到該筆資料於其他正常特徵上的資訊；其次，最高的離群值也同 GrLivArea 的資料一起刪除；再來，其相關係數影響不如 GrLivArea 高，故衡量資料的模型學習效果與極端值的影響後，決定予以保留。
      - 套用 Skewness 檢查資料分佈的偏移狀態，針對分數 > 0.5 之指標套用常態化，對此類指標取 log 。
      - 將文字資料先做 target encoding 轉換成數值類型後，計算整份資料之相關係數，選擇其中相關係數高的指標：文字類型共 7 項，數值類型共 10 項。將針對彼此欄位定義制定相對的資料整理邏輯。
      - 17 項指標單位不同以至於數據的尺度差異大，故使用 StandardScaler 降低尺度差異造成的影響。
      - 第一次執行 OLS 模型評估指標顯示有較明顯的異質變異數問題，故刪除殘差分佈超過三個標準差之資料，共 14 筆。
  - **填補**：
      - 釐清高度相關 17 項指標，其缺失值均為文字型態的無設施，因此補 0
      - 文字資料型態：其中 7 項文字型態，分別依照手冊說明定義映射方式與映射值。
      - Neighborhood 套用 Target Encodinge；Foundation 套用 One-Hot Encoding，其餘套用手動 mapping 轉換。
      - 透過 VIF 檢查，發現拆成 one hot encoding 的 Foundation 地基材質出現設計矩陣奇異的狀況，造成完美共線性。因此刪除數量最小的欄位，以解決線性模型檢定的問題。

- #### 針對隨機森林、線性模型
    - **清洗**：
        - 明確可以定義有順序的文字內容轉成 Ordinal Encoding
        - 指標基數 >=6 的轉成 Target Encoding 
        - 其餘轉成 One Hot Encode 模式
        - 刪除原本透過相關係數找出的兩筆 GrLivArea 之離群值
    - **填補**：
        - 找出特殊條件下資料的填補對應的值或是眾數。
        - Electrical 資料有缺失則填補眾數
        - 數值型態缺填補中位數
        - 文字型態缺失填補文字 NA
        - 文字型態缺失，但本身手冊並無定義 NA 者，補上眾數

## 4. 模型表現 (Model Performance)
- Baseline (OLS_Linear):
    - R-squared : 0.89
    - Train RMSE :
    - Test RMSE : 0.15
    - RMSE Degradation (Train vs Test):0.19
    - 此模型預測結果上傳 Kaggle， OLS 模型與 12項指標的 public Score 為 0.15445，將以此模型為基準線。
- Baseline (Linear Regression):
    - R-squared : 0.91
    - Train RMSE :
    - Test RMSE :0.12
    - RMSE Degradation (Train vs Test): 0.15
- Canditate Model (Random Forest Regression):
    - R-squared : 0.88
    - Train RMSE :
    - Test RMSE : 0.14
    - RMSE Degradation (Train vs Test): 1.90
    - 已先透過 GridSearchCV 找出該批資料中最適切的參數，仍得出過度擬合的結果，並且 Test RMSE : 0.14 略高於 Linear Regression。顯示本次特徵工程下，對於隨機森林模型有較大的限制。如為了降低過度擬合，會導致預測模型失準。
- Canditate Model (XGBoost Regression):
    - R-squared : 0.90
    - Train RMSE :
    - Test RMSE : 0.12
    - RMSE Degradation (Train vs Test): 3.45
    - 已先透過早停法、CV k fold 方式，找出該批資料中最適切的參數，名稱為 CVxgbModel。Test RMSE : 0.12 等於 Linear Regression，而 Train RMSE : Test RMSE : 0.12 ，兩者 RMSE Degradation (Train vs Test): 3.45 比沒使用 CV k fold 時低許多(RMSE Degradation (Train vs Test): 10.95 )，預測結果仍是過度擬合。因   Train RMSE 逼近  ？？？ ，為極端過度擬合之狀況，造成有高度的 RMSE Degradation。在本次特徵工程下，XGBoost 表現不如線性模型。
- Canditate Model (LightGBM Regression):
    - R-squared : 0.91
    - Train RMSE :
    - Test RMSE : 0.12
    - RMSE Degradation (Train vs Test): 0.95
    - 已先透過 GridSearchCV 找出該批資料中最適切的參數，仍得出過度擬合的結果，Test RMSE : 0.12 等於 Linear Regression。在本次特徵工程下，LightGBM 表現不如線性模型。
- Final Model (LassoCV): RMSE 0.12 (**提升 33%**)
    - R-squared : 0.91
    - Train RMSE :
    - Test RMSE: 0.12
    - RMSE Degradation (Train vs Test): 0.14
    - 此模型 Test RMSE: 0.12  等於 Linear Regression 不過 RMSE Degradation (Train vs Test): 0.14 略低於 Linear Regression 故選定期為最終勝出模型。
    - 此模型預測結果上傳 Kaggle， LassoCV 的 public Score 為 0.13276，優於 OLS 與 12 項指標的模型結果。
