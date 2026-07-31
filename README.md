# Python-Portfolio-HousePrice_Kaggle_20260305
It's a repository for Python data analysis portfolio about Kaggle Data - House Price Prediction. Here are .ipynb, requirement.txt, README.md in this repository.

The project aims to identify undervalued properties by building a predictive pricing model based on the Kaggle Dataset(https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset)

## 1. 專案摘要 (Title & Hook)
- **標題**：House Price Prediction: Evaluate whether the house price is underestimated.
- **摘要**：本專案橫跨傳統線性迴歸家族 (OLS, Linear and LassoCV) 及樹模型家族 (Random Forest, XGBoost 及 LightGBM) 預測房價，成功將誤差 Test RMSE(Log) 降低至 0.13081，並且有 Kaggle Public Score 0.13746 的成績。不僅找出有潛力的投資物件，也透過特徵係數得出「居住面積」與「整體品質」為影響房價的最關鍵因素。
## 2. 商業洞察 (Business Insights) 
### 初步洞察：
模型訓練前，資料處理過程中的發現：
  #### 相關係數：
  - 查看負相關的數值，發現最低相關為 KitchenAbvGr -0.134445 然而其欄位在說明手冊上卻沒有定義。依據欄位名稱應屬於地面上廚房的相關內容。

  **相關係數熱力圖解讀**

  _(這裡放一張漂亮的 SHAP 或特徵重要性圖表)_
  - 發現 1: 文字資料經過較精確的 Encode 邏輯，且對房價做常態化後，前三項如下，顯示均與房價有高度相關。
    - 最高者為 OverallQual (0.81)
    - 次高者為 GrLivArea (0.73)
    - 第三名為 Neighborhood (0.73)


### 模型結果商業應用：
本專案執行了數種模型，以 OLS 模型為基準，其他模型與之相較，選取房價誤差小以及過度擬合程度低者。<br>
多種樹模型之預測誤差 Test RMSE(Log)、RMSE Degradation (Train(<span>$</span>) vs Test(<span>$</span>))(%) 與 Test RMSE($) 均明顯大於線性模型，故推論在本次特徵工程下，線性模型表現較佳，可以兼顧最小的均方誤差與過度擬合問題，又 LassoCV 模型會排除共線性或是無相關的特徵項，故以 LassoCV 模型作全域特徵解釋，找出被低估 15% 之標的。

##### 安全條件
初步篩選時發現價差最高的房屋，落在平均價格最低的社區，其他重要特徵表現亦不亮眼，推測以下：
- 可能有嫌惡設施、事故等等，若希望投資需進一步勘查。
- 若真實房價區間已經歸在極低價的層級，而 LassoCV 模型的全局特徵訓練，將無法如實預測極低價房價。
- 透過殘差分析發現，LassoCV 模型預測在房價低於 USD$ 107,000 時， MAPE 為 16%，其上則有 7 ~8% 的誤差，故設定 USD$ 107,000 為門檻。
  <br>

  
故設定安全邊際如下：
*  真實房價為 USD$ 107,000 以上
*  OverallCond 整體屋況達 3 分


##### 深度案例分析：
目標物件: HouseID: 589
  - 市場實際售價 (Actual Price): USD$ 143,000
  - 模型預估價值 (Predicted Value): USD$ 245,523
  - 潛在毛利空間 (Potential Margin): USD$ 102,523 (+ 71%) 

模型高估原因分析 (Key Drivers)：
_SHAP_linear圖_
- 從圖中可知此棟房屋的整體屋況 OverallCond 提升房價的幅度最大，顯然房屋狀況為 Very Good 為其亮眼特色。
- 其次，所使用的交易模式為 Partial，是分類中平均房價最高的等級，亦為預測房屋的加分項目。
- 最後，其地下室完善面積為 1,324 平方英尺，也相對高幅度地提升房價預測。

綜合以上，該房屋所處的社區為 ClearCr 屬於中上的上區，雖非昂貴社區，不過房價仍有成長空間，預測有潛力高於當前的房價。

### 模型結果特徵分析：
LassoCV 模型適合做相關性解讀，故針對此模型保留之特徵進行分析。

(放 LassoCV feature best 10 and least 10 圖)
- 挑選最高與最低的 10 項特徵係數繪製水平直方圖。最高 10 項的特徵的影響力大於最低 10 項的負面影響力，顯示本次特徵工程在 LassoCV 的學習下，雖有扣分之特徵但其負面影響不如正面的特徵影響大。
- 特徵係數最高者為 GrLivArea 地面上居住面積，次高者為 OverallQual 整體品質，顯示這兩項指標會高度與房價正相關。
    #### 特徵係數之商業價值轉換

    ##### 數值資料
  |Top Features|實務變動單位 (約 1 Std.Dev)| 房價溢價佔比 premium(%) |相對 20 萬美元房價提升   |
  |----|----|-----|-----|
  |GrLivArea | +500 Sq.Ft 平方英尺|+13.88%|+27,768 美元|     
  |OverallQual| +1 Level| + 5.77% |+11,542 美元|
  |OverallCond| +1 Level| + 3.54%  |+7,095 美元|
  |Functional| +1 Level| + 3.30%|+6,605 美元|
  |TotalBsmtSF| +410 Sq.Ft 平方英尺| + 3.07%  |+6,143 美元|
  |BsmtFullBath| +1 room| +2.71% |+5,438 美元|
  |GarageCars| +1 car| +2.60% |+5,213 美元|       

    ##### 文字資料
  |Top Features|最高表現類別 (Top)|最低表現類別 (Bottom) |兩者溢價差距 premium(%)  |相對 20 萬美元房價提升   |
  |----|----|-----|-----|-----|
  |Neighborhood|NridgHt|BrkSide |+2.69% |+5,381 美元 |
  |SaleCondition|Partial |Abnorml|+1.50%| +3,003 美元 |


    #### 考量 ROI 之商業建議
-  相比地上居住空間與地下室指標之溢價百分比，前者較高，根據數據建議若翻修有預算考量，可優先改善地面上居住空間與整體品質之完善程度。
-  若房子的地點與地上總面積已固定，建議完善整體屋況與設施功能，以有效提升房價。
-  若手上的房屋已經大致完成地面上的設施功能，可針對地下室部分完善修建，以提高房價。


## 3. 資料處理 (Data Methodology) 
摘要較詳細的資料處理細節，說明本專案透過何種方式釐清資料，並如何判斷非數值類型資料之轉碼依據。根據模型特性分成兩種資料處理流程：「針對 OLS 線性模型」與「針對 `scikit-learn` 的 Linear, LassoCV 以及樹模型」，並於專案初期先完成資料切割，且僅根據帶有 "train" 等的關鍵字之資料集作為相關係數、特殊條件等判斷來源。
- #### 針對 OLS 線性模型
  - **離群值篩選解讀**：
  - 箱型圖解讀
    - 從 SalePrice、GrLivArea、GarageArea、TotalBsmtSF、1stFlrSF 等圖片與界外值的位置，可判斷這些欄位之資料並非常態分佈，明顯有右偏分布的傾向。
    - 最後將依據 Skewness 數值結果判斷右偏分佈之指標

  - 散佈圖解讀
    - 從 GrLivArea、GarageArea、TotalBsmtSF、1stFlrSF 等圖片，離散與密集程度可知指標具有不合理的離群值。

  - 離群值    
    - 綜合散佈圖資訊與查詢結果，有明顯遠離聚集或是遠離趨勢的點，發現 GrLivArea 的兩筆離群值，也正好是其他欄位的離群值項目。故優先刪減這兩筆資料。
    - GrLivArea 的相關係數高，若有極端值易影響模型預測結果。
    - 其他項目之影響不如 GrLivArea 來得高，故剩下資料予以保留。

  - **清洗**：
      - 若對所有特徵的離群值都進行整筆資料刪除，恐導致資料枯竭，故針對高相關係數之特徵篩除離群值，以免其極端值嚴重影響模型學習
      - 針對 GrLivArea 特徵，移除了 2 筆異常值。
      - 次要相關係數之離群值，予以保留。考量如下：
          - 保留該觀測值，可使模型能學習到該筆資料於其他正常特徵上的資訊
          - 最高的離群值亦隨 GrLivArea 的資料一起刪除
          - 其相關係數影響不如 GrLivArea 高，故衡量資料的模型學習效果與極端值的影響後，決定予以保留。
      - 檢查資料分佈的偏移狀態，針對 Skewness 分數 > 0.5 之偏態指標取對數以矯正偏態。
      - 將文字資料先做 Target Encoding 轉換成數值類型，計算 train 資料之相關係數，選擇其中相關係數高的指標 (Correlation > 0.5)：文字類型共 6 項，數值類型共 9 項。將針對彼此欄位定義制定相對的資料整理邏輯。
      - 15 項指標單位不同以至於數據的尺度差異大，故使用 StandardScaler 降低尺度差異造成的影響。
      - 首次執行 OLS 模型評估指標顯示有較明顯的異質變異數問題，故手動刪除訓練集中，殘差分佈超過三個標準差之資料，共 12 筆。
  - **填補**：
      - 釐清高度相關 15 項指標，其缺失值均為文字型態的無設施，因此補 0
      - 文字資料型態：其中 6 項文字型態，分別依照手冊說明定義映射方式與映射值。
      - Neighborhood 套用 Target Encoding，其餘套用手動 mapping 轉換。


- #### 針對 `scikit-learn` 的 Linear, LassoCV 及樹模型
    - **清洗**：
        - 明確可以定義有順序的文字內容轉成 Ordinal Encoding
        - 指標基數 >=6 者採用 Target Encoding 
        - 其餘轉成 One Hot Encoding 
        - 刪除原本透過相關係數找出的兩筆 GrLivArea 之離群值
    - **填補**：
        - 找出特殊條件下資料的填補對應的值或是眾數。
        - Electrical 指標，資料有缺失則填補眾數
        - 數值型態缺填補中位數
        - 文字型態缺失填補文字 NA
        - 文字型態缺失，但本身手冊並無定義 NA 者，補上眾數

## 4. 模型表現 (Model Performance)
羅列專案中執行模型學習成果，提供衡量標準與綜合考量。 <br>
衡量標準之公式：
*  RMSE Degradation (Train(<span>$</span>) vs Test(<span>$</span>))(%) = (Test RMSE(<span>$</span>) - Train RMSE(<span>$</span>)) / Train RMSE(<span>$</span>)，用以衡量模型對未見資料的衰退程度
*  RMSE Gap = Test RMSE(Log) - Train RMSE(Log)

**各模型衡量指標** 

|Regressions|R-squared|Train RMSE($)|Test RMSE($)|RMSE Degradation (Train(<span>$</span>) vs Test(<span>$</span>))(%)|Train RMSE(Log)|Test RMSE(Log)|RMSE Gap|Test MAE($)|Test WMAPE($)|verdict|
|---|----|---|---|---|----|---|---|---|---|---|
|OLS_11_Linear| 0.86|25.7k|30.5k |18.64%|0.13047|0.15935|0.02888| 19.0k|10.632% | ❌ 基準點  |
|Linear Regression|0.91|19.0k|23.0k|20.67%|0.10523|0.13015|0.02493|15.1k|8.464%| 🟢 略勝  |
|Random Forest Regression|0.88|11.1k|30.0k|169.57%|0.04924|0.14909|0.09984|16.7k|9.359%|❌ 過擬合|
|XGBoost Regression|0.90|2.3k|26.1k|1035.64%|0.01277|0.13735|0.12457|15.7k| 8.812%|❌ 過擬合 |
|LightGBM Regression|0.90|13.4k|27.9k |108.06%|0.05892|0.13846|0.07954|15.9k|8.946%|❌ 過擬合|
|LassoCV|0.91|19.6k|22.6k|15.30%|0.10834|0.13081|0.02247|14.9k|8.381%|🏆 **勝出**|

- 六種模型的平均絕對誤差百分比 (Test WMAPE (<span>$</span>)) 約在 8% ~ 11% 之間，達到商業上對於模型準確及格之要求。
- Linear Regression 模型的 R-squared: 0.91，Test RMSE(<span>$</span>): 23.0k 與 Test RMSE(Log): 0.13015 三項指標比 OLS_11_Linear 表現亮眼。唯獨在美金誤差的指標上 RMSE Degradation (Train(<span>$</span>) vs Test(<span>$</span>))(%): 20.67% 衰退率略高於 OLS_11_Linear。綜合以上，Linear Regression 將暫時成為目前最佳表現的模型。
- Random Forest Regression 已先透過 GridSearchCV 找出該批資料中最適切的參數，仍得出過度擬合的結果 (RMSE Degradation (Train(<span>$</span>) vs Test(<span>$</span>))(%): 169.57%)，並且 Test RMSE(Log): 0.14909 略高於 Linear Regression。在本次特徵工程下，需繼續調整參數才能取得更好的結果，當前參數下 Random Forest 表現不如線性模型。
- XGBoost Regression 已先透過早停法、CV k fold 方式，找出該批資料中最適切的參數，名稱為 CVxgbModel。Test RMSE(Log): 0.13735 略高於 Linear Regression，而 XGBoost 的 Train RMSE(<span>$</span>) 僅約 2.3k 造成 RMSE Degradation (Train(<span>$</span>) vs Test(<span>$</span>))(%): 1035.64% 為極端過度擬合之狀況。在本次特徵工程下，需繼續調整參數才能取得更好的結果，當前參數下 XGBoost 表現不如線性模型。
- LightGBM Regression 已先透過 GridSearchCV 找出該批資料中最適切的參數，發現 Test RMSE(Log): 0.13846 略高於 Linear Regression，又 Train RMSE(<span>$</span>): 13.4k 造成 RMSE Degradation (Train(<span>$</span>) vs Test(<span>$</span>))(%): 108.06% 高於 Linear Regression 的對應數值，為較明顯的過度擬合的結果。在本次特徵工程下，需繼續調整參數才能取得更好的結果，當前參數下 LightGBM 表現不如線性模型。
- 應對樹模型進行調整參數，才能進一步改善樹模型表現，然考量專案時間成本與效益，採用基礎模型已得出較優結果，故在此專案中選擇線性模型。
- 專案目的為準確預測房價，樹模型的 Test MAE(<span>$</span>)、Test RMSE(<span>$</span>) 以及 RMSE Degradation (Train(<span>$</span>) vs Test(<span>$</span>))(%) 三項指標的數值均大於 `scikit-learn` 線性模型 (Linear Regression 與 LassoCV) 的指標，因此選擇預測誤差較小、相對穩定，且 RMSE Degradation (Train(<span>$</span>) vs Test(<span>$</span>))(%) 退化率較低的線性模型。
- 雖然 LassoCV Test RMSE(Log): 0.13081 略高於 Linear Regression，不過 Test RMSE(<span>$</span>): 22.6k 以及 RMSE Degradation (Train(<span>$</span>) vs Test(<span>$</span>))(%): 15.30% 均比 Linear Regression 表現佳，又 LassoCV 排除共線性與篩除無相關特徵之特性，故選定 LassoCV 為最終勝出模型。

### 上傳 Kaggle 之比較

|     |Public Score| 
|------|----------|
|OLS_11_Linear|0.15940|
|LassoCV|0.13746|

顯示的確達到模型改善之效果。
