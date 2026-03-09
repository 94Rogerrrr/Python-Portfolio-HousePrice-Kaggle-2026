# Python-Portfolio-HousePrice_Kaggle_20260305
It's a repository for Python data analysis portfolio about Kaggle Data - House Price Prediction. Here are .ipynb, requirement.txt, README.md in this repository.

The project is trying to evaluate the house price is underestimated by building a house price predicting model according to the Kaggle Dataset(https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset)

first problem, there are 80+ columns, how do I be able to check every column has non, mean, or outlayer?
#### 1. 專案標題與一句話總結 (Title & Hook)
- **標題**：House Price Prediction: Advanced Regression Techniques
- **摘要**：利用 XGBoost 與 Lasso 回歸模型預測房價，成功將誤差 (RMSE) 降低至 0.12，並識別出「廚房品質」與「居住面積」為影響房價的最關鍵因素。
#### 2. 商業洞察 (Business Insights) —— _這是你最有價值的地方_
_(這裡放一張漂亮的 SHAP 或特徵重要性圖表)_
- 發現 1：翻修廚房比增建泳池更能顯著提升房價。
- 發現 2：雖然坪數重要，但當屋齡超過 20 年後，坪數對價格的邊際效應會遞減。
#### 3. 資料處理 (Data Methodology) —— _這裡寫剛剛的健檢結果_
- **清洗**：移除了 2 筆極端異常值與邏輯錯誤資料。
- **填補**：針對車庫缺失值，採用「無車庫」邏輯填補而非平均值填補。
#### 4. 模型表現 (Model Performance)
- Baseline (Linear): RMSE 0.18
- Final Model (XGBoost): RMSE 0.12 (**提升 33%**)
