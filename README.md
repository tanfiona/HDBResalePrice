# HDBResalePrice
[Github Repository Link](https://github.com/pillowtann/HDBResalePrice)<br>
Prediction of HDB resale prices on Kaggle for CS5228

# Task Description
The resale market of HDB flats is big business in Singapore. To find a good prices as either a buyer or a seller, it is important to have good understanding of what affects the market value of a HDB flat. Most people would accept that attributes such as the size and type of flat, its floor, but also its location to nearby amenities (e.g., MRT stations, parks, malls, commercial centers) influence the resale price of the flat. However, it is not obvious which attributes are indeed most important in a quantified sense.<br>
The goal of this project is to predict the resale price of a HDB flat based on its properties (e.g., size, #rooms, type, model, location). It is therefore first and foremost a regression task. Besides to prediction outcome in terms of a dollar value, other useful results include the importance of different attributes, the evaluation and comparison of different regression techniques, an error analysis and discussion about limitations and potential extensions, etc.

# Running the script
Please run the following script in command line with your applicable arguments.
The default arguments is provided as the first value for string values.
Please ensure all datasets are downloaded beforehand (See the Resources bullet).
```
python -W ignore main.py
	--model_name lgb/knn/...
	--tuning false/true
	--folds 5
```
<br>

Expected command line output for a first run where datasets will be generated into an 'out' folder:
```
16/04/2021 12:43:39 - INFO - root -   -- starting process
16/04/2021 12:43:39 - INFO - root -   -- starting train fe
16/04/2021 12:43:39 - INFO - root -   Generating fe data for main...
16/04/2021 12:43:55 - INFO - root -   Generating fe auxiliary data for "commercial"...
16/04/2021 12:44:32 - INFO - root -   Generating fe auxiliary data for "hawker"...
...
```

Once a dataset is created and not removed from the 'out' folder, future runs will read the generated csv file directly instead of recreating to save time. An example output is as follows:
```
07/04/2021 13:41:59 - INFO - root -   -- starting process
07/04/2021 13:41:59 - INFO - root -   -- starting train fe
07/04/2021 13:41:59 - INFO - root -   Opening fe data for all...
07/04/2021 13:42:09 - INFO - root -   -- training
07/04/2021 13:42:11 - INFO - root -   Formatted data has 431732 rows, 90 cols
07/04/2021 13:42:15 - INFO - root -   Conducting fold #0...
07/04/2021 13:53:23 - INFO - root -   TRAIN: n=276308 | VAL: n=69077, rmse=17019.389251371194 | TEST: n=86347, rmse=17017.41992185491
07/04/2021 13:53:33 - INFO - root -   Conducting fold #1...
07/04/2021 14:03:47 - INFO - root -   TRAIN: n=276308 | VAL: n=69077, rmse=16950.078474951322 | TEST: n=86347, rmse=17057.401956262573
07/04/2021 14:03:57 - INFO - root -   Conducting fold #2...
07/04/2021 14:14:13 - INFO - root -   TRAIN: n=276308 | VAL: n=69077, rmse=16927.966304355756 | TEST: n=86347, rmse=16976.927929097714
07/04/2021 14:14:23 - INFO - root -   Conducting fold #3...
07/04/2021 14:24:57 - INFO - root -   TRAIN: n=276308 | VAL: n=69077, rmse=17099.032604843687 | TEST: n=86347, rmse=17107.38152794173
07/04/2021 14:25:07 - INFO - root -   Conducting fold #4...
07/04/2021 14:35:16 - INFO - root -   TRAIN: n=276308 | VAL: n=69077, rmse=16897.011135961995 | TEST: n=86347, rmse=17016.524823460953
07/04/2021 14:35:25 - INFO - root -   OVERALL --> VAL: rmse=16978.84987476144 | TEST: rmse=16790.258614131195
07/04/2021 14:35:39 - INFO - root -   -- starting test fe
07/04/2021 14:35:39 - INFO - root -   Opening fe data for all...
07/04/2021 14:35:41 - INFO - root -   -- predicting
07/04/2021 14:36:54 - INFO - root -   -- complete process
```

# Notebooks
Some visualisations and models (XGBoost) was ran separately in Jupyter Notebooks under the 'notebooks' folder. To run them, you would need to first run the script mentioned above to create a full train and test dataset with exhaustive features. The datasets will be created in an 'out' folder, and named as 'train_df_fe_all.csv' and 'test_df_fe_all.csv' respectively. 

# Resources
[Kaggle Task Link](https://www.kaggle.com/c/cs5228-2020-semester-2-final-project/overview)<br>
Please use the Kaggle link to download and save ALL csv files into the 'data' folder without renaming or moving the folders around. 
These datasets are excluded from this repository due to their large sizes. 
Additional datasets used in the pipeline are uploaded into this repository under the 'data' folder.