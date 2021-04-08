# HDBResalePrice
Prediction of HDB resale prices on Kaggle for CS5228

# Task Description
The resale market of HDB flats is big business in Singapore. To find a good prices as either a buyer or a seller, it is important to have good understanding of what affects the market value of a HDB flat. Most people would accept that attributes such as the size and type of flat, its floor, but also its location to nearby amenities (e.g., MRT stations, parks, malls, commercial centers) influence the resale price of the flat. However, it is not obvious which attributes are indeed most important in a quantified sense.<br>
The goal of this project is to predict the resale price of a HDB flat based on its properties (e.g., size, #rooms, type, model, location). It is therefore first and foremost a regression task. Besides to prediction outcome in terms of a dollar value, other useful results include the importance of different attributes, the evaluation and comparison of different regression techniques, an error analysis and discussion about limitations and potential extensions, etc.

# Running the script
Please run the following script in command line with your applicable arguments.
The default arguments is provided as the first value for string values.
```
sudo /home/fiona/.conda/envs/env_01/bin/python3 -W ignore main.py
	--model_name lgb/knn
	--tuning false/true
	--folds 5
```

# Resources
[Kaggle Task Link](https://www.kaggle.com/c/cs5228-2020-semester-2-final-project/overview)<br>
Please use the Kaggle link to download and save ALL csv files into the 'data' folder without renaming or moving the folders around. 
These datasets are excluded from this repository due to their large sizes. 
Additional datasets used in the pipeline are uploaded into this repository under the 'data' folder.