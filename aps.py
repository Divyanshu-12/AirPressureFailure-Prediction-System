import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import random
from sklearn.metrics import f1_score
import joblib
from sklearn.metrics import f1_score
import math
from flask import Flask, jsonify, request
import flask
import time




app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'welcome to aps prediction system'

@app.route('/index')
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    """
    this function takes path of raw test data as input and
    print()
    """
    start = time.time()

    path = request.form.to_dict()
    print(path)
    test_df = pd.read_csv(path["file_name"])
    test_df = test_df.drop(["bn_000","bo_000","bp_000","bq_000","br_000"],axis=1)# drop these columns

    test_df[["ab_000","cr_000"]] = test_df[["ab_000","cr_000"]].fillna(1) # fill nan values with 1

    test_df.loc[test_df["ab_000"] != 1, "ab_000"] = 0# fill everything except 1 with zero
    test_df.loc[test_df["cr_000"] != 1, "cr_000"] = 0

    test_df_70 = test_df[["ab_000","cr_000"]]
    
    knn_imputer = "knn_imputer.pkl"
    new_imputer = joblib.load(knn_imputer)

    median = pd.read_csv("median.csv")
    miss_column_10 = np.load("median_column.npy",allow_pickle=True)
    miss_column_10_70 = np.load("knn_column.npy",allow_pickle=True)
    
    # replace values in those columns which have missing value less than 10% by median value of train data for corresponding feature
    test_df[miss_column_10] = test_df[miss_column_10].fillna(median["0"].median())
    test_df_10 = test_df[miss_column_10]
    
    test_df_10_70 = pd.DataFrame(new_imputer.transform(test_df[miss_column_10_70]))
    test_df_10_70.columns = miss_column_10_70
    test_df_10_70.isnull().values.any()
    
    test_cleaned = pd.concat([test_df_10,test_df_10_70,test_df_70],axis=1)
    
    top_10 = ['bi_000', 'ay_002', 'ay_006', 'cc_000', 'ay_008', 'al_000', 'ag_001','ag_002', 'bj_000', 'ay_005'] 
    top_4 = ["ay_006", "cc_000", "ay_008", "bj_000"]
    top_4_median = [165116.0, 2112040.0, 92906.0, 154640.0]
    
    for i in top_10:
        
        temp1 = i + "_sin"
        temp2 = i + "_log"
    
        test_cleaned[temp2] = test_cleaned[i].apply(lambda x: math.log(x+1))
        test_cleaned[temp1] = test_cleaned[i].apply(lambda x: math.sin(x))
        
    for i in range(4):
    
        temp1 = top_4[i] + "_median"
        test_cleaned[temp1] = test_cleaned[top_4[i]] - top_4_median[i]
        #test_cleaned["class"] = test_df["class"]
        
    my_model = "model.pkl"
    clf = joblib.load(my_model)
    # y_test = test_cleaned["class"]
    # x_test = test_cleaned.drop(["class"],axis=1)
    y_test_pred = clf.predict(test_cleaned)

    # x_test["class"] = y_test
    # x_test["predicted_class"] = y_test_pred
    # x_test.to_csv("C:/Users/mridul/Desktop/predicted.csv",index=False)
    # for i in y_test_pred:
    #     print("prediction  is - ",i)

    end = time.time()
    total = end - start
    if y_test_pred[0] == 0:
        return "Failue has nothing to do with APS, Total time taken is " + str(round(total,2)) + " seconds"
    else:
        return "Failue is due to do APS, Total time taken is " + str(round(total,2)) + " seconds"
    return "prediction is " + str(y_test_pred[0])
    #return "Predicted results are stored in Predicted.csv file in the desktop, Total time taken is " + str(round(total,2)/100) + " seconds for each datapoint"
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)