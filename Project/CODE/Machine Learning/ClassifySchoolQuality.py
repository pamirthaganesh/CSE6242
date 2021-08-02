import gc
import re
import os
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import interp
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from preprocess import fill_empty, trans_form, merge_data
from ML import select_features, seperate_modelandtest, hyperparameter, estimate, Kfold
pd.options.mode.chained_assignment = None  

'''
This code does following tasks:
1. Read the New York City High School data and their quality data based on Greatschools.org and merge them
2. Clean dataset and transform the predictor (more descriptions are illustrated in the preprocess code)
3. From the original dataset Extract out labeled high school data which has a quality not NaN
4. Train the prediction model with Random Forest (more descriptions are illustrated in the ML code)
5. Predict rest of unlabeled High School data with the model and re-attach to the labeled data
6. Save the dataset as CSV file to be used for visulization part of this project

'''

def main():

#PREPROCESS
    # 1. Read the dataset

    # url1 = 'https://github.gatech.edu/raw/asun45/cse6242_final_project/master/data/2020highschool.csv?token=AAAB5YAWIM6PPJZ7FYMV4MC6VA4YY'
    # url2 = 'https://github.gatech.edu/raw/asun45/cse6242_final_project/master/data/SchoolQualityResponse.csv?token=AAAB5YE7ZN3T4XQMCLQPSAS6VA6MQ'
    Path1 = os.path.abspath("2020highschool.csv")
    Path2 = os.path.abspath("SchoolQualityResponse.csv")
    schooldata = pd.read_csv(Path1)
    schoolrank = pd.read_csv(Path2)

  
    # Columns to include
    header = ["dbn","school_name","phone_number","language_classes","advancedplacement_courses","start_time","end_time",
            "extracurricular_activities","graduation_rate","attendance_rate","pct_stu_enough_variety","college_career_rate",
            "pct_stu_safe","borough_code","zip","shared_space","#of_bus","start_level","final_level","#_of_levels",
            "total_students","#of_ECA","#of_boys_sports","#of_girls_sports","#of_coed_sports","girls_school","boys_school",
            "co-ed_school","pbat","ptech","restrict_geoeligibility","school_accessibility_description"]

    schooldata = schooldata[header]

    # Rename some of the columns
    schooldata = schooldata.rename(columns={"phone_number":"Phone","zip":"zipcode","language_classes":"#_Lan_cours",
    "advancedplacement_courses":"#_AP_cours","graduation_rate": "grad_rate","pct_stu_enough_variety": "satisfaction_rate",
    "college_career_rate": "career_rate", "pct_stu_safe": "stu_sft","extracurricular_activities" :'extr_act'})

    # 2. Merge the two dataset
    result = merge_data(schooldata,schoolrank)

    # 3. Clean dataset and transform columns into more usable columns
    result = trans_form(result)

    # 4. Find all NaN values and fill them all with median value except the response variable
    result.loc[:,result.columns != 'Rate'] = fill_empty(result.loc[:,result.columns != 'Rate'])

    #End of data preparation--------------------------------------------------------------------

# Train and test model and predit unlabeled school data
    # 1. Extract data for train and data for prediction later
    X,y,x_predict,top_portion,bot_portion = seperate_modelandtest(result)

    # 2. split train and test data
    random_state = 6242
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state = random_state)

    # 3. Mutual information feature selection
    newfeature = select_features(x_train,y_train)
    x_train = x_train[newfeature]
    x_test = x_test[newfeature]
    x_predict = x_predict[newfeature]

    # 4. Hyperparamter search
    max_depth, max_features,n_estimators = hyperparameter(x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test, random_state=random_state,n_estimator = [200, 400],max_dep = [10, 110], max_features = ['auto', 'sqrt'])

    # 5. Performance measurement using confusion box, precision, recall through K-fold CV

    # 6. Train the model and estimate the qualities of 61 unassigned high schools
    qualities, accuracy = estimate(x_train,y_train,x_test,y_test,x_predict,random_state,n_estimators,max_depth,max_features)
    # print("Model's testing accuracy: {}".format(accuracy))

    # 7. Performance measurement with confusion matrix
    train_accuracy, test_accuracy, ConfusionBox, macroprecision, macrorecall = Kfold(X,y,random_state,max_depth,n_estimators)
    print("Model's train accuracy: {}".format(train_accuracy))
    print("Model's testing accuracy: {}".format(test_accuracy))
    print("Model's macro precision: {}".format(macroprecision))
    print("Model's macro recall: {}".format(macrorecall))
    print(ConfusionBox)

    # 8. Aggregate the pieces and save it as file to be used for visualization
    bot_portion['Rate'] = qualities
    NY_Highschool = top_portion.append(bot_portion)
    head = ['school_dist_code','dbn','school_name','zipcode','Rate']
    school_quality_rating = NY_Highschool[head]
    print('New York High School Quality Data----------------------------------------------------------')
    print(school_quality_rating.head())
    final = school_quality_rating.copy()
    final.to_csv("school_quality_rating.csv", index=False)
    dfff= pd.read_csv('school_quality_rating.csv')
    dfff.head()
if __name__ == "__main__":
    main()