import pandas as pd
import re

'''
input: High school data with over 400 columns and school quality data with name, phone number and rate
function: merge these two dataset based on their phone number
output: High school data with quality labeled (some rows are unlabeld due to insufficient data)
'''
def merge_data(schooldata,schoolrank):

    # clean phone numbers so that both dataset has same format
    for row,val in enumerate(schoolrank["Phone"]):
        val = val.strip(" ")
        a = val[0:3]+val[4:7]+val[8:]
        schoolrank.loc[row,"Phone"] = a

    for row,val in enumerate(schooldata["Phone"]):
        a = val[0:3]+val[4:7]+val[8:]
        schooldata.loc[row,"Phone"] = a

    sol = pd.merge(schooldata,schoolrank,on="Phone",how="inner")
    sol = sol.drop(["School"],axis=1)
    sol = sol.drop(["Phone"],axis=1)

    return sol

'''
input: school dataset with NaN values
function: fill all the NaN values with median
output: dataset without NaN values
'''
def fill_empty(x):

    for column in x.columns:
        if sum(pd.isnull(x[column]))>0:
            x.loc[:,column] = x.loc[:,column].fillna(x.loc[:,column].median())
    return x


'''
input: school dataset
function: transform columns into more usable form
ex. column with language courses each school offers transformed into number of language courses that each school offer
output: school dataset with transformed columns
'''

def trans_form(x):

    # 1. create number of language courses offer
    for row in range(len(x["#_Lan_cours"])):
        if pd.isna(x.loc[row,"#_Lan_cours"]) == False:
            a = x.loc[row,"#_Lan_cours"].split(",")
            x.loc[row,"#_Lan_cours"] = len(a)

    # 2. create number of AP courses offer
        if pd.isna(x.loc[row,"#_AP_cours"]) == False:
            a = x.loc[row,"#_AP_cours"].split(",")
            x.loc[row,"#_AP_cours"] = len(a)

    # 3. create number of extracurricular activities offer
        if pd.isna(x.loc[row,"extr_act"]) == False:
            a = x.loc[row,"extr_act"]
            b = re.split(',|/|;',a)
            x.loc[row,"extr_act"] = len(b)  

    # 4. create school district number from dbn

        if pd.isna(x.loc[row,'dbn']) == False:
            a = x.loc[row,'dbn']
            b = re.sub(r'\D.*','',a)
            b = re.sub(r'^0','',b)
            x.loc[row,'school_dist_code'] = float(b)

    # 5. create school duration
    # remove all the AM and PM
    for i in ["start_time","end_time"]:
        for row in range(len(x[i])):
            if pd.isna(x.loc[row,i]) == False:
                removeAPM = x.loc[row,i].rstrip(" ;mMpPaA")
                spl = re.split(':|;',removeAPM)
                if i == "start_time":
                    hr = float(spl[0].split("or")[0].strip(" ;mMpPaA"))
                else:
                    hr = float(spl[0].split("or")[0].strip(" ;mMpPaA"))+12
                m = 0
                if len(spl)>1:
                    m = float(spl[1].split("or")[0].strip(" ;mMpPaA"))
                total_min = hr*60 + m
                x.loc[row,i] = total_min
            
    # create school duration column and drop start and end time
    x["sch_dur"] = x["end_time"] - x["start_time"]
    x = x.drop(["start_time"],axis=1)
    x = x.drop(["end_time"],axis=1)
    return x



