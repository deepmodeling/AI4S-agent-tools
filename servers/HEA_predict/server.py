import argparse
import numpy as np
import pandas as pd
import joblib
import os
import mendeleev
import math
from typing import List
from featurebox.data.name_split import NameSplit
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
load_dotenv()
def parse_args():
    """Parse command line arguments for MCP server."""
    parser = argparse.ArgumentParser(description="DPA Calculator MCP Server")
    parser.add_argument('--port', type=int, default=50001, help='Server port (default: 50001)')
    parser.add_argument('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    try:
        args = parser.parse_args()
    except SystemExit:
        class Args:
            port = 50001
            host = '0.0.0.0'
            log_level = 'INFO'
        args = Args()
    return args
args = parse_args()
mcp = FastMCP("example",host=args.host,port=args.port)

@mcp.tool()
def HEA_predictor(name:List[str])->List[str]:
    '''
    Args:
        name: chemical formula of a type of high entropy alloy, for example: AlCoCr0.5FeNi
    return:List[str]
        res: res[0], predicted type of the given HEA(Solid-Solution,Intermetallic Compounds, or SS+IM)
             res[1], predicted crystal structure of the given HEA
    process:
    1.Split the chemical formula and convert it to a dataframe where Columns are the name of elements included,adn values are the corresponding molar ratio, and then filter the elements not included
    2.Calculate certain parameters like Hmix,Smix,etc and add to the dataframe
    3.Calculate if certain rules are satisfied, add the results to the dataframe
    4.Use the dataframe and a pretrained ML model to predict if the formula can form a solid-solution system, if so, predict its crystal structure
    '''
    HMIX_DATA_PATH =       os.getenv("HMIX_DATA_PATH", "data/binary_Hmix.csv")
    SCALER_TYPE_PATH =     os.getenv("SCALER_TYPE_PATH", "models/feats_scaler_type.pkl")
    SCALER_STRUC_PATH =    os.getenv("SCALER_STRUC_PATH","models/feats_scaler_struc.pkl")
    TYPE_MODEL_PATH =      os.getenv("TYPE_MODEL_PATH", "models/Type_predict_model.pkl")
    STRUCTURE_MODEL_PATH = os.getenv("STRUCTURE_MODEL_PATH", "models/Structure_predict_model.pkl")
    #Name_split--------------------------------------------------------------
    folds_file = 'tmp_folds.csv'
    expands_file = 'tmp_expands.csv'
    splitter = NameSplit() 
    splitter.transform(name,folds_name=folds_file,expands_name=expands_file)
    df = pd.read_csv(expands_file,index_col=0)
    first_cell = df.iloc[0,0]
    for i in range(0,len(df)):
        row_sum = df.iloc[i,1:].sum()
        if row_sum ==0:
            continue
        scaling_factor = 1/row_sum
        for j in range(1, len(df.columns)):
            df.iloc[i,j] = df.iloc[i,j] * scaling_factor
        df.iloc[0,0] = first_cell
    os.remove(folds_file)
    os.remove(expands_file)   
    #filter elements 
    VEC_dict = {'Be':2,'B':3,'Al':3,'Si':4,'Ti':4,'V':5,'Cr':6,'Mn':7,'Fe':0,'Co':0,'Ni':0,'Cu':1,'Zn':2,'Ge':4,'Y':3,'Zr':4,'Nb':5,'Mo':6,'Ce':4,'Hf':4,'Ta':5,'W':6,'Au':1,'Ru':0,'Rh':0,'Pd':0}
    df = df[[col for col in df if col in VEC_dict]]
    elements = [col for col in df.columns]
    #calculate VEC
    df['VEC'] = 0.0
    for index, row in df.iterrows():
        total_vec = 0.0
        for element in elements: 
            c_i = row[element]  
            vec_i = VEC_dict[element] 
            total_vec += c_i * vec_i
        df.at[index, "VEC"] = total_vec
    #calculate delta
    df['delta(%)'] = 0.0
    radius_dict={}
    for element in elements:
        if element != 'Ce':
           radius_dict[element] = mendeleev.element(element).metallic_radius_c12/100
        else:
           radius_dict[element] = 1.65
    Hmix_data = pd.read_csv(HMIX_DATA_PATH, index_col=0)

    for index, row in df.iterrows():
        average_radius = sum(row[element] * radius_dict[element] for element in elements)
        delta_square = sum(row[element] * ((1 - (radius_dict[element] / average_radius)) ** 2) for element in elements)
        delta = math.sqrt(delta_square)

    #CALCULATE AND FILLIN Hmix, Smix, Lambda------------------------------------------------------------------
        total_mixing_enthalpy = 0.0
        for i in range(len(elements)):
            for j in range(i + 1, len(elements)):
                element1 = elements[i]
                element2 = elements[j]
                c_1 = float(row[element1])
                c_2 = float(row[element2])
                binaryHmix = 0.0
                if (element1 == element2):
                    raise ValueError ('same element')
                elif (pd.notna(Hmix_data.loc[element1,element2])):
                    binaryHmix = Hmix_data.loc[element1,element2]
                else:
                    binaryHmix = Hmix_data.loc[element2,element1]
                total_mixing_enthalpy += 4*c_1*c_2*float(binaryHmix)

        total_mixing_entropy = 0.0
        for element in elements:
            if (row[element]> 0):
                total_mixing_entropy += -8.314*row[element]*math.log(row[element])  
            else:
                continue       

        df.at[index, "delta(%)"] = delta * 100
        df.at[index, "Hmix(kJ/mol)"] = total_mixing_enthalpy
        df.at[index, "Smix(J/Kmol)"] = total_mixing_entropy
        df.at[index, "Lambda"] = total_mixing_entropy / ((delta * 100) ** 2)      
   
    #RULES CHECK
    #Hmix-δ判据
    conditions_Hmix_delta = [
        (df["delta(%)"] < 6.66) & (df["Hmix(kJ/mol)"] < 6.92),       # Class SS
        (df["delta(%)"] > 5.64) & (df["Hmix(kJ/mol)"] < -20.0),      # Class Non-Crystal
        ~((df["delta(%)"] < 6.66) & (df["Hmix(kJ/mol)"] < 6.92)) & ~((df["delta(%)"] >5.64) & (df["Hmix(kJ/mol)"] <-20.0))  # Class compounds
    ]
    choices_Hmix_delta = [0, 1, 2]
    #λ判据
    conditions_Lambda = [
        (df["Lambda"] > 0.245),        # Class disorder SS
        ~ (df["Lambda"] > 0.245)       # Class Non-SS
    ]
    choices_Lambda = [0, 1] 
    #VEC判据
    conditions_VEC = [
        (df["VEC"] >= 2.909),                     #Class FCC
        (df["VEC"] < 0.5),                        #Class BCC
        (df["VEC"] >= 0.5) & (df["VEC"] < 2.909), #Class FCC+BCC
    ]
    choices_VEC = [0, 1, 2] 
    # δ判据
    conditions_delta = [
        (df["delta(%)"] > 7.44),                        # Class compounds
        (df["delta(%)"] <= 7.44) & (df["delta(%)"]>= 3) # Class BCC
    ]
    choices_delta = [0, 1] 
    df["Hmix_delta"] = np.select(conditions_Hmix_delta, choices_Hmix_delta, default=0)
    df["Lambda_rule"] = np.select(conditions_Lambda, choices_Lambda, default=0)
    df["VEC_rule"] = np.select(conditions_VEC, choices_VEC, default=0)
    df["delta_rule"] = np.select(conditions_delta, choices_delta, default=0)

    scaler_type = joblib.load(SCALER_TYPE_PATH)
    X_pred = df.iloc[:,0:36]
    X_pred_scaled = scaler_type.transform(X_pred)
    model_type = joblib.load(TYPE_MODEL_PATH)
    model_struc = joblib.load(STRUCTURE_MODEL_PATH)
  
    res = []
    y_pred_1 = model_type.predict(X_pred_scaled)
    if y_pred_1[0] == 1:
        res.append ('solid solution')
    elif y_pred_1[0] == 2:
        res.append ('Solid Solution + Intermetallic compounds')
    elif y_pred_1[0] == 0:
        res.append ('Intermetallic compounds')

    if (res[0] == 'solid solution'): 
       scaler_struc = joblib.load(SCALER_STRUC_PATH)
       X_pred_scaled = scaler_struc.transform(X_pred)
       y_pred_2 = model_struc.predict(X_pred_scaled)
       if y_pred_2[0] == 1:
          res.append('BCC')
       elif y_pred_2[0] == 5:
          res.append('FCC')
       elif y_pred_2[0] == 3:
          res.append('FCC+BCC')
       else:
          res.append('complicated structure')
    return(res)
    
if __name__ == "__main__":
    # Get transport type from environment variable, default to SSE
    transport_type = os.getenv('MCP_TRANSPORT', 'sse')
    mcp.run(transport=transport_type)