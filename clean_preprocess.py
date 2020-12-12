import glob
import pandas as pd
import numpy as np
import sys
import os


def get_labels(idx, label=1):
    """ Gets labels by filling in until a min in reached.
    """
    min = df['Volume'][idx]
    while 1:
        if label==1:
            if idx >= 1:
                idx -= 1
            else:
                return
        elif label==3:
            idx += 1
        curr_val = df['Volume'][idx]
        if curr_val > min:
            return
        else:
            df.at[idx, 'Label'] = label
            min = curr_val

if __name__ == "__main__":
    ID = 0

    if not os.path.exists('./patients_data'):
    	os.makedirs('./patients_data')

    for folder in glob.glob("./ABCPatientData_add/*"):
        ID += 1
        file_num = 0
        for fname in glob.glob(folder+"/*.dat"):
            f = open(fname)
            lines=f.readlines()
            file_num += 1
            df = pd.read_csv(fname,
                        sep=";",
                        skiprows=26,
                        usecols=[0,1],
                        names=['Time','Volume'])
            df.attrs["threshold"] = lines[2].split(":")[1]
            df['Label'] = (df['Volume']>float(df.attrs["threshold"])).astype(int)*2
            df["diff"] = df["Volume"].astype(float).shift(-1, fill_value=0).sub(df["Volume"].astype(float))
            df["from_thresh"] = df["Volume"].astype(float).sub(float(df.attrs["threshold"]))
            df["ID"] = ID
            df["file_num"] = file_num
            
            # get edges

            edges = df['Label'].shift(-1, fill_value=0)-df['Label']
            idx1 = np.where(edges>0)[0]
            idx3 = np.where(edges<0)[0][:-2]
            for left_edge_idx in idx1:
                get_labels(left_edge_idx, label=1)
            for right_edge_idx in idx3:
                get_labels(right_edge_idx, label=3)
            
		
            if not os.path.exists('./patients_data/patient_'+str(ID)):
                os.makedirs('./patients_data/patient_'+str(ID))
	
            #df.to_pickle("./test/"+fname.split('_')[-1].split('.')[0]+".pkl")
            if file_num == 1:
                train_X = df[["Time", "Volume", "from_thresh", "diff","ID","file_num"]]
                train_y = df["Label"]
                train_X.to_pickle("./patients_data/patient_"+str(ID)+"/train_X.pkl")
                train_y.to_pickle("./patients_data/patient_"+str(ID)+"/train_y.pkl")
            else:
                test_X = df[["Time", "Volume", "from_thresh", "diff","ID","file_num"]]
                test_y = df["Label"]
                test_X.to_pickle("./patients_data/patient_"+str(ID)+"/test_X"+str(file_num-1)+".pkl")
                test_y.to_pickle("./patients_data/patient_"+str(ID)+"/test_y"+str(file_num-1)+".pkl")

            
            size_before = df.shape[0]
            # print("size_before",size_before)

            df["state_change"] = df['Label'].astype(float).shift(-1, fill_value=0).sub(df['Label'].astype(float))

            df_00 = df[(df["state_change"]==0) & (df["Label"]==0)]
            
            df_00["time_diff"] = df_00['Time'].astype(float).shift(-1, fill_value=0).sub(df_00['Time'].astype(float))
            
            
            df_00_del = df_00[(df_00["time_diff"]> 1) & (df_00["Time"]>30) ]
            
            cursor = 30
            for index, row in df_00_del.iterrows():
                #print(row['Time'], round(row['time_diff'],2))
                
                df_cut = df[(df['Time']>cursor) &  (df['Time']<=(row['Time']-6))]
                cursor = row['Time']+round(row['time_diff'],2)
                # print(df_cut[df_cut["Label"]==1])
                df.drop(df_cut.index, inplace = True)
            df.reset_index(inplace = True, drop = True)
            df.drop(columns=["state_change"],inplace = True)
            size_after = df.shape[0]
            # print("size_after",size_after)
            # print("Downsampling numbers:",size_before - size_after)
            # print("Downsampling ratio:",(size_before - size_after)/(size_before+0.001))

            if not os.path.exists('./downsampled_data/patient_'+str(ID)):
                os.makedirs('./downsampled_data/patient_'+str(ID))
            
            
            if file_num == 1:
                train_X = df[["Time", "Volume", "from_thresh", "diff","ID","file_num"]]
                train_y = df["Label"]
                train_X.to_pickle("./downsampled_data/patient_"+str(ID)+"/train_X.pkl")
                train_y.to_pickle("./downsampled_data/patient_"+str(ID)+"/train_y.pkl")
            else:
                test_X = df[["Time", "Volume", "from_thresh", "diff","ID","file_num"]]
                test_y = df["Label"]
                test_X.to_pickle("./downsampled_data/patient_"+str(ID)+"/test_X"+str(file_num-1)+".pkl")
                test_y.to_pickle("./downsampled_data/patient_"+str(ID)+"/test_y"+str(file_num-1)+".pkl")
            

            
        
