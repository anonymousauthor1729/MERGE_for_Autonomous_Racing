import pickle
import numpy as np


#########################################################################################################
#########################################################################################################


def traj_data_dict_get():
    traj_data_dict = {
        "name": "",
        "states": [],
        "t_steps": [],
        "progress": [],
        "deviations": [],
    }
    return traj_data_dict


#########################################################################################################
#########################################################################################################


def save_traj_data(TDList:list, filepath:str):
    with open(filepath, 'wb') as f:
        pickle.dump(TDList, f)
    print(f"[INFO]: Trajectory Data saved succesfully -> {filepath}")


def load_traj_data(filepath:str):
    with open(filepath, 'rb') as f:
        TDList = pickle.load(f)
    print(f"[INFO]: Trajectory Data loaded succesfully -> {filepath}")
    return TDList


#########################################################################################################
#########################################################################################################


def extract_valid_trajectories(TDRaw):
    if isinstance(TDRaw, dict):
        splitting_indices = np.argwhere(
            np.abs(np.diff(TDRaw["progress"])) > 0.75
        ).squeeze()
        
        splitting_indices = np.clip(
            splitting_indices + 1, 0, len(TDRaw["t_steps"])
        )
        
        TDList = []
        num_splits = len(splitting_indices)

        for i in range(num_splits + 1):
            start = 0 if i == 0 else splitting_indices[i-1]
            stop = -1 if i == num_splits else splitting_indices[i]
            TD = traj_data_dict_get()

            TD["t_steps"] = TDRaw["t_steps"][start:stop]
            TD["progress"] = TDRaw["progress"][start:stop]
            TD["states"] = TDRaw["states"][start:stop]
            TD["deviations"] = TDRaw["deviations"][start:stop]
            TD["t_steps"] -= TD["t_steps"][0]
            TDList.append(TD)
    
    else: TDList = TDRaw
        
    valid_demos = np.where([
        np.isclose(TD["progress"][0], 0.0, atol=1e-2) and 
        np.isclose(TD["progress"][-1], 1.0, atol=1e-2) 
        for TD in TDList
    ])[0]
    
    TDList_valid = [
        TDList[i] for i in valid_demos
    ]
    return TDList_valid

    
#########################################################################################################
#########################################################################################################


def print_details(TDList:list):
    print(f"[INFO]: Lap Time Stats -> Min = {round(np.min([np.max(TD['t_steps']) for TD in TDList]), 4)} s")
    print(f"[INFO]: Lap Time Stats -> Mean = {round(np.mean([np.max(TD['t_steps']) for TD in TDList]), 4)} s")
    print(f"[INFO]: Lap Time Stats -> Max = {round(np.max([np.max(TD['t_steps']) for TD in TDList]), 4)} s")
    print(f"[INFO]: Speed Stats -> Min = {round(np.min([np.min(TD['states'][:, 3]) for TD in TDList]), 4)} m/s")
    print(f"[INFO]: Speed Stats -> Mean = {round(np.mean([np.mean(TD['states'][:, 3]) for TD in TDList]), 4)} m/s")
    print(f"[INFO]: Speed Stats -> Max = {round(np.max([np.max(TD['states'][:, 3]) for TD in TDList]), 4)} m/s")
    print(f"[INFO]: Deviation Stats -> Min = {round(np.min([np.min(TD['deviations']) for TD in TDList]), 4)} m")
    print(f"[INFO]: Deviation Stats -> Mean = {round(np.mean([np.mean(TD['deviations']) for TD in TDList]), 4)} m")
    print(f"[INFO]: Deviation Stats -> Max = {round(np.max([np.max(TD['deviations']) for TD in TDList]), 4)} m")


#########################################################################################################
#########################################################################################################