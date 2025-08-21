import os
import pandas as pd
import pickle

# Read CSV file
files_path = "../data/sensors/"

for file in os.listdir(files_path):

    csv_path = os.path.join(files_path, file)
    df = pd.read_csv(csv_path, sep=",")

    # Save data as pickle
    pickle_file_path = f"{csv_path.rsplit('.', 1)[0]}.pickle"
    # with open(pickle_file_path, 'wb') as picklefile:
    #     pickle.dump(df, picklefile)
    print(csv_path)
    print(pickle_file_path)
