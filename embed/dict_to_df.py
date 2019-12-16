# Function to convert the model_accuracy.json dictionary file to a pandas dataframe, then export to csv


import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))  # needed for BlueWaters
import argparse
import json
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('--path', type=str, default='./', help='path of file')

args = parser.parse_args()

datafile = json.load(open(args.path, 'r'))
df = pd.DataFrame()

for key in datafile[0]:
    df[key] = [value[key] for value in datafile]

df.to_csv(os.path.join(os.path.dirname(args.path), 'model_accuracy.csv'))

