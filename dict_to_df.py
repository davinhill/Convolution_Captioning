import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))  # needed for BlueWaters
import argparse
import json
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('--path', type=str, default='./', help='path of file')

args = parser.parse_args()

datafile = json.load(args.path)
df = pd.DataFrame()

for key in datafile[0]:
   df[key] = [value[key] for value in datafile]

import pdb; pdb.set_trace()

df.to_csv('model_accuracy.csv')
