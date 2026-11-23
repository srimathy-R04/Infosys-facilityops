import pandas as pd

cols = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]

train = pd.read_csv("maintenance_data/train_FD001.txt", sep=r"\s+", header=None, names=cols)
test = pd.read_csv("maintenance_data/test_FD001.txt", sep=r"\s+", header=None, names=cols)
rul = pd.read_csv("maintenance_data/RUL_FD001.txt", header=None)

print("train shape:", train.shape)        # expect (20631, 26)
print("engines in train:", train.unit.nunique())   # expect 100
print("test shape:", test.shape)          # expect (13096, 26)
print("RUL file shape:", rul.shape)       # expect (100, 1)

life = train.groupby("unit").cycle.max()  # how many cycles each engine lasted
print(life.describe())                    # min about 128, max about 362, mean about 206

print(train[cols[5:]].std().round(3))     # which sensors never change?
print(train.head())