import pandas as pd

df = pd.DataFrame({
    "week": [101, 102, 103, 104, 105, 106, 107, 108],
    "sales": [10, 12, 9, 15, 11, 14, 13, 16]
})

# Task: split this into train and test
# train = all rows where week <= 105
# test = all rows where week > 105

train = df[df['week']<=105]
test = df[df['week']>105]

print(train)
print(test)
print(f"Train shape: {train.shape}, Test shape: {test.shape}")