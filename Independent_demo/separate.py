import pandas as pd

df = pd.read_csv('../data/SAML-D.csv')

df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

separate_dict = {}

for month, group in df.groupby(df['Date'].dt.strftime('%Y-%m')):
    separate_dict[month] = group

    filename = f"{month}.csv"
    group.to_csv(filename, index=False)

    print(f"separate save: {filename}")