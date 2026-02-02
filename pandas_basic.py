#pandas basic for ML
import pandas as pd
print("================== creating DataFrame ================ ")
data={
    "name": ["aman", "riya", "bharti", "abhay"],
    "study_hours": [5, 6, 7, 8],
    "marks": [50, 60, 91, 90]
}
df=pd.DataFrame(data)
print("DataFrame:\n",df)
print("\n=== basic data inspection ===")
print("\nhead:\n",df.head())
print("\ninfo:\n",df.info())
print("\ndescribe:\n",df.describe())

print("\n=== accessing columns ===")
print("\nmarks column:\n",df["marks"])

print("\n===accessing new column (feature engineering) ===")
df["result"]=df["marks"]>=60
print("DataFrame with result column:\n",df)

print("\n\n=== filtering data ===")
print("\nstudents who scored more than 60 marks:\n",df[df["marks"]>60])

print("\n\n=== handling missing values ===\n")
#creating missing values
df.loc[2,"marks"]=None                 
print("\nAfter inserting missing value:\n",df)
print("\nchecking for missing values:\n",df.isnull())
print("\ndrop missing values:")
df_cleaned=df.dropna()
print(df_cleaned)
print("\n=== done with pandas basics ===")
