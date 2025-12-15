import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("videogamesales.csv")
df["global_sales"] = (
    df["NA_Sales"] +
    df["EU_Sales"] +
    df["JP_Sales"] +
    df["Other_Sales"]
)

df_sorted = df.sort_values("global_sales", ascending=False)
print(df_sorted.head(10))
genre_sales = df.groupby("Genre")["global_sales"].sum()

plt.figure()
genre_sales.plot(kind="bar")
plt.title("Total Global Sales by Genre")
plt.xlabel("Genre")
plt.ylabel("Global Sales (Millions)")
plt.tight_layout()
plt.show()
gta_df = df[df["Name"].str.contains("Grand Theft Auto", case=False, na=False)]

gta_result = gta_df[["Name", "Platform", "Year"]].copy()
gta_result["EU_JP_Sales"] = gta_df["EU_Sales"] + gta_df["JP_Sales"]

print(gta_result)
gta_sales = [
    gta_df["NA_Sales"].sum(),
    gta_df["EU_Sales"].sum(),
    gta_df["JP_Sales"].sum(),
    gta_df["Other_Sales"].sum()
]

labels = ["North America", "Europe", "Japan", "Other"]

plt.figure()
plt.pie(gta_sales, labels=labels, autopct="%1.1f%%")
plt.title("Regional Sales Distribution of Grand Theft Auto Games")
plt.show()
