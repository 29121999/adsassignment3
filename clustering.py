import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import seaborn as sns
from sklearn import cluster

df = pd.read_csv('dataset.csv')

df.head()

"""# Cleaning the Data"""

def summary_stats(df, wide=True):
    """Takes in a dataframe and prints the number of rows, columns and missing values.""" 
    print(f"""Number of rows{" (i.e. countries)" if wide else ""} in the dataframe: {df.shape[0]:,}
Number of columns{" (i.e. indicators)" if wide else ""} in the dataframe: {df.shape[1]}
Number of missing values in the dataframe: {df.isna().sum().sum():,}""")

summary_stats(df, False)

iso_countries = ['AFG', 'ALB', 'DZA', 'ASM', 'AND', 'AGO', 'ATG', 'ARG', 'ARM', 'ABW', 'AUS', 'AUT', 'AZE', 'BHS', 'SXM',
                 'BHR', 'BGD', 'BRB', 'BLR', 'BEL', 'BLZ', 'BEN', 'BMU', 'BTN', 'BOL', 'BIH', 'BWA', 'BRA', 'VGB', 'BRN',
                 'BGR', 'BFA', 'BDI', 'CPV', 'KHM', 'CMR', 'CAN', 'CYM', 'CAF', 'TCD', 'CHI', 'CHL', 'CHN', 'COL', 'COM',
                 'COD', 'COG', 'CRI', 'CIV', 'HRV', 'CUB', 'CUW', 'CYP', 'CZE', 'DNK', 'DJI', 'DMA', 'DOM', 'ECU', 'EGY',
                 'SLV', 'GNQ', 'ERI', 'EST', 'SWZ', 'ETH', 'FRO', 'FJI', 'FIN', 'FRA', 'PYF', 'GAB', 'GMB', 'GEO', 'DEU',
                 'GHA', 'GIB', 'GRC', 'GRL', 'GRD', 'GUM', 'GTM', 'GIN', 'GNB', 'GUY', 'HTI', 'HND', 'HKG', 'HUN', 'ISL',
                 'IND', 'IDN', 'IRN', 'IRQ', 'IRL', 'IMN', 'ISR', 'ITA', 'JAM', 'JPN', 'JOR', 'KAZ', 'KEN', 'KIR', 'PRK',
                 'KOR', 'XKX', 'KWT', 'KGZ', 'LAO', 'LVA', 'LBN', 'LSO', 'LBR', 'LBY', 'LIE', 'LTU', 'LUX', 'MAC', 'MDG',
                 'MWI', 'MYS', 'MDV', 'MLI', 'MLT', 'MHL', 'MRT', 'MUS', 'MEX', 'FSM', 'MDA', 'MCO', 'MNG', 'MNE', 'MAR',
                 'MOZ', 'MMR', 'NAM', 'NRU', 'NPL', 'NLD', 'NCL', 'NZL', 'NIC', 'NER', 'NGA', 'MKD', 'MNP', 'NOR', 'OMN',
                 'PAK', 'PLW', 'PAN', 'PNG', 'PRY', 'PER', 'PHL', 'POL', 'PRT', 'PRI', 'QAT', 'ROU', 'RUS', 'RWA', 'WSM',
                 'SMR', 'STP', 'SAU', 'SEN', 'SRB', 'SYC', 'SLE', 'SGP', 'SVK', 'SVN', 'SLB', 'SOM', 'ZAF', 'SSD', 'ESP',
                 'LKA', 'KNA', 'LCA', 'MAF', 'VCT', 'SDN', 'SUR', 'SWE', 'CHE', 'SYR', 'TJK', 'TZA', 'THA', 'TLS', 'TGO',
                 'TON', 'TTO', 'TUN', 'TUR', 'TKM', 'TCA', 'TUV', 'UGA', 'UKR', 'ARE', 'GBR', 'USA', 'URY', 'UZB', 'VUT',
                 'VEN', 'VNM', 'VIR', 'PSE', 'YEM', 'ZMB', 'ZWE']
df = df[df['Country Code'].isin(iso_countries)]
print("Number of countries:", df['Country Code'].nunique())

summary_stats(df, False)

len(iso_countries)

df = df.iloc[:,:-1] # The last column is full of missing values and so we can drop it.

plt.figure(figsize=(17,7))
df.iloc[:, -10:].isna().sum().plot(kind='barh')
plt.title('Number of missing values for the years 2010 to 2019 in the World Bank Dataset')
plt.ylabel('Year')
plt.xlabel('Number of Missing Values');

df_2017 = df[['Country Name', 'Indicator Name', '2017']]

summary_stats(df_2017, False)

# changing the dataframe shape to make it easier to identify and remove missing values
df_wide = df_2017.pivot_table(values='2017', index='Country Name', columns='Indicator Name')

df_wide.head(2)

summary_stats(df_wide)

df_wide = df_wide.dropna(axis=1, thresh=168) # selecting the indicators at least 168 countries

summary_stats(df_wide)

df_wide = df_wide.fillna(df_wide.mean())

summary_stats(df_wide)

"""# Clustering"""

X = df_wide.values
country_names = list(df_wide.index)

from sklearn.preprocessing import StandardScaler
ss1 = StandardScaler()
X_ss1 = ss1.fit_transform(X)

# plt.plot(df["1960"], df["2020"],  'ko'); 

# #plt.plot(df["2020"], df["1960"], 'b-', linewidth = 2,)
# plt.plot(df["1960"], df["2020"], 'b-', linewidth = 4)
# plt.legend()
# plt.xlabel('x'); plt.ylabel('y'); plt.title('Data');

#fit_poly(train, y_train, test, y_test, degrees = 25, plot='test')

# empty list for inertia values
inertia = []

for i in range(1,10):
    # instantiating a kmeans model with i clusters
    kmeans = cluster.KMeans(n_clusters=i)
    
    # fitting the model to the data
    kmeans.fit(df_wide)
    
    # appending the inertia of the model to the list
    inertia.append(kmeans.inertia_)
    
    # ignore this if statement
    if i == 3:
        elbow = kmeans.inertia_

# creating a list with the number of clusters
number_of_clusters = range(1,10)

ig = plt.figure(figsize=[7,5])
plt.plot(number_of_clusters, inertia)
plt.plot(3, elbow, 'ro', label='Elbow')
plt.legend()
plt.xlabel('# of clusters')
plt.ylabel('Inertia')
plt.savefig("inertia_plot.png")
plt.show()

# Define a function to perform clustering and plot the results
def clust(k, std_data):
    # Initialize KMeans object with k clusters
    kmeans = cluster.KMeans(k)
    # Fit the KMeans model to standardized data and predict cluster labels
    clusters = kmeans.fit_predict(std_data)
    # Scatter plot of CO2 emissions per PPP $ of GDP vs CO2 emissions per capita with points colored by cluster
    plt.scatter(df_wide["CO2 emissions (kg per PPP $ of GDP)"], df_wide["CO2 emissions (metric tons per capita)"], c=clusters, cmap='rainbow')
    # Set the x and y-axis labels
    plt.xlabel('CO2 emissions (kg per PPP $ of GDP')
    plt.ylabel('CO2 emissions (metric tons per capita)')
    # Show the plot
    plt.show()

# Call the clust function with k=3 clusters and standardized data X_ss1
clust(3, X_ss1)

# Create a dendrogram of hierarchical clustering with ward linkage method, country names as labels, and standardized data X_ss1
fig, ax = plt.subplots(figsize=(24,40))
sch.dendrogram(sch.linkage(X_ss1, method='ward'), labels=country_names, orientation='left', leaf_font_size=8)
# Set the tick font size for y-axis to 14 and x-axis to 22
ax.tick_params(axis='y', which='major', labelsize=14)
ax.tick_params(axis='x', which='major', labelsize=22)
# Set the title of the plot
ax.set_title('Agglomerative Hierarchical Clustering Diagram of the Countries', size=34);
# Set the x-axis label
ax.set_xlabel('Height', size=22)
# Save the plot as a PNG file
fig.savefig(r'County Clustering with Population Dependent Indicators.png');

# Compute summary statistics for df_wide
summary_stats(df_wide)

# Select columns with population adjusted titles
df_wide_non_pop = df_wide.filter(regex='|'.join(['%', ' per ', ' index ', 'days']) )
# Compute summary statistics for df_wide_non_pop
summary_stats(df_wide_non_pop)

# Print a list of all 277 indicators used in the final analysis
print(df_wide_non_pop.columns.tolist())

# Assign values and country names for clustering with non-population dependent indicators
X = df_wide_non_pop.values

country_names = list(df_wide_non_pop.index)

# Standardize the data
ss2 = StandardScaler()
X_ss2 = ss2.fit_transform(X)

# Create a dendrogram of hierarchical clustering with ward linkage method, country names as labels, color threshold at 37, and standardized data X_ss2
fig, ax = plt.subplots(figsize=(24,40))
sch.dendrogram(sch.linkage(X_ss2, method='ward'), labels=country_names, orientation='left', leaf_font_size=8, color_threshold=37)
# Set the tick font size for y-axis to 14 and x-axis to 22
ax.tick_params(axis='y', which='major', labelsize=14)
ax.tick_params(axis='x', which='major', labelsize=22)
# Set the title of the plot
ax.set_title('Agglomerative Hierarchical Clustering Dendrogram of Countries\nwith Non-Population Dependent Indicators', size=34);
ax.set_xlabel('Height', size=22)
fig.savefig(r'County Clustering with Non-Population Dependent Indicators.png');





