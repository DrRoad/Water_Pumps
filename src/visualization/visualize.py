# Hide deprecation warnings
import warnings
warnings.filterwarnings('ignore')

# Common imports
import numpy as np
import pandas as pd
import seaborn as sns
import squarify
import missingno as msno
from statsmodels.graphics.mosaicplot import mosaic


# To plot pretty figures
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# To format floats
from IPython.display import display
pd.set_option('display.float_format', lambda x: '%.5f' % x)


# Load the csv files that we'll need for the EDA phase into Pandas dataframes, properly parsing dates
def visualize(df_train, df_labels):
    print(df_train.head().T)
    print(df_train.info())
    msno.matrix(df_train)

    # Numerical features
    print(df_train.describe())

    # Let's inspect now the categorical features
    cat_df = pd.DataFrame(columns=["Feature", "Cardinality","% Missings"])

    total_cardinality = 0

    i=0

    for col in df_train.columns:
        if (df_train[col].dtype == np.object):
            cat_df.loc[i,"Feature"] = col
            cat_df.loc[i,"Cardinality"] = len(df_train[col].unique())
            total_cardinality += len(df_train[col].unique())
            pct_of_missing_values = float((len(df_train[col]) - df_train[col].count()) / len(df_train[col]))
            cat_df.loc[i,"% Missings"] = pct_of_missing_values*100
            i+=1

    print("Total cardinality of categorical features:",total_cardinality)

    print(cat_df)

    # Visualizations
    data_viz = pd.concat([df_train,df_labels['status_group']],axis=1)

    # Label distribution
    plt.figure(figsize=(14,7))
    sns.countplot(x='status_group',data=data_viz, palette="Greens_d");
    plt.show()

    # Construction year distribution
    # We need to filter the instances with year 0 that will be taken care of in the Data Preparation part
    plt.figure(figsize=(14,7))
    sns.distplot(data_viz['construction_year'][data_viz['construction_year']>0]);
    plt.show()

    # Water pump geographical distribution with population proportional circles and year of pump color bar
    # We need to filter the instances with year 0, longitud 0 and latitude 0 in this case
    data_viz[data_viz['longitude']>0][data_viz['latitude']<0][data_viz['construction_year']>0].plot\
        (kind="scatter", x="longitude", y="latitude", alpha=0.4,
        s=data_viz["population"]/10, label="population", figsize=(14,10),
        c="construction_year", cmap=plt.get_cmap("jet"), colorbar=True,
        sharex=False)
    plt.legend;
    plt.show()

    # Correlation heatmap of the numerical features
    cor = data_viz.corr()
    plt.figure(figsize=(14,13))
    sns.heatmap(cor, square=True, annot=True, cbar=False);
    plt.show()

    # Boxplot of label distribution by pump construction year
    plt.figure(figsize=(14,7))
    sns.boxplot(x='status_group', y="construction_year", data=data_viz[data_viz['construction_year']>0]);
    plt.show()

    # A different way of seeing this same concept, with proportions within the distribution plot, using violin plots
    fig, ax = plt.subplots(figsize=(14,12));
    ax = sns.violinplot(x='status_group', y="construction_year",\
                        data=data_viz[data_viz['construction_year']>0], split=True)
    plt.show()

    # Mosaic of permit distribution per label
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))
    fig = mosaic(data_viz, ['status_group', 'permit'], axes, title="Permit distribution per label")
    plt.show()

    # Mosaic of public meeting distribution per label
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))
    fig = mosaic(data_viz, ['status_group', 'public_meeting'], axes, title="Public meeting distribution")
    plt.show()

    # Mosaic of source class distribution per label
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))
    fig = mosaic(data_viz, ['status_group', 'source_class'], axes, title="Source class distribution per label")
    plt.show()

    # Bar charts of some relevant categorical features per label
    variables = ['quantity','payment','source_type','waterpoint_type']
    label = 'status_group'
    plot_tables(data_viz,label,variables)
    plt.show()

    # Value distribution for some important features with low cardinalit
    variables = ['basin','extraction_type_class','management','management_group',\
                 'water_quality','source','source_class']
    plot_proportions(data_viz,variables)

    # Height distribution per label
    plt.figure(figsize=(14,10))
    p1=sns.kdeplot(data_viz[['gps_height','status_group']][data_viz.status_group == 'functional']\
                   [data_viz.gps_height > 0].gps_height, shade=True, color="g",label='functional')
    p1=sns.kdeplot(data_viz[['gps_height','status_group']][data_viz.status_group == 'non functional']\
                   [data_viz.gps_height > 0].gps_height, shade=True, color="r",label='non functional')
    p1=sns.kdeplot(data_viz[['gps_height','status_group']][data_viz.status_group == 'functional needs repair']\
                   [data_viz.gps_height > 0].gps_height, shade=True, color="y",label='functional needs repair')
    plt.show()

    # Pair plot of the relevant numerical features against each other, differentiating by label value
    sns.set(style="ticks")
    sns.pairplot(data_viz[['population','num_private','amount_tsh','status_group']],\
                        hue="status_group", diag_kind="kde");
    plt.show()

def plot_tables(data,label,variables):
    fig, axes = plt.subplots(nrows=len(variables), ncols=1, figsize = (14,14))
    for i,variable in enumerate(variables):
        ax = pd.pivot_table(data[[label,variable]], index = [label,variable], aggfunc = len)\
        .unstack(variable).plot(kind='bar', ax=axes[i], sharex=True, title=variable, rot=0, colormap='viridis')
        ax.legend(bbox_to_anchor=(1,1),loc="upper left")
    plt.show()

def plot_proportions(data,variables):
    colors = ["red","green","blue", "grey", "black", "brown", "yellow", "pink", "purple", "khaki", "cyan","white"]
    fig, axes = plt.subplots(nrows=len(variables), ncols=1, figsize = (14,len(variables)*4))
    for i,variable in enumerate(variables):
        ax = squarify.plot(sizes=data[variable].value_counts().tolist(), \
                      label=data[variable].value_counts().index.tolist(), color=colors, ax=axes[i], alpha=.4)
        ax.set_title(variable)
    plt.show()
