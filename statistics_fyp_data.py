import pandas
import matplotlib.pyplot as plt;
import seaborn as sns;

# load dataset
columns = ['day', 'hour', 'op', 'ftype', 'parent', 'fileSize',
           'p1', 'p2', 'p3', 'p4', 'filename',
           's1', 's2', 's3', 's4']
dataframe = pandas.read_csv("/home/anuradha/PycharmProjects/data/fyp/final/test-after-filtered.csv", header=None, names=columns)

print (dataframe.info())
print (dataframe.describe())
# print (dataframe.sample(5))
# print (dataframe.drop_duplicates().fileSize.value_counts())
# print (dataframe.drop_duplicates().filename.value_counts())
print (dataframe.drop_duplicates().s1.value_counts())
# print (dataframe.drop_duplicates().parent.value_counts())


# dataframe['op'].value_counts().plot(kind='bar', title='Training examples by operation types');
# dataframe['ftype'].value_counts().plot(kind='bar', title='Training examples by file types');
# dataframe['parent'].value_counts().plot(kind='bar', title='Training examples by parent folder');
# dataframe['hour'].value_counts().plot(kind='bar', title='Training examples by hour');
# dataframe['fileSize'].value_counts().plot(kind='bar', title='Training examples by file size');

# plt.show()

corr = dataframe.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
sns.plt.title("correlations among variables")
sns.plt.show()

#day of week, hour, operation type, file size have very weak correlations

#parent folder, file type has moderate correlation