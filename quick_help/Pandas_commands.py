# Reading
sen_pdf = pd.read_csv("../data/sentiment_raw.csv", sep=',',  error_bad_lines=False)

# Null Check
sen_pdf[['word', 'polarity', 'example']].isnull().sum()
# Filter Null
filtered_df = df[df['var2'].isnull()]
# check NaN type
pd.isna(x)
df[pd.isnull(df.col1)]
np.isnan(x)

# Column Rename
df = df.rename(columns={'oldName1': 'newName1', 'oldName2': 'newName2'})
df.rename(columns={'oldName1': 'newName1', 'oldName2': 'newName2'}, inplace=True)
df.columns = ['col1', 'col2', 'col3']

# Slicing
df.loc[:5, 'Address'] # df.loc[0:5, 'Address'] works as well
df.iloc[:5, 1]

# New column create and assign
df.loc[df.Price > 1000000, 'IsExpensive'] = 1
df.loc[(df.Price > 1400000) & (df.Type == 'h'), 'Category'] = 'Expensive House'
df.loc[df.Price > 1400000, 'Price'] = df.Price * 0.95

# Proper way to append Pandas DataFrame
dictinary_list = []
for i in range(0, end_value, 1):
    dictionary_data = {k: random.random() for k in range(30)}
    dictinary_list.append(dictionary_data)
df_final = pd.DataFrame.from_dict(dictinary_list)

# Convert dict to dataframes. Dict like [{'col1':'value'},{'col1':'value'},{'col1':'value'}]
df.to_dict('records')

# Pandas joins
merged_inner = pd.merge(left=survey_sub, right=species_sub, left_on='species_id', right_on='species_id')
merged_left = pd.merge(left=survey_sub, right=species_sub, how='left', left_on='species_id', right_on='species_id')
merged_left = pd.merge(left=survey_sub, right=species_sub, how='right', left_on='species_id', right_on='species_id')
merged_left = pd.merge(left=survey_sub, right=species_sub, how='outer', left_on='species_id', right_on='species_id')

# Sorting
df.sort_values(['Col_1', 'Col_2'], ascending=True).head()

# Group by count
df.groupby(['col1', 'col2']).size().reset_index(name='counts')

# Apply function with progress
def process_text(sample_review):
    return sample_review + "."
tqdm().pandas()
data_df['ProcessedTexts'] = data_df['Texts'].progress_apply(process_text)

def check_similarity(sampl_a, sample_b):
    return sampl_a + sample_b
tqdm().pandas()
data_df['AspectCategories_Similarity'] = data_df[['AspectCategories_split', 'AspectCategories_Similarity']].progress_apply(lambda x:check_similarity(*x) , axis=1)