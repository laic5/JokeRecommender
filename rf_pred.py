
# coding: utf-8

# # Random Forest! :D

# In[1]:

import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error as mse

import seaborn as sns; sns.set(color_codes=True)
import matplotlib.pyplot as plt

import random

from itertools import compress

random.seed(2)


# In[2]:

# GLOBAL VARIABLES
train_split = .8
show_plot = True
num_pred_jokes = 10 # number of jokes you want to predict for a user

# using a random person for demo purposes. can be changed
sample_user = {'major':'Statistics', 'age':21, 'birth_country':"China", 'gender':"Female",                'id':5, 'genre1':"Programming", 'genre2':None, 'type':'Puns', 'music':"Blues", 'movie':None}
c = 15 # for sample weights
train = False # either train/test split, or use all the data


# In[3]:

def lsa_fn(X_tfidf, dim_reduce = 20, print_var=False):
    from sklearn.decomposition import TruncatedSVD 
    from sklearn.preprocessing import Normalizer
    """
    INPUT:
    
    dim_reduce: the number of columns you expect for the results
    X_tfidf: ti-idf matrix 
    
    OUTPUT:
    matrix with reduced dim (should be number_of_jokes x dim_reduce) 
    """
    
    lsa = TruncatedSVD(dim_reduce, algorithm = 'arpack')

    # X_tfidf : 153 x 788 tf-idf matrix
    dtm_lsa = lsa.fit_transform(X_tfidf)

    #reduced matrix (combine this matrix w/ other features)
    dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa)
    
    if print_var:
        print(str(lsa.explained_variance_.cumsum()[-1] * 100) + "%")
    
    return(dtm_lsa)


# In[4]:

def preprocess_feature_df(tfidf, add_df):
    tfidf_df = pd.DataFrame(tfidf)
    tfidf_columns = ["tfidf" + str(i) for i in range((tfidf_df.shape[1]))]
    tfidf_df.columns = tfidf_columns
    feat = pd.concat([add_df, tfidf_df], axis=1)
    feat.rename(columns = {'id':'joke_id'}, inplace = True)
    
    return feat


# In[5]:

def remove_low_variance_users(df):
    user_var = {}
    for rater in df.joke_rater_id.unique():
        entries = df[(df['joke_rater_id']==rater)]
        ratings = entries.rating
        var = np.nanvar(ratings)
        if np.isnan(var) == False:
            user_var[rater] = var
            #print(str(rater) + ": " + str(np.nanvar(ratings)))

    bad_keys = dict((k, v) for k, v in user_var.items() if v < 0.4).keys()

    df = df[~df['joke_rater_id'].isin(bad_keys)].reset_index(drop=True) # remove low variance users
    # df = df.loc[0:13398] # remove single entry NaN users -- will cause an issue if the database changes
    
    return df


# In[72]:

def impute_NA(df):
    which_drop = df[df.isnull().sum(axis=1) > 2].index
    new_df = df.drop(which_drop)

    modes = new_df.mode()
    new_df.birth_country = new_df.birth_country.fillna("United States")
    new_df.preferred_joke_type = new_df.preferred_joke_type.fillna("Puns")
    new_df.preferred_joke_genre2 = new_df.preferred_joke_genre.fillna("Programming")
    new_df = new_df.drop(new_df[new_df.joke_type.isnull() == True].index)
    
    return new_df


# In[74]:

def change_category_to_dummy(df):
    
    # ignoes all numeric entries
    ignore_col = [i for i in range((df.shape[1])) if (df.iloc[:,i].dtype == np.int64) or (df.iloc[:,i].dtype == np.float64)]
    ignore_col.extend([list(df.columns).index("joke_id")])
    ignore_col = sorted(ignore_col)
    
    #new_df.iloc[:,string_col] = pd.get_dummies(new_df.iloc[:,string_col])
    string_col = []
    for i in range((df.shape[1])):
        if i not in ignore_col:
            string_col.append(i)
    # same thing as
    #string_col = [3, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    
    df = pd.concat([df.iloc[:,ignore_col], pd.get_dummies(df.iloc[:,string_col])], axis=1)
    
    return df


# In[78]:

def lasso_selection(df2):
    
    disclude_col = ['rating', 'joke_rater_id', 'joke_id']
    features = [col for col in df2.columns if col not in disclude_col]

    lasso = Lasso(alpha=.001, random_state=2).fit(df2[features], df2.rating)
    model = SelectFromModel(lasso, prefit=True)

    lasso_X = model.transform(df2[features])

    for i, feature in zip(model.get_support(), features): # get headers, since they get lost after lasso
        if i:
            disclude_col.append(feature)

    df3 = pd.concat([df2.rating.reset_index(drop=True), df2.joke_rater_id.reset_index(drop=True), 
                    df2.joke_id.reset_index(drop=True), pd.DataFrame(lasso_X)], axis=1)
    df3.columns = disclude_col
    
    return df3


# In[9]:

def weigh_samples_vector(df, user_id=None, c=2):
    '''
    Sets all weights equal to 1.
    If user_id exists in database/csv, then increase weights to c, where c >= 1.
    Works if user already exists (already rated jokes) or if new user.
    Return np array that is to be used in rf.fit
    c is tuneable to how much you want to weight the user's personal ratings.
    '''
    vector_length = df.shape[0]
    vector = np.ones(vector_length)
    if user_id in df.joke_rater_id.unique(): # is user already exists in database, increase weights
        idx = df3[df3.joke_rater_id == user_id].index
        vector[idx] = c # increase -- set to c
    return vector


# In[10]:

def mse(predicted, real):
    real = np.array(real)
    predicted = np.array(predicted)
    temp =  (real - predicted) * (real - predicted)
    n = len(real)
    mse = 1.0 / n * sum(temp)
    return mse


# In[11]:

def plot_pred_vs_actual(y_pred, y_test):
    ax = sns.regplot(x=y_test, y=y_pred.astype('float'), scatter_kws={'alpha':0.1})
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted Rating vs. Actual Rating")
    plt.show()


# In[12]:

def categorize_multiclass(label, user_label, entry, features, numRow):
    '''
    label is what the dummy string category name begins with, i.e. "birth_country_"
    user_label is the quantity inside the user_dict, accessed by specific key, i.e. user_dict['birth_country']
    '''
    user_class = label + str(user_label) 
    avail_labels = list(compress(features, [item.startswith(label) for item in features]))
    label_cols = [i for i, x in enumerate(entry.columns) if x in avail_labels]

    for col in label_cols:
        entry.iloc[:, col] = np.repeat(0, numRow) # set all to 0 for blank slate
    if user_class in avail_labels:
        entry.iloc[:,user_class] = np.repeat(1, numRow) # if user class present, set to 1
        
    return entry


# In[13]:

# convert data into one-hot

# assume data is a dict user_dict
def convert_sample_onehot(user_dict, df, features):
    '''
    Works with new user or existing user.
    Input: rater_id = user_dict (age, gender, birth_country, major, id)
           df = combined, cleaned dataframe (df3)
           
    Converts user data into variables inside the dataframe (i.e. df3) so you can pass into the random forest.
    '''
    pd.options.mode.chained_assignment = None
    
    if user_dict['id'] in df.joke_rater_id.unique(): # is user already exists in database
        return df[df.joke_rater_id == user_dict['id']]        
    
    entry = df[df.joke_rater_id == 476] # chose 476 randomly because they rated all 153 jokes
    numRow = entry.shape[0]
    
    if user_dict['gender'] == "Male": # gender
        entry.gender_Female = np.repeat(a=0, repeats=numRow)
    
    entry.age = np.repeat(user_dict['age'], numRow) # age
    entry.joke_rater_id = np.repeat(user_dict['id'], numRow)
    
    ## COUNTRY
    entry = categorize_multiclass("birth_country_", "birth_country", entry, features, numRow)
    
    ## MAJOR
    entry = categorize_multiclass("major_", "major", entry, features, numRow)
    
    ## PREFERRED JOKE GENRE 1
    entry = categorize_multiclass("preferred_joke_genre_", "genre1", entry, features, numRow)
    
    ## PREFERRED JOKE GENRE 2
    entry = categorize_multiclass("preferred_joke_genre2_", "genre2", entry, features, numRow)
    
    ## PREFERRED JOKE TYPE
    entry = categorize_multiclass("preferred_joke_type_", "type", entry, features, numRow)
    
    ## MOVIE
    entry = categorize_multiclass("favorite_movie_genre_", "movie", entry, features, numRow)
    
    ## MUSIC
    entry = categorize_multiclass("favorite_music_genre_", "music", entry, features, numRow)
        
    return entry

#entry = df3[df3.joke_rater_id == 476]
#numRow = entry.shape[0]
#categorize_multiclass("preferred_joke_genre_", "joke1", entry, features, numRow)


# In[14]:

def get_topk_jokes(user_df, rf, joke_ids, features, k=10):
    '''
    Returns top k jokes for user (default=10).
    user_df is output from convert_sample_onehot.
    Assumes random forest rf is already trained.
    '''
    preds = rf.predict(user_df[features])
    
    df = pd.DataFrame(joke_ids)
    df['pred'] = preds
        
    return df


def train_and_test(df3, user_id):

    unique_rater = df3.joke_rater_id.unique() # all unique users
    train_size = round(len(unique_rater) * train_split) # 80/20 train/test split!

    train_idx = np.random.choice(unique_rater, train_size, replace=False) # get randomly train_size number of users to put into train
    test_idx = [i for i in unique_rater if i not in train_idx] # remaining users go to test

    train_df = df3.loc[df3['joke_rater_id'].isin(train_idx)]
    test_df = df3.loc[df3['joke_rater_id'].isin(test_idx)]
    
    # time to run random forest regressor
    rf = train_rf(train_df, user_id)
    
    # testing
    y_test = test_df.rating
    
    disclude = ['joke_rater_id', 'rating']
    features = [col for col in df3.columns if col not in disclude]
    
    y_pred = rf.predict(test_df[features]).astype('float')
    print("Test MSE is: " + str(mse(y_test, y_pred)))
    
     # see distribution of predicted joke score vs. actual joke value
    if show_plot:
        plot_pred_vs_actual(y_pred, y_test)


# In[34]:

def train_rf(df3, user_id, print_importance=False):
    y = df3.rating
    Y_list = list(y.values)
    
    disclude = ['joke_rater_id', 'rating']
    features = [col for col in df3.columns if col not in disclude]

    sample_weights = weigh_samples_vector(df=df3, user_id=user_id, c=c) # weigh user's ratings more
    sample_weights = np.ravel(normalize(sample_weights.reshape((-1, 1)), axis=0))
    min_weight = min(sample_weights) + 0.001

    rf = RandomForestRegressor(n_estimators=50, max_features='sqrt', random_state=42,                                    max_depth=10, min_weight_fraction_leaf=min_weight) # tuneable parameters
    rf.fit(df3[features], y, sample_weight=sample_weights)
    
    if print_importance:
        # see what factors are most important
        s = pd.DataFrame((rf.feature_importances_))

        s = s.transpose()
        s.columns = features
        s = s.transpose()

        print("10 most important features: ")
        print(s.sort_values(by=0, ascending=False).head(10))
    
    return rf


# In[81]:

def get_preds(user, df3, rf, k):
    
    ## QUERYING JOKE PREDICTIONS FOR NEW USER
    
    disclude = ['joke_rater_id', 'rating']
    features = [col for col in df3.columns if col not in disclude]
    
    user_df = convert_sample_onehot(user, df3, features)
    joke_ids = df3[df3.joke_rater_id == 476].joke_id 
    preds = get_topk_jokes(user_df, rf, features=features, joke_ids=joke_ids, k=k)
    
    return preds


# In[84]:

def prepare_df():
    # 3 dfs from csvs
    rater_df = pd.read_csv("JokeRater.csv")
    rating_df = pd.read_csv("JokeRating.csv")
    joke_df = pd.read_csv("Joke.csv")
    
    ## PRE-PROCESSING DATAAAAAA
    
    # change column names for merging purposes
    rater_df.rename(columns = {'id':'joke_rater_id'}, inplace = True)
    joke_df.rename(columns = {'id':'joke_id'}, inplace = True)
    joke_df['joke_id'] = joke_df['joke_id'].astype(float)
    rater_df = rater_df.drop('joke_submitter_id', axis=1)
    joke_df = joke_df.drop('joke_submitter_id', axis=1)
    joke_df = joke_df.drop('joke_source', axis=1)
    
    # add tf-idf features
    feature_df = pd.read_csv("feature_tfidf.csv")
    add_features = feature_df.iloc[:,1:5] # misc. features like avg length, num words

    X_tfidf = feature_df.iloc[:,6:]
    
    # use LSA to reduce down number of tfidf columns
    reduced_tfidf = lsa_fn(X_tfidf, 100) # 79.3% variance explained
    
    feat = preprocess_feature_df(add_df=add_features, tfidf=reduced_tfidf)

    joke_df = pd.merge(joke_df, feat, on='joke_id', how='outer') # combine new features to joke dataframe
    
    # combine joke raters with their ratings
    df = pd.merge(rating_df, rater_df, on="joke_rater_id", how="outer")
    df = df.drop('id', axis=1)
    
    # deal with low variance users
    df = remove_low_variance_users(df)
    
    # finally, add jokes in
    df = pd.merge(df, joke_df, on='joke_id', how='outer')
    df = df.drop('subject', axis=1)
    df = df.drop('joke_text', axis=1)
    
    # get rid of high NaN entries, and replace categories with modes
    df = impute_NA(df)

    # convert categorical variables into dummies (one-hot)
    df2 = change_category_to_dummy(df)
    
    # lasso feature selection
    df3 = lasso_selection(df2)
    
    return df3


# In[85]:

def query(user):
    
    df3 = prepare_df()
    
    ## RANDOM FORESTTTT
    
    if train: # train/test to get test MSE
        train_and_test(df3, user['id']) # fitted to train dataset
        
    # now using all the data to train random forest
    rf = train_rf(df3, user['id'], print_importance=True)
    
    ## QUERYING JOKE PREDICTIONS FOR NEW USER
    preds = get_preds(user, df3, rf, num_pred_jokes)
    
    return preds # preds contains all the joke predicted scores

train = False
p = query(sample_user)


# In[ ]:

if __name__ == '__main__':
    query()

