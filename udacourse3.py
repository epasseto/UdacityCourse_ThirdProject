#import libraries
import numpy as np
import pandas as pd
import scipy.stats as stats
import progressbar
import pickle

#graphs
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.style as mstyles
import matplotlib.pyplot as mpyplots #plt
#from matplotlib.pyplot import hist
#from matplotlib.figure import Figure

#First part
from statsmodels.stats import proportion as proptests
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
from time import time

#second part
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from scipy.sparse import csr_matrix
from collections import defaultdict
from IPython.display import HTML

#altered 2021-11-18
###Recommendations by IBM#######################################################
#########1#########2#########3#########4#########5#########6#########7#########8
def fn_create_test_and_train_user_item(df,
                                       train_size,
                                       verbose=False):
    '''This function takes a dataset, splits it in Train and Test and returns
    some useful dataframes and series for analysis.
    
    The code here presented is strongly based on the Udacity Jupyter Notebook
    Recommendations wit IBM, and is part of the learning exercise, necessary to
    complete the third project of Data Scientist course. It was conceived as a
    learning guide for students who wants to go deeper in Python/Pandas.


    Inputs:
      - df (mandatory) - a kind of df_inter_enc Dataframe to be splitted into a 
        Train and a Test Dataframe (Users as rows and Articles as columns) - 
        (Pandas Dataframe)
      - train_size (mandatory) - size (number of rows) for the Train Dataframe
        (Integer)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Outputs:
      - df_user_item_train - a Pandas user-item Dataframe for Train
      - df_user_item_test - a Pandas user-item Dataframe for Test
      - test_id - all of the test User IDs
      - test_art - all of the test Article IDs
    '''
    if verbose:
        print('###function create test and train user-item started')
                
    start_main = time()
    orig_row = df.shape[0]
    orig_col = df.shape[1]

    if verbose:
        print('original dataset: {} x {}'.format(orig_row, orig_col))

    if train_size >= orig_row:
        raise Exception('cannot process, train is as or larger than df')
    else:
        df_train = df.head(train_size)
        df_test = df.tail(orig_row-train_size)
        
    #first step: create my user items Arrays
    df_user_item_train = fn_create_user_item_matrix(
                             df=df_train,
                             verbose=verbose
    )
    df_user_item_test = fn_create_user_item_matrix(
                            df=df_test,
                            verbose=verbose
    )
    if verbose:
        print('dataset sizes - train: {} x {}, test: {} x {}'\
              .format(df_train.shape[0], 
                      df_train.shape[1], 
                      df_test.shape[0], 
                      df_test.shape[1]
    ))
    #second step, rows (Users) and columns (Articles) for test dataframe
    sr_test_id = df_user_item_test.index.values
    sr_test_art = df_user_item_test.columns.values

    end = time()

    if verbose:
        print('elapsed time: {:.4f}s'.format(end-start_main))

    return df_user_item_train, df_user_item_test, sr_test_id, sr_test_art
    
#########1#########2#########3#########4#########5#########6#########7#########8
def fn_create_user_item_matrix(df,
                               verbose=True):
    '''This function returns a matrix with user ids as rows and article ids on 
    the columns with 1 values where a user interacted with  an article and a 0. 
    Otherwise create the user-article matrix with 1's and 0's.

    The code here presented is strongly based on the Udacity Jupyter Notebook
    Recommendations wit IBM, and is part of the learning exercise, necessary to
    complete the third project of Data Scientist course. It was conceived as a
    learning guide for students who wants to go deeper in Python/Pandas.

    Inputs:
      - df (mandatory) - pandas dataframe in the format of df_inter_enc,
        having article_id as index, plus title and user_id columns
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - df_out - user-item matrix, having users as rows and articles as column,
        maximum value allowed is 1 for each entry. 
    '''
    if verbose:
        print('###function create user item matrix started')
    
    #only to avoid modifying the original dataframe
    df_out = df.copy()
    start = time()

    #step 1: creating the user-item matrix
    group = df_out.groupby(by=['user_id', df_out.index])['title'].count()
    group = group.unstack()

    #step 2: ensuring that there will be no NaNs in dataframe
    group = group.fillna(0)

    #step 3: ensuring that the only possible values will be 0 or 1
    for article in group:
        group[article] = group[article].apply(lambda x: x if x == 0 else 1)
    
    end = time()
    
    if verbose:
        print('user matrix has {} users (rows) and {} articles (columns)'.\
        format(group.shape[0], group.shape[1]))
        print('elapsed time: {:.4f}s'.format(end-start))
    
    return group #user-item dataframe

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_email_mapper(df_input,
                    verbose=False):
    '''This function maps the user e-mail to a user ID. Then e-mail column will
    be removed. This is a standard function on this notebook
    Inputs:
      - df_input (mandatory) - your relationship dataframe (Pandas Dataframe)
      - verbose (optional) - if you want some verbosity on this process, please
      turn it on (default=False)
    Output:
      - df_output - your relationship dataframe, modified (Pandas Dataframe)
    '''
    if verbose:
      print('###e-mail mapper function started')
    #just to prevent modifying the original dataframe  
    df_output = df_input.copy()#deep=True)
    coded_dict = dict()
    cter = 1
    ls_encoded = []
    
    for val in df_output['email']:
        #if verbose:
        #    print('iteration')
        if val not in coded_dict:
            if verbose:
              print('*for value {} it was append {}'.format(val, cter))
            coded_dict[val] = cter
            cter+=1
        ls_encoded.append(str(coded_dict[val]))

    #exclude old column and add the new one
    del df_output['email']
    df_output['user_id'] = ls_encoded    

    return df_output
    
#########1#########2#########3#########4#########5#########6#########7#########8
def fn_find_similar_user(df_user_item,
                         user_id,
                         max_usr=None,
                         verbose=False):
    '''This function computes the similarity of every pair of users based on 
    the dot product.

    The code here presented is strongly based on the Udacity Jupyter Notebook
    Recommendations wit IBM, and is part of the learning exercise, necessary to
    complete the third project of Data Scientist course. It was conceived as a
    learning guide for students who wants to go deeper in Python/Pandas.
    
    Inputs:
      - user_item (mandatory) - (pandas dataframe) matrix of users by articles: 
        1's when a user has interacted with an article, 0 otherwise with index
        and column names in numerical string format
      - user_id (mandatory) - (int) a user_id
      - max_usr (optional) - if you want to inform the maximum number of entries
        that you wish for your list (default=None)
      - verbose (optional) - if you want some verbosity during the processing
        (default=False)
    Output:
      - similar_user - (list) an ordered list where the closest users 
        (largest dot product users) are listed first
    '''
    if verbose:
        print('###function find similar users started')

    start = time()
    user = df_user_item.loc[str(user_id)]

    if verbose:
        print('user {} has {} articles viewed'\
              .format(user_id, user[user == 1].count()))

    #first, compute the similarity
    similar = df_user_item.dot(user) #for each user -> provided user
    #second, remove his own ego from the series
    similar = similar.drop('1') #bye bye himself!
    #third, sort by similarity
    similar_most = similar.sort_values(ascending=False)

    if verbose:
        print('*for reference: 10th value is', similar_most[9])
    
    #restrict maximum number of entries
    if max_usr is not None:
      similar_most = similar_most[:max_usr]
      
    if verbose:
        print('returning list of {} max users'.format(max_usr))

    #finally, take an id list
    similar_lst = similar_most.index.to_list()

    end = time()
    if verbose:
        print('elapsed time: {:.4f}s'.format(end-start))

    return similar_lst #users ids in order from most to least similar   

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_get_article_name(df,
                        ls_article,
                        alternative_df=False, 
                        verbose=False):
    '''This function takes the articles IDs and return a list of the articles
    titles.

    The code here presented is strongly based on the Udacity Jupyter Notebook
    Recommendations wit IBM, and is part of the learning exercise, necessary to
    complete the third project of Data Scientist course. It was conceived as a
    learning guide for students who wants to go deeper in Python/Pandas.
    
    Inputs:
      - df (mandatory) - an article detailing dataframe (Panda Dataframe)
        *option 1: df_inter_enc (default) - dataset for working
        *option 2: df_article (alternative) - an alternative dataset
      - ls_article (mandatory) - a list of article ids (Python List)
      - alternative_df (optional) - if you want to work with df_article, please
        check it as True - (boolean, default=False)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - art_name - a list of the name of the articles (Python List)
    '''
    if verbose:
        print('###get article name started')        

    start = time()

    #improving robustness of the system
    ls_aux = []
    for art in ls_article:
        ls_aux.append(int(float(art)))
    ls_article = ls_aux
    
    if alternative_df: #df_article
        if verbose:
            print('working with df_article type of dataframe')
        art = df.filter(items=ls_article, axis=0)['doc_full_name'] #collecting
        art_name = art.values.tolist()
    else: #df_inter_enc
        if verbose:
            print('working with df_inter_enc type of dataframe')
        art_name = []

        for art_id in ls_article: #collecting names
            art_name.append(df[df.index == art_id]['title'].values[0])   
    
    end = time()

    if verbose:
        print('elapsed time: {:.6f}s'.format(end-start))
    
    return art_name #articles names for articles IDs
    
#########1#########2#########3#########4#########5#########6#########7#########8
def fn_get_top_article(df,
                       num_art=10,
                       as_index=True,
                       verbose=False):
    '''This function takes a dataframe of user vs articles interactions and
    return the titles of the most accessed ones. 

    The code here presented is strongly based on the Udacity Jupyter Notebook
    Recommendations wit IBM, and is part of the learning exercise, necessary to
    complete the third project of Data Scientist course. It was conceived as a
    learning guide for students who wants to go deeper in Python/Pandas.
    
    Inputs:
      - df (mandatory) -  is a kind of df_inter_enc (intermediary encoded)
        dataframe type (Pandas Dataframe)
      - num_art - (optional) - the number of most accessed articles in the 
        dataframe (Integer, default=10)
      - as_index - if you want to return it as an Pandas Index, instead of a 
        list (default=True)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - top_article - (list) A list of the top 'n' article titles 
    '''
    if verbose:
        print('###function get top articles started')  

    start = time()
    most_art = df['title'].value_counts()[:num_art].index

    if not as_index:
        most_art = most_art.tolist()

    end = time()

    if verbose:
        print('elapsed time: {:.6f}s'.format(end-start))
    
    return most_art #article titles only
    
#########1#########2#########3#########4#########5#########6#########7#########8
def fn_get_top_article_id(df, 
                          num_art=10,
                          as_index=True,
                          verbose=False):
    '''This function takes a dataframe of user vs articles interactions and
    return the titles of the most accessed ones. Old version.

    The code here presented is strongly based on the Udacity Jupyter Notebook
    Recommendations wit IBM, and is part of the learning exercise, necessary to
    complete the third project of Data Scientist course. It was conceived as a
    learning guide for students who wants to go deeper in Python/Pandas.
    
    Inputs:
      - df (mandatory) -  is a kind of df_inter_enc (intermediary encoded)
        dataframe type (Pandas Dataframe)
      - num_art - (optional) - the number of most accessed articles in the 
        dataframe (Integer, default=10)
      - as_index - if you want to return it as an Pandas Index, instead of a 
        list (default=True)   
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - top_article - (list) A list of the top 'n' article titles
    '''
    if verbose:
        print('###function get top articles started')        
 
    start = time()
    most_art = df.index.value_counts()[:num_art].index

    if not as_index:
        most_art = most_art.tolist()

    end = time()

    if verbose:
        print('elapsed time: {:.6f}s'.format(end-start))
 
    return most_art #most accessed articles ids

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_get_top_article_id2(df,
                          max_art=10,
                          special_out=False,
                          verbose=False):
    '''This function provides a list with the most accessed articles.

    The code here presented is strongly based on the Udacity Jupyter Notebook
    Recommendations wit IBM, and is part of the learning exercise, necessary to
    complete the third project of Data Scientist course. It was conceived as a
    learning guide for students who wants to go deeper in Python/Pandas.

    Inputs:
      - df (mandatory) - dataframe containing the articles IDs (m) vs users IDs 
        (n) in a way as df_inter_enc (Pandas Dataframe)
      - max_art (optional) - maximum number of articles (Integer, default=10)
      - special_out (optional) - if you want in the required format, please set
        it as True (default=False)
      - verbose (optional) - if you want some verbosity (default=False)
    Output:
      - a list of strings in format '9999.0' (required for validation)
    '''
    if verbose:
        print('###function get top articles started')

    article_id = pd.Series(df.index.value_counts()[:max_art].index)

    if special_out:
        article_id = article_id.apply(lambda x: str(float(x)))

    return list(article_id)
    
#########1#########2#########3#########4#########5#########6#########7#########8
def fn_get_top_sorted_user(df_user_item,
                           df_inter_enc,
                           user_id, 
                           verbose=False):
    '''This function takes two dataframes and creates, for one user, a list of
    most similar users by two criteria.

    The code here presented is strongly based on the Udacity Jupyter Notebook
    Recommendations wit IBM, and is part of the learning exercise, necessary to
    complete the third project of Data Scientist course. It was conceived as a
    learning guide for students who wants to go deeper in Python/Pandas.

    Inputs:
      - df_user_item (mandatory) - a dataframe in the shape users vs items
        (articles) - (Pandas Dataframe)
      - df_inter_enc (mandatory) - a dataframe connecting user IDs and items 
        IDs in a way that it looks like a intermediary dataframe (n-users 
        viewed m-items) - (Pandas Dataframe)
      - user_id (mandatory) - (integer/string) - an User ID for being our
        original User, from the viewpoint of our search
      - verbose (optional) - if you want some verbosity during the processing,
        please turn it on
    Output:
      - a Pandas dataframe, sorted by two different criteria for choosing a
        nearest user for it
      - a index, containing only User Ids from the top similars
    '''
    if verbose:
        print('###function get top sorted users started')
        
    start = time()
    user_id = str(user_id) #ensure that ID will be a numerical string
    
    #first step, creating dataframe
    #extracting self vector
    main_user = df_user_item.loc[user_id]
    #making the dot product
    series2 = df_user_item.dot(main_user)
    #making access counts
    series3 = df_inter_enc.user_id.value_counts()
    #concatenating both series
    df_closer = pd.concat([series2, series3], axis=1)
    df_closer.columns = ['cartesian_similarity', 'access_count']
    df_closer.index.name = 'user_id'
    if verbose:
        print('*rows for the new dataset: {}'.format(df_closer.shape[0]))

    #second step, removing self
    df_closer = df_closer[df_closer.index != user_id]
    if verbose:
        print('*rows after self removal: {}'.format(df_closer.shape[0]))

    #third step, sorting & embellishing
    df_closer = df_closer.sort_values(
                by=['cartesian_similarity', 'access_count'], 
                ascending=False
    )
    df_closer.columns.name = 'user=' + user_id
    end = time()
    
    if verbose:
        print('elapsed time: {:.4f}s'.format(end-start))

    return df_closer, df_closer.index #closer users, by two criteria, 2 ways
    
#########1#########2#########3#########4#########5#########6#########7#########8
def fn_get_user_article(df_user_item,
                        df, #df_inter_enc
                        user_id, 
                        max_art=10,
                        id_as_str=False,
                        top_sort=False,
                        verbose=False):
    '''This function gives a list of article_ids and article titles that have 
    been seen by a user.

    New add: top_sort function added, for access count. Original behavior was
    not affected.
    
    The code here presented is strongly based on the Udacity Jupyter Notebook
    Recommendations wit IBM, and is part of the learning exercise, necessary to
    complete the third project of Data Scientist course. It was conceived as a
    learning guide for students who wants to go deeper in Python/Pandas.

    Inputs:
      - df_user_item (mandatory) - dataframe of users by articles 
        it have a 1 when an article was viewed by an user (Pandas Dataframe)
       - df (mandatory) - an article detailing dataframe (normally df_inter_enc) 
        (Panda Dataframe)       
      - user_id (mandatory) - an user id (Integer)
      - max_art (optional) - maximum number of entries for output (default=None)
      - id_as_str (optional) - if you want the list of the articles IDs in 
        string format (Boolean, default=False)
      - top_sort (optional) - if you want articles sorted by their access count
        (Booleal, default=False)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Outputs:
      - ls_id - (list) a Python list of the article ids accessed by the user
      - ls_name - (list) a list of the names of the articles 
        (this is identified by the doc_full_name column in df_article)    
    '''
    if verbose:
        print('###function get user article started')
        
    start = time()
    art_cols = ['count', 'access_count']
    #locate the rellevant articles for the focused user
    user = df_user_item.loc[str(user_id)]==1
    article = df_user_item.loc[str(user_id)][user]

    if top_sort:
        if verbose:
            print('*top sort enhancement activated')

        #first, count all the accesses for each article
        article_val = df.index.value_counts()

        #second, concatenate data  
        art_concat = pd.concat(objs=[article, article_val], 
                               axis=1, 
                               join='inner') 
        art_concat.columns = art_cols
        art_concat = art_concat.drop(['count'],
                                     axis=1)
        #third, return it as a Series
        sr_article = art_concat.sort_values(by=['access_count'], 
                                            ascending=False)
    else:
        if verbose:
            print('*traditional function behavior')

        #do nothing as a better sorting effort
        sr_article = article

    #final 1, get article IDs that were rellevant to an user
    ls_id = sr_article.index.tolist()[0:max_art]

    if verbose:
        print('*return format: {} items'.format(max_art))

    #final 2, get articles names
    ls_name = fn_get_article_name(
                  df=df,
                  ls_article=ls_id,
                  alternative_df=False,
                  verbose=verbose
    )
        
    #final 3, if return as string names
    if id_as_str:
        if verbose:
            print('*returning articles IDs as strings')
        list_aux = []

        for item in ls_id:
            list_aux.append(str(item))

        ls_id = list_aux
    
    end = time()
    
    if verbose:
        print('elapsed time: {:.4f}s'.format(end-start))
    
    return ls_id, ls_name #list of ids and names
    
#########1#########2#########3#########4#########5#########6#########7#########8
def xfn_get_user_article(df_user_item,
                        df,
                        user_id, 
                        max_art=10,
                        id_as_str=False,
                        verbose=False):
    '''This function gives a list of article_ids and article titles that have 
    been seen by a user.
    
    DEPRECATED, use instead: fn_get_user_article
    
    Inputs:
      - df_user_item (mandatory) - dataframe of users by articles 
        it have a 1 when an article was viewed by an user (Pandas Dataframe)
       - df (mandatory) - an article detailing dataframe (normally df_inter_enc) 
        (Panda Dataframe)       
      - user_id (mandatory) - an user id (Integer)
      - max_art (optional) - maximum number of entries for output (default=None)
      - id_as_str (optional) - if you want the list of the articles IDs in 
        string format (Boolean, default=False) 
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Outputs:
      - ls_id - (list) a Python list of the article ids accessed by the user
      - ls_name - (list) a list of the names of the articles 
        (this is identified by the doc_full_name column in df_article)    
    '''
    if verbose:
        print('###function get user article started')
        
    start = time()

    #first, get article IDs that were rellevant to an user
    article = df_user_item.loc[str(user_id)][df_user_item.loc[str(user_id)]==1]
    ls_id = article.index.tolist()[0:max_art]

    if verbose:
        print('return format: {} items'.format(max_art))

    #second, get articles names
    ls_name = fn_get_article_name(
                  df=df,
                  ls_article=ls_id,
                  alternative_df=False,
                  verbose=verbose
    )
    
    if id_as_str:
        if verbose:
            print('returning articles IDs as strings')
        list_aux = []

        for item in ls_id:
            list_aux.append(str(item))

        ls_id = list_aux
    
    end = time()
    
    if verbose:
        print('elapsed time: {:.4f}s'.format(end-start))
    
    return ls_id, ls_name #list of ids and names

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_svd_plot(df_user_item,
                lat_start=10,
                lat_stop=710,
                lat_step=20,
                plot=False,
                verbose=False):
    '''This function works as a toolkit for plotting/validating SVD Arrays.

    The code here presented is strongly based on the Udacity Jupyter Notebook
    Recommendations wit IBM, and is part of the learning exercise, necessary to
    complete the third project of Data Scientist course. It was conceived as a
    learning guide for students who wants to go deeper in Python/Pandas.

    Inputs:
      - df_user_matrix (mandatory) - an Array in the shape of User-Item (Pandas
        Dataframe)
      - latent parameters (optional) - defining (start, stop, step) for the
        size of latent features to be measured in each step
        (default: (10,710,20))
        *if you want to simulate starting from less than 10 latent, please 
         alter lat_start parameter
        *if you want a shorter/longer, alter lat_stop
        *if you need more or less graining, ater lat_step
      - plot (optional) - if you want to plot the graph (default=False)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - a plotted graphic with the results
    '''
    start = time()

    if verbose:
        print('###function SVD plot toolkit started!')

    #step zero - preparation
    #Making the SVD on the User-Item array
    #decomposing the Array to his basic elements
    #(two Autovectors and one fundamental parameter)
    u, s, vt =  np.linalg.svd(df_user_item)

    #first, create an Array 
    #for simulating different choices on latent features
    num_latent_feat = np.arange(lat_start,
                                lat_stop,
                                lat_step) #start, stop, step (X-axis)
    sum_err = []

    for k in num_latent_feat: #test for diverse parameters
        #restructure with k latent features
        #u[:, :k] <-this guy is an Array and all his rows will be shortened to k
        #s[:k]<-this one is a vector (representing a diagonal Array), will be 
        #       chopped in both directions
        #vt[:k, :] <-this one is an Array will be chopped to k lines
        #the idea is to reduce dimensionallity
        #less explanatory equations for the model, trying to NOT map noise
        s_new, u_new, vt_new = np.diag(s[:k]), u[:, :k], vt[:k, :]    
    
        #make the dot product
        #rebuild the user_item (now a estimated one), just makind the dot
        #product between its components
        df_user_item_est = np.around(np.dot(np.dot(u_new, s_new), vt_new))
    
        #compute the error for each prediction to actual value
        #takes the user-item array and subtrat from it the estimated user-item
        #this gives us an idea of the total error that we are impetrating by 
        #using a model, instead of raw data
        diffs = np.subtract(df_user_item, df_user_item_est)
    
        #total errors and keep track of them
        #sum the modules for both dimmensions
        err = np.sum(np.sum(np.abs(diffs)))
        #append them in a list, for plotting the graph (Y-axis)
        sum_err.append(err)
      
    if plot:
        item_num = df_user_item.sum().sum() #getting the num of items
        if verbose:
            print('*plotting for {:.0f} items'.format(item_num))
        
        plt.plot(num_latent_feat, 1-np.array(sum_err)/item_num);
        plt.xlabel('Latent Features');
        plt.ylabel('Accuracy');
        plt.title('Accuracy vs Latent Features');

    end=time()
    
    print('time to process: {:.4f}s'.format(end-start))

    return True
    
#########1#########2#########3#########4#########5#########6#########7#########8
def fn_svd_latent(df_user_item,
                  ls_latent_feat,
                  validate=False,
                  df_test=None,
                  verbose=False):
    '''This function iterates latent features for SVD Arrays.

    New version: implemented Train feature. So the function was adapted in a way
    that it will not affect his original behavior. Now it is possible to plot
    graphs for model validation;

    The code here presented is strongly based on the Udacity Jupyter Notebook
    Recommendations wit IBM, and is part of the learning exercise, necessary to
    complete the third project of Data Scientist course. It was conceived as a
    learning guide for students who wants to go deeper in Python/Pandas.

    Inputs:
      - df_user_item (mandatory) - dataframe ind user-item for making the
        copycat model and compare to the original
      - ls_latent_feat (mandatory) - a list of latent features sizes to be 
        simulated (List)
      - validate (optional) - if you will be validating, set it as True -
        (Boolean, default=False) 
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Outputs:
      - sum_err - a list of summation errors for each step of the simulation
      - s_new - the intermediary diagonal matrix (for future validation, with
        new data)
    '''
    start = time()

    if verbose:
        print('###function SVD latent started')

    #Making the SVD on the User-Item array
    #decomposing the Array to his basic elements
    #(two Autovectors and one fundamental parameter in a Diagonal Array)
    u, s, vt =  np.linalg.svd(df_user_item)
    sum_err = []

    if validate:
        if verbose:
            print ('*validation behavior started')

        #step zero (preparation)
        sum_test = []
        sr_main_id = df_user_item.index #now user-item will be our Train data
        sr_test_id = df_test.index
        sr_test_art = df_test.columns.values

        comm_id = list(set(sr_main_id) & set(sr_test_id)) #users in common
        comm_col = df_user_item.columns.intersection(sr_test_art)

        main_comm_id = df_user_item.index.isin(sr_test_id)
        main_comm_col = df_user_item.columns.isin(sr_test_art)

        u_test = u[main_comm_id, :]
        vt_test= vt[:, main_comm_col]
        
        for k in ls_latent_feat: #more explanations on normal behavior    
            u_main, s_main, vt_main = u[:, :k], np.diag(s[:k]), vt[:k, :]
            #only two to chop, as Test uses main parameter learned from Train 
            u_val, vt_val = u_test[:, :k], vt_test[:k, :]

            #1.creating model estimates
            df_user_item_main = np.around(np.dot(np.dot(u_main, s_main), 
                                         vt_main))
            df_user_item_val = np.around(np.dot(np.dot(u_val, s_main), 
                                         vt_val))
            
            #2.taking the differences between actual and modellized
            diff_main = np.subtract(df_user_item, 
                                    df_user_item_main)
            diff_val = np.subtract(df_test.loc[comm_id, :],
                                   df_user_item_val) #predicted user_item

            err_main = np.sum(np.sum(np.abs(diff_main)))
            err_val = np.sum(np.sum(np.abs(diff_val)))
            #print('*err {} err test {}'.format(err_main, err_val))
            #take errors for each simulation cicle
            sum_err.append(err_main)
            sum_test.append(err_val)

        end=time()
        if verbose:
            print('time to process: {:.4f}s'.format(end-start))

        return sum_err, sum_test, comm_id

    else:
        if verbose:
            print ('*normal behavior started')

        for k in ls_latent_feat: #test for diverse parameters
            #restructure, rebuilding with k latent features
            #u[:, :k] <-this guy is an Array and rows will be shortened to k
            #s[:k]<-is a vector (representing a diagonal Array), will be 
            #       chopped in both directions
            #vt[:k, :] <-this one is an Array will be chopped to k lines
            #the idea is to reduce dimensionallity
            #less explanatory equations for the model, trying to NOT map noise
            u_new, s_new, vt_new = u[:, :k], np.diag(s[:k]), vt[:k, :]    
    
            #make the dot product between its components, rebuild user_item
            #a similar Array, following the rules of our reduced model
            df_user_item_est = np.around(np.dot(np.dot(u_new, s_new), vt_new))
    
            #compute the error for each prediction to actual value
            #takes user-item array and subtrat from it the estimated user-item
            #this gives us an idea of the total error that we are impetrating by 
            #using a model, instead of raw data
            diffs = np.subtract(df_user_item, df_user_item_est)
    
            #total errors and keep track of them
            #sum the modules for both dimmensions
            err = np.sum(np.sum(np.abs(diffs)))
            #append them in a list, for plotting the graph (Y-axis)
            sum_err.append(err)

        end=time()
        if verbose:
            print('time to process: {:.4f}s'.format(end-start))

        return sum_err

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_svd_plot (df_user_item,
                 lat_start=10,
                 lat_stop=710,
                 lat_step=20,
                 df_test=None,
                 validate=False,
                 plot=False,
                 verbose=False):
    '''This function works as a toolkit for plotting/validating SVD Arrays.

    The code here presented is strongly based on the Udacity Jupyter Notebook
    Recommendations wit IBM, and is part of the learning exercise, necessary to
    complete the third project of Data Scientist course. It was conceived as a
    learning guide for students who wants to go deeper in Python/Pandas.

    Inputs:
      - df_user_item (mandatory) - an Array in the shape of User-Item (Pandas
        Dataframe)
      - latent parameters (optional) - defining (start, stop, step) for the
        size of latent features to be measured in each step
        (default: (10,710,20))
        *if you want to simulate starting from less than 10 latent, please 
         alter lat_start parameter
        *if you want a shorter/longer, alter lat_stop
        *if you need more or less graining, ater lat_step
      - validate (optional) - if you are doing a validation, please, inform
        the second dataframe (df_test) - (default=False)
      - plot (optional) - if you want to plot the graph (default=False)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - a plotted graphic with the results
    '''
    start = time()

    if verbose:
        print('###function SVD plot toolkit started')

    #step zero - preparation    
    #for simulating different choices on latent features
    ls_latent_feat = np.arange(lat_start,
                               lat_stop,
                               lat_step) #start, stop, step (X-axis)
    if validate:
        sum_err, sum_test, comm_id = fn_svd_latent(
                                          df_user_item,
                                          ls_latent_feat=ls_latent_feat,
                                          validate=True,
                                          df_test=df_test,
                                          verbose=verbose
    )
    else:    
        sum_err = fn_svd_latent(
                      df_user_item,
                      ls_latent_feat=ls_latent_feat,
                      validate=False,
                      verbose=verbose
    )
    if plot:
        #determination of the normalization factior for only one curve
        item_num = df_user_item.sum().sum() #get the number of valid items
      
        if validate:
            #determination of the normalization factor for both curves
            #this part is very tricky, because I need to normallize the sum of
            #errors by ne number of elements. And what is that number?
            #so, for train, I took the total number of users and multiplied it
            #by a factor that says the total number of 
            #item_num_add = df_test.sum().sum() #complete the number of valids
            #tot_item = item_num + item_num_add #new valid items total
            #tot_row = df_user_item.shape[0] + df_test.shape[0] #total users
            #item_num = df_user_item.shape[0] * (tot_item/tot_row)
            #this part is really tricky, as we are dealing with only a group of
            #selected, most active users (20 with this dataset). So, how to find
            #a reasonable metric?
            #my old metric don´t gave me a fair result, let´s try another idea 
            #item_num_test = df_test.shape[0] * (tot_item/tot_row)
            #the idea is that the population from Test cames from only 20 Users
            #and what is the population? The total of items consumed from our
            #original user_item dataframe, so:
            #item1 = df_user_item.loc[comm_id].sum().sum()
            #item2 = df_user_item.loc[comm_id].sum().sum()
            #item_num_test = item1 + item2
            #Now, a new metric, takes the total users, and the total items
            #then take users for training and divide by total users
            #and the same thing for testing
            #finally multiplices the total item by the factor for user/train
            #and users/test
            tot_usr = df_user_item.shape[0] + df_test.shape[0]
            tot_item = df_user_item.sum().sum() + df_test.sum().sum()
            train_rel = df_user_item.shape[0] / tot_usr
            test_rel = df_test.shape[0] / tot_usr
            item_num = tot_item * train_rel
            #considering that only 20% of these parameters have some
            #explanatory power (this number is a bit arbitrary and
            #perhaps needs a later revision)
            item_num_test = (tot_item * test_rel) * 0.20
        
        sim_val = 1 - np.array(sum_err) / item_num #simulated values

        if verbose:
            print('*plotting for {:.0f} items'.format(item_num))

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(ls_latent_feat, 
                sim_val, 
                color='blue', 
                label='train')

        if validate:
            if verbose:
                print('*plotting test for {:.0f} items'.format(item_num_test))

            sim_val2 = 1 - np.array(sum_test) / item_num_test
            ax.plot(ls_latent_feat, 
                    sim_val2, 
                    color='red',
                    label='test')
            ax.vlines(x=245, 
                      ymin=0.35, 
                      ymax=1., 
                      colors='green', 
                      linestyles='dashed', 
                      label='best fitting')

        ax.legend(loc='best')
        ax.set_xlabel('Latent Features')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy vs Latent Features');
 
    end=time()
    print('time to process: {:.4f}s'.format(end-start))

    return True
    
#########1#########2#########3#########4#########5#########6#########7#########8
def Xfn_svd_latent(df_user_item,
                  ls_latent_feat,
                  validate=False,
                  df_test=None,
                  verbose=False):
    '''This function iterates latent features for SVD Arrays.
    
    DEPRECATED

    New version: implemented Train feature. So the function was adapted in a way
    that it will not affect his original behavior. Now it is possible to plot
    graphs for model validation;

    The code here presented is strongly based on the Udacity Jupyter Notebook
    Recommendations wit IBM, and is part of the learning exercise, necessary to
    complete the third project of Data Scientist course. It was conceived as a
    learning guide for students who wants to go deeper in Python/Pandas.

    Inputs:
      - df_user_item (mandatory) - dataframe ind user-item for making the
        copycat model and compare to the original
      - ls_latent_feat (mandatory) - a list of latent features sizes to be 
        simulated (List)
      - validate (optional) - if you will be validating, set it as True -
        (Boolean, default=False) 
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Outputs:
      - sum_err - a list of summation errors for each step of the simulation
      - s_new - the intermediary diagonal matrix (for future validation, with
        new data)
    '''
    start = time()

    if verbose:
        print('###function SVD latent started')

    #Making the SVD on the User-Item array
    #decomposing the Array to his basic elements
    #(two Autovectors and one fundamental parameter)
    u, s, vt =  np.linalg.svd(df_user_item)
    sum_err_main = []

    if validate:
        if verbose:
            print ('*validation behavior started')

        #step zero (preparation)
        sum_err_main = []
        sr_main_id = df_user_item.index #now user-item will be our Train data
        sr_test_id = df_test.index
        sr_test_art = df_test.columns.values

        comm_id = list(set(sr_main_id) & set(sr_test_id)) #users in common
        comm_col = df_user_item.columns.intersection(sr_test_art)

        main_comm_id = df_user_item.index.isin(sr_test_id)
        main_comm_col = df_user_item.columns.isin(sr_test_art)

        u_test = u[main_comm_id, :]
        vt_test= vt[:, main_comm_col]
        
        for k in ls_latent_feat: #more explanations on normal behavior
            u_main, s_main, vt_main = u[:, :k], np.diag(s[:k]), vt[:k, :]
            #only two to chop, as Test uses main parameter learned from Train 
            u_test, vt_test = u_test[:, :k], vt_test[:k,:]

            #1.creating model estimates
            df_user_item_main = np.around(np.dot(np.dot(u_main, s_main), 
                                         vt_main))
            df_user_item_test = np.around(np.dot(np.dot(u_test, s_main), 
                                         vt_test))
            
            #2.taking the differences between actual and modellized
            diff_main = np.subtract(df_user_item, 
                               df_user_item_main)
            diff_test = np.subtract(df_test.loc[comm_id, :],
                                    df_user_item_test) #predicted user_item

            err_main = np.sum(np.sum(np.abs(diff_main)))
            err_test = np.sum(np.sum(np.abs(diff_test)))

            #take errors for each simulation cicle
            sum_err_main.append(err_main)
            sum_err_test.append(err_test)

        end=time()
        if verbose:
            print('time to process: {:.4f}s'.format(end-start))

        return sum_err_main, sum_err_test

    else:
        if verbose:
            print ('*normal behavior started')

        for k in ls_latent_feat: #test for diverse parameters
            #restructure, rebuilding with k latent features
            #u[:, :k] <-this guy is an Array and rows will be shortened to k
            #s[:k]<-is a vector (representing a diagonal Array), will be 
            #       chopped in both directions
            #vt[:k, :] <-this one is an Array will be chopped to k lines
            #the idea is to reduce dimensionallity
            #less explanatory equations for the model, trying to NOT map noise
            u_new, s_new, vt_new = u[:, :k], np.diag(s[:k]), vt[:k, :]    
    
            #make the dot product between its components, rebuild user_item
            #a similar Array, following the rules of our reduced model
            df_user_item_est = np.around(np.dot(np.dot(u_new, s_new), vt_new))
    
            #compute the error for each prediction to actual value
            #takes user-item array and subtrat from it the estimated user-item
            #this gives us an idea of the total error that we are impetrating by 
            #using a model, instead of raw data
            diffs = np.subtract(df_user_item, df_user_item_est)
    
            #total errors and keep track of them
            #sum the modules for both dimmensions
            err = np.sum(np.sum(np.abs(diffs)))
            #append them in a list, for plotting the graph (Y-axis)
            sum_err_main.append(err)

        end=time()
        if verbose:
            print('time to process: {:.4f}s'.format(end-start))

        return sum_err_main
        
#########1#########2#########3#########4#########5#########6#########7#########8
def Xfn_svd_plot (df_user_item,
                 lat_start=10,
                 lat_stop=710,
                 lat_step=20,
                 validate=False,
                 df_test=None,
                 plot=False,
                 verbose=False):
    '''This function works as a toolkit for plotting/validating SVD Arrays.
    
    DEPRECATED

    The code here presented is strongly based on the Udacity Jupyter Notebook
    Recommendations wit IBM, and is part of the learning exercise, necessary to
    complete the third project of Data Scientist course. It was conceived as a
    learning guide for students who wants to go deeper in Python/Pandas.

    Inputs:
      - df_user_item (mandatory) - an Array in the shape of User-Item (Pandas
        Dataframe)
      - latent parameters (optional) - defining (start, stop, step) for the
        size of latent features to be measured in each step
        (default: (10,710,20))
        *if you want to simulate starting from less than 10 latent, please 
         alter lat_start parameter
        *if you want a shorter/longer, alter lat_stop
        *if you need more or less graining, ater lat_step
      - validate (optional) - if you are doing a validation, please, inform
        the second dataframe (df_test) - (default=False)
      - plot (optional) - if you want to plot the graph (default=False)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - a plotted graphic with the results
    '''
    start = time()

    if verbose:
        print('###function SVD plot toolkit started')

    #step zero - preparation    
    #for simulating different choices on latent features
    ls_latent_feat = np.arange(lat_start,
                               lat_stop,
                               lat_step) #start, stop, step (X-axis)
    if validate:
        sum_err, sum_test = fn_svd_latent(
                                df_user_item,
                                ls_latent_feat=ls_latent_feat,
                                validate=True,
                                df_test=df_test,
                                verbose=verbose)
    else:    
        sum_err = fn_svd_latent(
                      df_user_item,
                      ls_latent_feat=ls_latent_feat,
                      validate=False,
                      verbose=verbose)
    
    if plot:
        item_num = df_user_item.sum().sum() #get the number of items
        sim_val = 1 - np.array(sum_err) / item_num #simulated values

        if verbose:
            print('*plotting for {:.0f} items'.format(item_num))

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(ls_latent_feat, 
                sim_val, 
                color='blue', 
                label='train')

        if validate:
            if verbose:
                print('*plotting for test')
            item_num2 = df_test.sum().sum()
            sim_val2 = 1 - np.array(sum_test) / item_num2
            ax.plot(ls_latent_feat, 
                    sim_val2, 
                    color='red',
                    label='test')
            ax.vlines(x=245, 
                      ymin=0, 
                      ymax=1, 
                      colors='green', 
                      linestyles='dashed', 
                      label='best fitting')

        ax.legend(loc='best')
        ax.set_xlabel('Latent Features')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy vs Latent Features');
 
    end=time()
    print('time to process: {:.4f}s'.format(end-start))

    return True
    
#########1#########2#########3#########4#########5#########6#########7#########8
def fn_user_user_rec(df_user_item,
                     df,
                     user_id,
                     max_art=None,
                     max_usr=None,
                     max_rec=10,
                     verbose=False):
    '''This function takes and user and gives recommendations of new articles 
    for him. The engine is based on cartesian (dot product) similarity.
    
    The code here presented is strongly based on the Udacity Jupyter Notebook
    Recommendations wit IBM, and is part of the learning exercise, necessary to
    complete the third project of Data Scientist course. It was conceived as a
    learning guide for students who wants to go deeper in Python/Pandas.

    It works in this way:
    1. First, I give a user-item matrix (preprocessed), and one user id;
    2. Then, the system goes into the user-item matrix and retrieves all the
       articles viewed by this user and stores it (we don´t want to recommend as
       a new article something the user has already viewed!);
    3. In sequence, the system calculates the dot product for the user-item
       matrix vs specific user vector and order it by the better value,
       decreasing and gives the list of the most similar users for this user;
    4. And then, the systems iterate into the first most similar user, takes the
       viewed articles by him and it they were not already been viewed by the
       main user, then store it for recommendation;
    5. Finally, if the recommendation list was not already complete, then the
       system goes into de second most similar user and so on.

    If the possibilities exausted and we had not found a complete recommendation
    list, we just give up and give an incomplete (or in the worst case, a empty
    list for recommendations for this user, based on users similarity).    
    
    Inputs:
      - df_user_item (mandatory) - a previously processed dataframe of users by 
        articles (Pandas Dataframe)
        *it have an 1 when the article was viewed by an user (Pandas Dataframe)
       - user_id (mandatory) - an user id (Integer/String)
       - df (mandatory) - an article detailing dataframe (normally df_inter_enc) 
        (Panda Dataframe)       
      - max_art (optional) - maximum number of articles for each other user
        (one user per time) - (default=None)
      - max_usr (optional) - maximum number of user for each other user
        (one user per time) - (default=None)
      - num_rec (optional) - the maximum number of recommendations you want for 
        the main user - (Integer, default=10)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - rec - (list) a list of recommendations for the user
      
    Description:
      Loops through the users based on closeness to the input user_id
      For each user - finds articles the user hasn't seen before and provides them as recs
      Does this until m recommendations are found
    Notes:
      Users who are the same closeness are chosen arbitrarily as the 'next' user
      For the user where the number of recommended articles starts below m 
      and ends exceeding m, the last items are chosen arbitrarily
    '''
    if verbose:
        print('###function user to user recommendations started')        

    start1 = time()
    rec_art = []

    #get articles viewed by the main user
    ls_art_view_id, ls_art_view_name = fn_get_user_article(
                                           df_user_item=df_user_item,
                                           df=df,
                                           user_id=user_id, 
                                           max_art=max_art,
                                           verbose=verbose
    )
    ls_sim_user = fn_find_similar_user(
                      df_user_item=df_user_item,
                      user_id=user_id,
                      max_usr=max_usr,
                      verbose=verbose
    )
    #iterating on new users
    for user in ls_sim_user:
        if verbose:
            print('*searching new articles from user', user)
        ls_sim_art_id, ls_sim_art_name = fn_get_user_article(
                                           df_user_item=df_user_item,
                                           df=df,
                                           user_id=user, 
                                           max_art=max_art,
                                           verbose=verbose
        )
        if verbose:
            print('retrieved articles:', ls_sim_art_id)
        #getting different articles from the new user
        #rec_new = np.setdiff1d(
        #              ls_sim_art_id, 
        #              ls_art_view_id, 
        #              assume_unique=True)
        rec_new = list(set(ls_sim_art_id) ^set(ls_art_view_id))
        if verbose:
            print('new arts:', rec_new)
        #adding new recommendations, if they exist    
        if len(rec_new) > 0:
            more = max_rec - len(rec_art)
            rec_art = rec_art + rec_new[:more]
            if verbose:
                print('*adding {} new recommendations from user {}'\
                      .format(more, user))
            #testing for earlier break
            if len(rec_art) == max_rec:
                if verbose:
                    print('*earlier break condition reached')
                break #earlier break condition
            elif len(rec_art) > max_rec:
                raise Exception('something went wrong with recommendations')
    
    end = time()

    if verbose:
        print('elapsed time: {:.6f}s'.format(end-start1))
    
    return rec_art #articles recommendation for this user_id
    
#########1#########2#########3#########4#########5#########6#########7#########8
def fn_user_user_rec_part2(df_user_item,
                           df,#df_inter_enc
                           user_id,
                           max_art=None,
                           max_usr=None,
                           max_rec=10,
                           verbose=False):
    '''This function takes and user and gives recommendations of new articles 
    for him. The engine is based on cartesian (dot product) similarity.

    The code here presented is strongly based on the Udacity Jupyter Notebook
    Recommendations wit IBM, and is part of the learning exercise, necessary to
    complete the third project of Data Scientist course. It was conceived as a
    learning guide for students who wants to go deeper in Python/Pandas.

    Enhancements over old fn_user_user_rec:
    1. Users now have two selection criteria
       - the first is cartesian product, for similarity (main)
       - the second is by users that viewed more articles (secondary)
    2. Articles now have a sorting enhancement:
       - function fn_get_user_article was modified, for sorting by most viewed
         articles before giving the recommendation
    
    It works in this way:
    1. First, I give a user-item matrix (preprocessed), and one user id;
    2. Then, the system goes into the user-item matrix and retrieves all the
       articles viewed by this user and stores it (we don´t want to recommend as
       a new article something the user has already viewed!);
    3. In sequence, the system calculates the dot product for the user-item
       matrix vs specific user vector and order it by the better value,
       decreasing and gives the list of the most similar users for this user;
    4. And then, the systems iterate into the first most similar user, takes the
       viewed articles by him and it they were not already been viewed by the
       main user, then store it for recommendation;
    5. Finally, if the recommendation list was not already complete, then the
       system goes into de second most similar user and so on.

    If the possibilities exausted and we had not found a complete recommendation
    list, we just give up and give an incomplete (or in the worst case, a empty
    list for recommendations for this user, based on users similarity).    
    
    Inputs:
      - df_user_item (mandatory) - a previously processed dataframe of users by 
        articles (Pandas Dataframe)
        *it have an 1 when the article was viewed by an user (Pandas Dataframe)
       - user_id (mandatory) - an user id (Integer/String)
       - df (mandatory) - an article detailing dataframe (normally df_inter_enc) 
        (Panda Dataframe)       
      - max_art (optional) - maximum number of articles for each other user
        (one user per time) - (default=None)
      - max_usr (optional) - maximum number of user for each other user
        (one user per time) - (default=None)
      - num_rec (optional) - the maximum number of recommendations you want for 
        the main user - (Integer, default=10)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - rec_art - (list) a list of articles IDs recommendations for the user
      - rec_name - (list) the names of the articles, extracted from df dataframe
      
    Description:
      Loops through the users based on closeness to the input user_id
      For each user - finds articles the user hasn't seen before and provides 
      them as recs
      Does this until m recommendations are found
    Notes:
      Users who are the same closeness are chosen arbitrarily as the 'next' user
      For the user where the number of recommended articles starts below m 
      and ends exceeding max_rec, the last items are chosen arbitrarily
    '''
    if verbose:
        print('###function user to user recommendations started')        

    start1 = time()
    #for user in similar_users:
    #    similar_article_ids, similar_article_names = get_user_articles(user)
    #    recommendations = np.setdiff1d(similar_article_ids, viewed_article_ids, assume_unique=True)
    #    for rec in recommendations:
    #        if len(recs) < m:
    #            recs.append(rec)
    #        else:
    #            break
    #rec_names = get_article_names(recs)
    #output = rec, rec_name

    rec_art = []

    #get articles viewed by the main user
    #using traditional calling, as we don´t need these ones sorted!
    ls_art_view_id, ls_art_view_name = fn_get_user_article(
                                           df_user_item=df_user_item,
                                           df=df, #df_inter_enc
                                           user_id=user_id, 
                                           max_art=max_art,
                                           top_sort=False, #not sorting!
                                           verbose=verbose
    )
    #new fancy function to do this job
    top_user = fn_get_top_sorted_user(
                   df_user_item=df_user_item,
                   df_inter_enc=df, #df_inter_enc
                   user_id=user_id,
                   verbose=verbose
    )
    ls_sim_user = list(top_user[1])

    #iterating on new users
    for user in ls_sim_user:
        if verbose:
            print('*searching new articles from user', user)
        #using the function enhancement, for sorting articles too!
        ls_sim_art_id, ls_sim_art_name = fn_get_user_article(
                                           df_user_item=df_user_item,
                                           df=df, #df_inter_enc
                                           user_id=user, 
                                           max_art=max_art,
                                           top_sort=True, #sorting!
                                           verbose=verbose
        )
        if verbose:
            print('retrieved articles:', ls_sim_art_id)
        #getting different articles from the new user
        #rec_new = np.setdiff1d(        #using numpy was not so practical
        #              ls_sim_art_id,   #I will try it later!
        #              ls_art_view_id, 
        #              assume_unique=True)
        rec_new = list(set(ls_sim_art_id) ^set(ls_art_view_id))
        if verbose:
            print('new arts:', rec_new)
        #adding new recommendations, if they exist    
        if len(rec_new) > 0:
            more = max_rec - len(rec_art)
            rec_art = rec_art + rec_new[:more]
            if verbose:
                print('*adding {} new recommendations from user {}'\
                      .format(more, user))
            #testing for earlier break
            if len(rec_art) == max_rec:
                if verbose:
                    print('*earlier break condition reached')
                break #earlier break condition
            elif len(rec_art) > max_rec:
                raise Exception('something went wrong with recommendations')
    
    rec_name = fn_get_article_name(
                  df=df, #df_inter_enc
                  ls_article=rec_art,
                  alternative_df=False,
                  verbose=verbose
    )              
    end = time()

    if verbose:
        print('elapsed time: {:.6f}s'.format(end-start1))
    
    return rec_art, rec_name #articles recommendation for this user_id

###Recomendation Engines - Collaborative Filter#################################
#########1#########2#########3#########4#########5#########6#########7#########8
def fn_all_recommendation_collab(df_dist,
                                 df_user_movie,
                                 df_movie,
                                 num_rec=10,
                                 limit=100,
                                 min_rating=7,
                                 sort=False, 
                                 verbose=False):
    '''This function creates a dictionnary for possible recommendations for
    a set of users, preprocessed at df_dist dataset.
    
    Source: Udacity Data Science Course - Lesson 6  - Identifying Reccomendations
    Forth Notebook - Class 17 - Collaborative Filtering -Recommendations with 
    MovieTweetings - Collaborative Filtering
    
    Input:
      - df_dist (mandatory) - a preprocessed dataset wit the euclidian distancies
        between two users (Pandas dataset)
      - df_user_movie (mandatory) - dataset in the shape user by movie -
        (Pandas dataset)
      - df_movie (mandatory) - dataset in the shape for movies - 
        (Pandas dataset)
      - num_rec (optional) - (int) number of recommended movies to return
      - limit (optional) - extra parameter for fn_find_closest_neighbor - 
        it limits the number of neighbors (normally 100 is more than enough) 
      - min_rating (optional) - extra parameter for fn_movie_liked2() - it is
        the worst score for considering a movie as liked (normally rate 7 is 
        enough)
      - sort (optional) - extra parameter for fn_movie_liked2() - if you want
        to show the best rated movies first (in this algorithm it is useless)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - all_recs - a dictionary where each key is a user_id and the value is 
        an array of recommended movie titles
    '''
    if verbose:
        print('###function all_recommendations started')

    begin = time()

    #take all the unique users from df_dist dataset
    user_all = np.unique(df_dist['user1'])
    n_user_all = len(user_all)
    
    if verbose:
        print('*taken {} users to find recommendations'.format(n_user_all))
    
    #create a dictionnary for keeping all recommendations for each user
    all_rec = dict()
    
    #iterate users calling the function for recommendations
    for user in user_all:
        if verbose:
            print('*making recommendations for user', user)
            
        filt_dist=df_dist[df_dist['user1'] == user]
        all_rec[user] = udacourse3.fn_make_recommendation_collab(
                            filt_dist=filt_dist,
                            df_user_movie=df_user_movie,
                            df_movie=df_movie,
                            num_rec=num_rec,
                            limit=limit,
                            min_rating=min_rating,
                            sort=sort,
                            verbose=verbose)       
    end = time()
    
    if verbose:
        print('elapsed time: {:.4f}s'.format(end-begin))
    
    return all_rec

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_calculate_distance(x, 
                          y,
                          dist_type=None,
                          verbose=False):
    '''This function calculates the euclidian distance between two points,
    considering that X and Y are gived as fractions of distance.
    
    Source: Udacity Data Science Course - Lesson 6 - Recomendation Engines - 
    Class 14 - Third Notebook - More Personalized Ways - Collaborative Filtering
    & Content Based - Measuring Similarity

    Inputs:
      - x (mandatory) - an array of matching length to array y
      - y (mandatory - an array of matching length to array x
      - dist_type (mandatory) - if none is informed, returns False and nothing
        is calculated:
        * 'euclidean' - calculates the euclidean distance (you pass through
          squares or edifices, as Superman)
        * 'manhattan' - calculates the manhattan distance (so you need to turn
          squares and yoy are not Superman!)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - distance - the euclidean distance between X and Y.
    '''
    if verbose:
        print('###function calculate distances started')
        
    begin = time()
    
    if dist_type == None:
        if verbose:
            print('nothing was calculated, type was not informed')
        return False
        
    elif dist_type == 'euclidean':
        if verbose:
            print('calculating euclidian distance')
        distance = np.linalg.norm(x - y)
        
    elif dist_type == 'manhattan':
        if verbose:
            print('calculating manhattan distance')
        distance = sum(abs(e - s) for s, e in zip(x, y))
    
    end = time()
    
    if verbose:
        print('{} distance: {}'.format(dist_type, distance))
        print('elapsed time: {:.4f}s'.format(end-begin))
    
    return distance

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_compute_correlation(x, 
                           y,
                           corr_type=None,
                           verbose=False):
    '''This function calculates correlation between two variables  A negative 
    value means an inverse correlation. Nearer to 1 or -1, stronger is the 
    correlation. More  data you have, lower tends to be the error. Two points 
    give us a 1 or -1 value.
    No correlation = 0 (a straight line). p-value is important too. A small
    p-value means greater confidence in our guess.
    
    Source: Udacity Data Science Course - Lesson 6 - Recomendation Engines - 
    Class 14 - Third Notebook - More Personalized Ways - Collaborative Filtering
    & Content Based - Measuring Similarity

    Inputs:
      - x (mandatory) - an array of matching length to array y (numpy Array)
      - y (mandatory) - an array of matching length to array x (numpy Array)
      - corr_ype (mandatory) - {'kendall_tau', 'pearson', 'spearman'}
        (default: None)
        * 'kendall_tau' - Kendall´s Tau correlation coefficient. It uses a more
          advancet technique and offers less variability on larger datasets.
          It is not so computional efficient, as it runs on O(n^2).
        * 'pearson' - is the most basic correlation coefficient. The denominator
          just normalizes the raw coorelation value. Even for quadratic it keeps
          suggesting a relationship.
        * 'spearman' - Spearman Correlation can deal easily with outliers, as 
          it uses ranked data. It don´t follow normal distribution, or binomial, 
          and is an example of non-parametric function. Runs on basis O(nLog(n)).
          from Udacity course notes:
          "Spearman's correlation can have perfect relationships e.g.:(-1, 1) 
          that aren't linear. Even for quadratic it keeps
          suggesting a relationship.
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output
      - correlation_spearman - the Spearman correlation coefficient for comparing 
        x and y.values from (-1 to zero to +1).        
    '''    
    if verbose:
        print('###function correlation started')
        
    begin = time()
    
    if corr_type is None:
        if verbose:
            print('no parameter calculated, you need to inform type')
        return False
    
    elif corr_type == 'intersection':
        #transform series into datafames
        df1 = x.to_frame()
        df2 = y.to_frame()
        correlation = pd.concat([df1, df2], axis=1).corr().iloc[0,1]
    
    elif corr_type == 'kendall_tau':
        #rank both data vectors
        x = x.rank()
        y = y.rank()        
        correlation = fn_onsquare(x, y, verbose=verbose)
    
    elif corr_type == 'pearson':
        correlation = fn_ologn(x, y, verbose=verbose)
        
    elif corr_type == 'spearman': 
        #rank both data vectors
        x = x.rank()
        y = y.rank()
        correlation = fn_ologn(x, y, verbose=verbose)
        
    else:
        if verbose:
            print('invalid parameter')
        return False
            
    end = time()
    
    if verbose:
        print('{} correlation: {:.4f}'.format(corr_type, correlation))
        print('elapsed time: {:4f}s'.format(end-begin))
                            
    return correlation 

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_create_ranked_df(df_movie, 
                        df_review,
                        verbose=True):
    '''This function creates a ranked movies dataframe, that are sorted by 
    highest avg rating, more reviews, then time, and must have more than 4 
    ratings. Laterly this function can be forked for other purposes.
    
    Source: Udacity Data Science Course - Lesson 6 - Recomendation Engines - 
    Class 5 - First Notebook - Intro to Recommendation data - Part I - Finding
    Most Popular Movies.
    
    Inputs:
      - df_movie (mandatory) - the movies dataframe
      - df_review (mandatory) - the reviews dataframe
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - df_ranked_movie - a dataframe with movies 
    '''
    if verbose:
        print('###function create_ranked_df started')
        
    begin = time()
    
    #for review - take mean, count, last rating
    movie_rating = df_review.groupby('movie_id')['rating']
    avg_rating = movie_rating.mean()
    num_rating = movie_rating.count()
    last_rating = pd.DataFrame(df_review.groupby('movie_id').max()['date'])
    last_rating.columns = ['last_rating']

    #create a new dataset for dates
    df_rating_count = pd.DataFrame({'avg_rating': avg_rating, 
                                    'num_rating': num_rating})
    df_rating_count = df_rating_count.join(last_rating)

    #turn them only one dataset
    df_movie_rec = df_movie.set_index('movie_id').join(df_rating_count)

    #rank movies by the average rating, and then by the rating counting
    df_ranked_movie = df_movie_rec.sort_values(['avg_rating', 'num_rating', 'last_rating'], 
                                               ascending=False)

    #for the border of our list, get at least movies with more than 4 reviews
    df_ranked_movie = df_ranked_movie[df_ranked_movie['num_rating'] >= 5]
    
    end = time()
    
    if verbose:
        print('elapsed time: {:.4f}s'.format(end-begin))
        
    return df_ranked_movie

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_create_train_test(df_review, 
                         order_by, 
                         train_size, 
                         test_size,
                         verbose=False):
    '''This function creates a train test for our FunkSVD system.
    
    Source: Udacity Data Science Course - Lesson 7  - Matrix Factorization for
    Recommendations - Third Notebook - Class 18 - How are we doing w/ FunkSVD
    
    Inputs:
      - review (mandatory) - (pandas df) dataframe to split into train and test
      - order_by (mandatory) - (string) column name to sort by
      - train_size (mandatory) - (int) number of rows in training set
      - test_size (mandatory) - (int) number of columns in the test set
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Outputs:
      - df_train - (pandas df) dataframe of the training set
      - df_validate - (pandas df) dataframe of the test set
    '''
    if verbose:
        print('###function create_train_test started')
    
    begin = time()
    df_review_new = df_review.sort_values(order_by)
    df_train = df_review_new.head(train_size)
    df_validate = df_review_new.iloc[train_size:train_size+test_size]
    
    output = df_train, df_validate
    end = time()
    
    if verbose:
        print('elapsed time: {:.4f}s'.format(end-begin))
    
    return output

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_create_user_movie_dict(df_user_movie,
                              lower_filter=None,
                              verbose=False):
    '''This function creates a dictionnary structure based on an array of movies
    watched by a user.
    
    Source: Udacity Data Science Course - Lesson 6  - Identifying Reccomendations
    Forth Notebook - Class 17 - Collaborative Filtering -Recommendations with 
    MovieTweetings - Collaborative Filtering    
    
    Input:
      - df_user_movie (mandatory) - Pandas dataset with all movies vs users at
        the type user by movie, to be processed
        or a dictionnary to be filtered (if it is a dict, please informa the
        lower_filter parameter)
      - lower_filter (mandatory) - elliminate users with a number of watched
        videos below the number (Integer, default=None)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output: 
      - movies_seen - a dictionary where each key is a user_id and the value is 
      an array of movie_ids. Creates the movies_seen dictionary
    '''
    if verbose:
        print('###function create user movie started')
        
    begin = time()
    dic_movie_registered = dict()
    
    if type(df_user_movie) == dict: #dictionnary already processed
        if lower_filter == None or lower_filter < 1:
            if verbose:
                print('run as post processing without a lower value don´t make sense')
            return False
        else:
            if verbose:
                print('running as post processing filter, for lower:', lower_filter)
            for user, movie in df_user_movie.items():
                if len(movie) > lower_filter:
                    dic_movie_registered[user] = movie
    else:
        if verbose:
            print('running as original dictionnary builder, for lower:', lower_filter)
        num_user = df_user_movie.shape[0]
        #creating a list of movies for each user
        for i_user in range(1, num_user+1): #added lower_filter param
            content = fn_movie_watched(df_user_movie=df_user_movie,
                                       user_id=i_user,
                                       lower_filter=lower_filter,
                                       verbose=verbose)
            if content is not None:
                dic_movie_registered[i_user] = content #if there is some movie
            
    end = time()
    
    if verbose:
        print('elapsed time: {:.1f}s'.format(end-begin))
    
    return dic_movie_registered

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_create_user_movie(df_user_item, 
                         verbose=False):
    '''This function creates user by movie matrix. As it is normally a big data,
    please make a pre-filter on rewiew dataset, using:
    user_item = review[['user_id', 'movie_id', 'rating']]
    
    Source: Udacity Data Science Course - Lesson 6  - Identifying Reccomendations
    Forth Notebook - Class 17 - Collaborative Filtering - Recommendations with 
    MovieTweetings - Collaborative Filtering  
    
    Inputs:
      - df_user_item (mandatory) - a Pandas dataset containing all reviews for 
        movies made by users, in a condensed way (a thin dataset)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - df_user_by_movie -  a thick dataframe, ordered by best rated first, each 
        row representing an user and each column a movie
        (a sparse dataset, as most of the movies were never rated by each user!)
    '''
    if verbose:
        print('###function create user by movie matrix started')
        
    begin = time()
    
    #remember that...
    #df_user_item = df_review[['user_id', 'movie_id', 'rating']]
    
    df_user_by_movie = df_user_item.groupby(['user_id', 'movie_id'])['rating'].max()  
    df_user_by_movie = df_user_by_movie.unstack()
    
    end = time()
    
    if verbose:
        print('elapsed time: {:.1f}s'.format(end-begin))
    
    return df_user_by_movie

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_find_closest_neighbor(df_filt_user1,
                             limit=None,
                             verbose=True):
    '''This function takes a distance dataset and returns the closest users.
    
    Source: Udacity Data Science Course - Lesson 6  - Identifying Reccomendations
    Forth Notebook - Class 17 - Collaborative Filtering -Recommendations with 
    MovieTweetings - Collaborative Filtering

    Inputs:
      - df_filt_user1 (mandatory) - (Pandas dataset) the user_id of the individual 
        you want to find the closest users.
        e.g: filt_user1=df_dists[df_dists['user1'] == user]
      - limit (optional) - maximum number of closest users you want to return -
        (integer, default=None)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - closest_neighbor - a numpy vector containing all closest users, from
        the cloest ones until the end
    '''    
    try:
        user1 = df_filt_user1['user1'].iloc[0]
    except TypeError:
        if verbose:
            print('you must inform a filtered dataset, see documentation')
        return False
    except IndexError:
        if verbose:
            print('an empty dataset was informed, this user does not exist')
        return False

    if verbose:
        print('###function find closest neighbors started for user', user1)
        
    begin = time()
    df_filt_user2 = df_filt_user1[df_filt_user1['user2'] != user1]
    closest_user = df_filt_user2.sort_values(by='eucl_dist')['user2']
    closest_neighbor = np.array(closest_user)
    
    if limit is not None:
        closest_neighbor = closest_neighbor[:limit]
        #if verbose:
        #    print('*limit:', len(closest_neighbor))

    end = time()
    
    if verbose:
        print('returned {} neighbors'.format(len(closest_neighbor)))
        print('elapsed time: {:.4f}s'.format(end-begin))
    
    return closest_neighbor

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_find_similar_movie(df_movie,
                          dot_product,
                          movie_id,
                          cold=False,
                          verbose=False):
    '''This function takes one movie_id, find the physical position of it in
    our already processed dot product movie matrix and find the best similarity
    rate. Then search on the dot product matrix movies with the same rate, for
    their idxs (they are the most collinear from your movie-vector!). Finally,
    it retrieves the movie names, based on these idxs.
    
    It is similar to fn_find_closest_neighbor(), for Collaborative Filter.
    
    Source: Udacity Data Science Course - Lesson 6 - Ways to Reccomend
    Fifth Notebook - Class 21 - Content Based Recommendations
        
    Inputs:
      - df_movie (mandatory) - your movies dataset, including movie names
      - dot_product (mandatory) - your dot product matrix autocorrelation for 
        movies rating, with movies collinearity values. 
        You need to preprocess this earlier
      - movie_id (mandatory) - a movie id to be asked for
      - cold (optional) - if are in a cold start situation
        (Boolean, default=False)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - similar_movies - an array of the titles of the most similar movies
    '''
    if verbose:
        print('###function find similar movies started')
        
    begin = time()
    
    #retrieve the physical position (iloc) of your movie
    movie_idx = np.where(df_movie['movie_id'] == movie_id)[0][0]
    
    if verbose:
        our_movie = df_movie[df_movie['movie_id'] == movie_id]['movie']
        print('*our movie iloc: ', our_movie) #.values[0])

    if cold:
        # find the most similar movie indices - to start I said they need to be the same for all content
        similar_idx = np.where(dot_product[movie_idx] == np.max(dot_product[movie_idx]))[0]
    
        #retrieve the movie titles based on these indices
        similar_movie = np.array(df_movie.iloc[similar_idx, ]['movie'])
        
    else:        
        #get the maximum similarity rate
        max_rate = np.max(dot_product[movie_idx])
        if verbose:
            print('*max collinearity:', max_rate)
    
        #retrieve the movie indices for maximum similarity
        similar_idx = np.where(dot_product[movie_idx] == max_rate)[0]
    
        #make an array, locating by position on dataset, the similars + title
        similar_movie = fn_get_movie_name(
                            df_movie=df_movie,
                            movie_id=similar_idx,
                            by_id=False,
                            as_list=False,
                            verbose=verbose)
    end = time()
    
    if verbose:
        print('similar movies:', len(similar_movie))
        print('elapsed time: {:.4f}s'.format(end-begin))
    
    return similar_movie

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_FunkSVD(rating_mat, 
               latent_feature=4, 
               learning_rate=0.0001, 
               num_iter=100,
               verbose=False):
    '''
    This function performs matrix factorization using a basic form of FunkSVD 
    with no regularization. This version don´t have a Sigma matrix for matrix
    factorization!
    
    Source: Udacity Data Science Course - Lesson 7  - Matrix Factorization for
    Recommendations - Second Notebook - Class 15 - Implementing FunkSVD
    
    Inputs:
      - rating_mat (mandatory) - (numpy array) a matrix with users as rows,
        movies as columns, and ratings as values
      - latent_feature (optional) - (integer) the number of latent features 
        used for calculations (use this wisely, as each latent feature means
        one dimension added to your model)
      - learning_rate (optional) - (float) the learning rate used for each step
        of iteration process (it is a kind of fine-grain process)
      - num_iter (optional) - (integer) the maximum number of iterations (is a 
        kind of break method when things goes not so well)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Outputs:
      - user_mat - (numpy array) a user by latent feature matrix
      - movie_mat - (numpy array) a latent feature by movie matrix
    '''
    if verbose:
        print('###function FunkSDV started')
    
    begin = time()
    #these values will be used later, so keep them!
    num_user = rating_mat.shape[0]
    num_movie = rating_mat.shape[1]
    num_rating = np.count_nonzero(~np.isnan(rating_mat))
    if verbose:
        print('number of users:', num_user)
        print('number of movies:', num_movie)
        print('number of valid ratings:', num_rating)
    
    #start the users and movies with completelly alleatory numbers
    user_mat = np.random.rand(num_user, latent_feature)
    movie_mat = np.random.rand(latent_feature, num_movie)
    
    #Sum of Standard Errors start with zero, for our first iteration
    sse_accum = 0
    
    if verbose:
        print('Optimizaiton Statistics')
        print('Iteration Mean Squared Error')

    #main iteration loop
    for iteration in range(num_iter):
        old_sse = sse_accum #save old value
        sse_accum = 0 #new value
        
        #user-movie pairs treatment
        for i in range(num_user): #i for users
            for j in range(num_movie):#j for movies
                if rating_mat[i, j] > 0: #if there is some rating
                    #error = actual - dot prof of user and movie latents
                    diff = rating_mat[i, j] - np.dot(user_mat[i, :], movie_mat[:, j])
                    #tracking sum of sqr errors on the matrix
                    sse_accum += diff**2
                    #updating vals for each matrix, in the gradient way
                    for k in range(latent_feature):
                        user_mat[i, k] += learning_rate * (2*diff*movie_mat[k, j])
                        movie_mat[k, j] += learning_rate * (2*diff*user_mat[i, k])

        mse_val = sse_accum / num_rating
        if verbose:
            print('{:3d}       {:.6f}'.format(iteration+1, mse_val))
        
    output = user_mat, movie_mat 
    end = time()
    if verbose:
        #https://stackoverflow.com/questions/27779677/how-to-format-elapsed-time-from-seconds-to-hours-minutes-seconds-and-milliseco
        hour, remaining = divmod(end-begin, 3600)
        minute, second = divmod(remaining, 60)
        print('elapsed time: {:.4f}s ({:0>2}:{:0>2}:{:05.4f}s)'.format(end-begin, 
                                                                      int(hour),
                                                                      int(minute),
                                                                      second))        
    return output

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_get_movie_name(df_movie,
                      movie_id,
                      by_id=True,
                      as_list=True,
                      verbose=False):
    '''This function finds a movie (or a list of movies) by its index and return
    their titles as a list.
    
    There are only two ways implemented at this moment:
    
    First, it takes a single movie id and returns a single name, inside a list
    (function default)
    
    Second, it takes a list of movies idx and returns a numpy Array, with
    multiple names.
    (for providing service for fn_find_similar_movie())
    
    Source: Udacity Data Science Course - Lesson 6 - Ways to Reccomend
    Fifth Notebook - Class 21 - Content Based Recommendations
    
    Inputs:
      - df_movie (mandatory) - a preformated dataset of movies
      - movie_id (mandatory) - a list of movie_id or one single movie_id 
        (depends on the case)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - movie - a list of names of movies
    '''
    if verbose:
        print('###function get movie names started')
        
    begin = time()
    
    if as_list:
        if by_id:
            if verbose:
                print('taking movies id, returning a list')
                print('*function default')
            try:
                movie_list = list(df_movie[df_movie['movie_id'].isin(movie_id)]['movie'])
            except TypeError:
                movie_id = [movie_id]
                movie_list = list(df_movie[df_movie['movie_id'].isin(movie_id)]['movie'])
        else:
            if verbose:
                print('taking a movies idx, returning a list') #id=idx
                print('not yet implemented!')
            return False
    else:
        if by_id:
            if verbose:
                print('taking a movies id, returning numpy vector')
                print('not yet implemented!')
            return False
        else:
            if verbose:
                print('taking movies idx, returning a list') 
                print('*default for fn_find_similar_movie') #id=idx
            movie_list = np.array(df_movie.iloc[movie_id]['movie']) 
            
    end = time()
    
    if verbose:
        print('elapsed time: {:.6f}s'.format(end-begin))
   
    return movie_list

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_make_recommendation_cold(df_train_data, #train_data_df
                                df_review, #train_df
                                df_movie, #movie 
                                user_mat, #user_mat
                                movie_mat, #movie_mat
                                dot_product, #dot_prod_movie
                                _id, 
                                _id_type='movie', 
                                rec_num=5, 
                                verbose=False):
    '''This function makes recommendation, with cold start. It is the most
    complete way to make recommendations.
    
    Source: Udacity Data Science Course - Lesson 7  - Matrix Factorization for
    Recommendations - Forth Notebook - Class 20 - Cold Start Problem w/ FunkSVD
    
    Inputs:
      - df_train_data (mandatory)  - dataframe of data as user-movie matrix
      - df_train (mandatory) - dataframe of training data reviews
      - df_movie (mandatory) - movies dataframe (Pandas dataframe)
      - user_mat (mandatory) - the U matrix of matrix factorization
        (Numpy Array)
      - movie_mat (mandatory) - the V matrix of matrix factorization
        (Numpy Array)
      - dot_product (mandatory) - your dot product matrix autocorrelation for 
        movies rating, with movies collinearity values. 
        You need to preprocess this earlier
      - _id (mandatory) - either a user or movie id (integer)
      - _id_type (optional) - "movie" or "user" (string)
      - rec_num (optional) - number of recommendations to return (integer)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - rec - (array) a list or numpy array of recommended movies like the 
        given movie, or recs for a user_id given
    '''
    if verbose:
        print('###function movies watched started')
        
    begin = time()

    #if the user exists in the matrix factorization data 
    #use him and rank movies based on predicted values
    #so, for an indexed user
    val_user = df_train_data.index
    rec_id = fn_create_ranked_df(df_movie=df_movie, 
                                 df_review=df_review,
                                 verbose=verbose)
    if _id_type == 'user':
        if _id in df_train_data.index:
            #retrieve the index of which row the user is in 
            #for an user in U matrix
            idx = np.where(val_user == _id)[0][0]
            
            #make the dot product of this row and the V matrix
            pred = np.dot(user_mat[idx,:], movie_mat) #SEE 
            #Standard Error of the Estimate (accuracy measure)
            
            #given the prediction, take top movies
            index = pred.argsort()[-rec_num:][::-1] #indices
            rec_id = df_train_data.columns[index]
            rec_name = fn_get_movie_name(
                           df_movie=df_movie,
                           movie_id=rec_id,
                           by_id=True,
                           as_list=True, #False?
                           verbose=verbose)         
        else:
            #if the user des not exist, give top ratings
            rec_name = fn_popular_recommendation(
                           user_id=_id, 
                           num_top=rec_num, 
                           df_ranked_movie=rec_id,
                           verbose=verbose)
            
    #if I give a movie, find other movies for it
    else:
        rec_id = fn_find_similar_movie(
                     df_movie=df_movie,
                     dot_product=dot_product,
                     movie_id=_id,
                     cold=True,
                     verbose=verbose)
        
        rec_name = fn_get_movie_name(
                       df_movie=df_movie,
                       movie_id=rec_id,
                       by_id=True,
                       as_list=True,
                       verbose=verbose)
        
    output = rec_id, rec_name
    
    end = time()
    
    if verbose:
        print('elapsed time: {}s'.format(end-begin))
    
    return output

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_make_recommendation_collab(df_filt_dist,
                                  df_user_movie,
                                  df_movie,
                                  num_rec=10,
                                  limit=None,
                                  min_rating=7,
                                  sort=False,
                                  verbose=False):
    
    '''This function takes filtered distances from a focus user and find the
    closest users. Then retrieves already watched movies by the user, just to
    not recommend them. And in sequence, iterate over closest neighbors, retrieve
    their liked movies and it they were not already seen, put in a recommend
    list. For finish, transform movies ids into movies names and return a list.
    
    Source: Udacity Data Science Course - Lesson 6  - Identifying Reccomendations
    Forth Notebook - Class 17 - Collaborative Filtering - Recommendations with 
    MovieTweetings - Collaborative Filtering
    
    Input:
      - df_filt_dist (mandatory) - a user by movie type dataframe, created with 
        the function fn_create_user_item(), from user distances dataset - 
        (Pandas dataset)
        e.g.: filt_dist=df_dist[df_dist['user1'] == user]
      - df_user_movie (mandatory) - dataset in the shape user by movie -
        (Pandas dataset)
      - df_movie (mandatory) - dataset in she shape movies - 
        (Pandas dataset)
      - num_rec (optional) - (int) number of recommended movies to return
      - limit (optional) - extra parameter for fn_find_closest_neighbor - 
        it limits the number of neighbors (normally 100 is more than enough) 
      - min_rating (optional) - extra parameter for fn_movie_liked2() - it is
        the worst score for considering a movie as liked (normally rate 7 is 
        enough)
      - sort (optional) - extra parameter for fn_movie_liked2() - if you want
        to show the best rated movies first (in this algorithm it is useless)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - ls_recommendation - a list of movies - if there are num_rec 
        recommendations return this many otherwise return the total number 
        of recommendations available for the user which may just be an 
        empty list
    '''
    try:
        user_id = df_filt_dist['user1'].iloc[0]
    except TypeError:
        if verbose:
            print('you must inform a filtered dataset, see documentation')
        return False
    except IndexError:
        if verbose:
            print('an empty dataset was informed, this user does not exist')
        return False

    if verbose:
        print('###function make recommendations started')
        
    begin = time()
    
    # movies_seen by user (we don't want to recommend these)
    movie_user = df_user_movie.loc[user_id].dropna()    
    ls_movie_seen = movie_user.index.tolist() 
    
    if verbose:
        print('*seen {} movies by user {} - first is {}'.format(len(ls_movie_seen), user_id, ls_movie_seen[0]))
    
    ls_closest_neighbor = udacourse3.fn_find_closest_neighbor(
                             df_filt_user1=df_filt_dist,
                             limit=limit,
                             verbose=verbose)
    if verbose:
        print('*{} closest neigbors'.format(len(ls_closest_neighbor)))
        
    #creating your recommended array
    rec = np.array([])
    
    #from closest neighbors, 1-take move & 2-that had not been watched
    for i in range (0, len(ls_closest_neighbor)):
        neighbor_id = ls_closest_neighbor[i]
        df_filt_user = df_user_movie.loc[neighbor_id].dropna()
        if verbose:
            print('comparing with neighbor', neighbor_id)
        ls_neighbor_like = udacourse3.fn_movie_liked2(
                               item=df_filt_user,
                               min_rating=7,
                               sort=False,
                               verbose=False)
        if verbose:
            print('...that liked {} movies'.format(len(ls_neighbor_like)))            

        #take recommendations by difference
        new_rec = np.setdiff1d(ls_neighbor_like, 
                               ls_movie_seen, 
                               assume_unique=True)
        if verbose:
            print('...and so, {} were new!'.format(len(new_rec)))
                
        #store rec
        rec = np.unique(np.concatenate([new_rec, rec], axis=0))
        if verbose:
            print('...now I have {} movies in my record'.format(len(rec)))
            #print(rec)
            #print()
         
        #if store is OK, exit the engine
        if len(rec) > num_rec-1:
            break
    
    #take the titles
    ls_recommendation = udacourse3.fn_movie_name(
                            df_movie=df_movie,
                            movie_id=rec,
                            verbose=False)
    end = time()
    
    if verbose:
        print('elapsed time: {:.4f}s'.format(end-begin))
    
    return ls_recommendation

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_make_recommendation_content(df_movie,
                                   dot_product,
                                   user,
                                   verbose=False):
    '''This function makes recommendation for new movies to a user, based on
    content.
    
    Source: Udacity Data Science Course - Lesson 6 - Ways to Reccomend
    Fifth Notebook - Class 21 - Content Based Recommendations
    
    Input:
      - df_movie (mandatory) - your movies dataset (Pandas dataset)
      - dot_product (mandatory) - your dot product matrix autocorrelation for 
        movies rating, with movies collinearity values. 
        You need to preprocess this earlier
      - user (mandatory) - the user ID (Integer)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - recommendation - a Python dictionary containing user as keys and
        values for recommendations
    '''
    if verbose:
        print('###function make recommendations started')
        
    begin = time()

    # Create dictionary to return with users and ratings
    dic_recommendation = defaultdict(set)
    # How many users for progress bar
    try:
        num_user = len(user)
    except TypeError:
        user = [user]
        if verbose:
            print('only one user was informed, putting it in a list')
        num_user = len(user)

    # Create the progressbar
    counter = 0
    bar = progressbar.ProgressBar(maxval=num_user+1, 
                                  widgets=[progressbar.Bar('=', '[', ']'), 
                                                           ' ',
                                                           progressbar.Percentage()])
    bar.start()
    
    #iterate user by user
    for one_user in user:
        
        #only updating the progress bar
        counter += 1 
        bar.update(counter)

        #taking only the reviews seen by this user
        review_temp = ranked_review[ranked_review['user_id'] == one_user]
        movie_temp = np.array(review_temp['movie_id'])
        movie_name = np.array(udacourse3.fn_get_movie_name(
                                  df_movie=movie,
                                  movie_id=movie_temp,
                                  verbose=verbose))

        #iterate each of these movies (highest ranked first) 
        #taking only the movies that were not watched by this user
        #and that are most similar - these are elected to be the
        #recommendations - I need only 10 of them!
        #you keep going until there are no more movies in the list
        #of this user
        for movie_id in movie_temp:
            recommended_movie = udacourse3.fn_find_similar_movie(
                                    dot_product=dot_prod_movie,
                                    df_movie=df_movie,
                                    movie_id=movie_id,
                                    verbose=True)
            
            temp_recommendation = np.setdiff1d(recommended_movie, movie_name)
            
            dic_recommendation[one_user].update(temp_recommendation)

            #if you have the enough number of recommendations, you will stop
            if len(recommendation[one_user]) >= 10:
                break

    bar.finish()
    end = time()
    
    if verbose:
        print('elapsed time: {:.2f}s'.format(end-begin))
    
    return dic_recommendation

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_movie_liked2(item, 
                    min_rating=7,
                    sort=True,
                    verbose=False):
    '''This function takes all the items for one user and return the best rated
    items.
    
    Source: Udacity Data Science Course - Lesson 6  - Identifying Reccomendations
    Forth Notebook - Class 17 - Collaborative Filtering -Recommendations with 
    MovieTweetings - Collaborative Filtering

    Inputs:
      - item (mandatory) - a dataset on the shape user by movie, filtered 
        for an individual user.
        e.g. item=user_by_movie.loc[user_id].dropna()  
      - min_rating (optional) - the trigger point to consider an item to be 
        considered "nice" for an user (integer, default=7)
      - sort (optional) - if you want to show the most rated items first
        (Boolean, default=True)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - ls_movie_liked - an array of movies the user has watched and liked
    '''
    user = item.name

    if verbose:
        print('###function movies liked started for user', user)

    begin = time()
    movie_liked = item[item > 7]
    
    if sort:
        if verbose:
            print('*sorting the list - best rated first')
        movie_liked = movie_liked.sort_values(ascending=False)

    ls_movie_liked = np.array(movie_liked.index)
    
    end = time()
    
    if verbose:
        print('elapsed time: {:.4f}s'.format(end-begin))
    
    return ls_movie_liked

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_movie_name(df_movie,
                  movie_id,
                  verbose=False):
    '''This function takes a list of movies_id (liked from other user) and
    returns ne movie titles
    
    Source: Udacity Data Science Course - Lesson 6  - Identifying Reccomendations
    Forth Notebook - Class 17 - Collaborative Filtering -Recommendations with 
    MovieTweetings - Collaborative Filtering    
    
    Inputs:
      - df_movie (mandatory) - the movies dataset - Pandas Dataset
      - movie_id (mandatory) - a numpy vector containing a list of movie_ids
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - movies - a list of movie names associated with the movie_ids (python
        list)
    '''
    if verbose:
        print('###function movie names started')
        
    begin = time()
    
    df_movie_get = df_movie[df_movie['movie_id'].isin(movie_id)]
    movie_list = df_movie_get['movie'].to_list()
    
    end = time()
    
    if verbose:
        print('elapsed time: {:.4f}s'.format(end-begin))
   
    return movie_list

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_movie_watched(df_user_movie,
                     user_id,
                     lower_filter=None,
                     verbose=False):
    '''This function creates a array structure. Keys are users, content is
    a list of movies seen. DF is a user by movie dataset, created with the
    function fn_create_user_item.
    
    Source: Udacity Data Science Course - Lesson 6  - Identifying Reccomendations
    Forth Notebook - Class 17 - Collaborative Filtering -Recommendations with 
    MovieTweetings - Collaborative Filtering

    Input:
      - df_user_movie (mandatory) df_user_movie (mandatory) - a user by movie 
        type dataset, created with the function fn_create_user_item(), from 
        user (Pandas dataset)
      - user_id (mandatory) - the user_id of an individual as int
      - lower_filter (optional) - elliminate users with a number of watched
        videos below the number (Integer, default=None)      
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - movies - an array of movies the user has watched
    '''
    if verbose:
        print('###function movies watched started')
        
    begin = time()
    
    movie = df_user_movie.loc[user_id][df_user_movie.loc[user_id].isnull() == False].index.values
    
    end = time()
    if verbose:
        print('elapsed time: {:.4f}s'.format(end-begin))
        
    if lower_filter == None:
        return movie
    elif lower_filter > 0:
        if len(movie) > lower_filter:
            return movie
        else:
            return None
    else:
        raise Exception('something went wrong with lower filter parameter')

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_popular_recommendation(df_ranked_movie,
                              user_id,
                              num_top,
                              verbose=False):
    '''this function makes a list of top recommended movies, by title. Laterly 
    this function can be forked for other purposes.
    
    Source: Udacity Data Science Course - Lesson 6 - Recomendation Engines - 
    Class 8 - Second Notebook - Intro to Recommendation data - Finding Most 
    Popular Movies.

    Input:
      - df_ranked_movie (mandatory) - a pandas dataframe of the already ranked 
        movies based on avg rating, count, and time
      - user_id (mandatory) - the user_id (str) of the individual you are making 
        recommendations for
      - num_top (mandatory) - an integer of the number recommendations you want
        back
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - ls_top_movies - a Pandas Series of the num_top recommended movies by 
        movie title in order, from the best to the worst.
    '''
    if verbose:
        print('###function popular recomendations started')
        
    begin = time()

    ls_top_movie = list(df_ranked_movie['movie'][:num_top])

    end = time()
    
    if verbose:
        print('elapsed time: {:.6f}s'.format(end-begin))

    return ls_top_movie

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_popular_recommendation_filtered(df_ranked_movie,
                                       user_id, 
                                       num_top, 
                                       year=None, 
                                       genre=None,
                                       verbose=False):
    '''This function creates some filter for adding robustness for our model.
    Laterly this function can be forked for other purposes.
    
    Source: Udacity Data Science Course - Lesson 6 - Recommendation Engines - 
    Class 5 - First Notebook - Intro to Recommendation data - Part II - Adding
    Filters.
    
    Inputs:
      - df_ranked_movie (mandatory) - a pandas dataframe of the already ranked movies
        based on average rating, count, and time
      - user_id (mandatory) - the user_id (str) of the individual you are making 
        recommendations for
      - num_top (mandatory) - an integer of the number recommendations you want
        back
      - year (mandatory) - a list of strings with years of movies
      - genre (mandatory) - a list of strings with genres of movies
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - ls_top_movie - a list of the num_top recommended movies by movie title in 
        order from the best to worst one
    '''    
    if verbose:
        print('###function popular recommendations (filtered) started')
        
    begin = time()

    #a year filter
    if year is not None:
        if verbose:
            print('*year filter activated')
        df_ranked_movie = df_ranked_movie[df_ranked_movie['date'].isin(year)]

    #a genre filter    
    if genre is not None:
        if verbose:
            print('*genre filter activated')
        num_genre_match = df_ranked_movie[genre].sum(axis=1)
        #at least one was found!
        df_ranked_movie = df_ranked_movie.loc[num_genre_match >= 1, :] 
                  
    #recreate a top list for movies (now filtered!)
    #num_top is the cutting criteria!
    ls_top_movie = list(df_ranked_movie['movie'][:num_top]) 

    end = time()
    
    if verbose:
        print('elapsed time: {:.4f}s'.format(end-begin))

    return ls_top_movie

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_predict_rating(df_train,
                      user_matrix, 
                      movie_matrix, 
                      user_id, 
                      movie_id,
                      verbose=False):
    '''This function predicts the rating for the FunkSVD engine.
    
    Source: Udacity Data Science Course - Lesson 7  - Matrix Factorization for
    Recommendations - Third Notebook - Class 18 - How are we doing w/ FunkSVD
    
    Inputs:
      - df_train (mandatory) - your train data dataset (Pandas dataset)
      - user_matrix (mandatory) - user by latent factor matrix
      - movie_matrix (mandatory) - latent factor by movie matrix
      - user_id (mandatory) - the user_id from the reviews df
      - movie_id (mandatory) - the movie_id according the movies df
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)    
    Output:
    predicted 
      - the predicted rating for user_id-movie_id according to FunkSVD
    '''
    if verbose:
        print('###function movies watched started')
        
    begin = time()

    #your training data -> vector of users & movies
    #+ same order of train data
    user_id_vector = np.array(df_train.index)
    movie_id_vector = np.array(df_train.columns)
    
    #user & movie vectors
    user_row = np.where(user_id_vector == user_id)[0][0]
    try:
        movie_col = np.where(movie_id_vector == movie_id)[0][0]
        if verbose:
            print('for movie {} our prediction is'.format(movie_id))
            print(movie_col)
    except IndexError:
        if verbose:
            print('*cannot predict for movie {}'.format(movie_id))
        return False
    
    #dot product of user x movie -> make your predictions
    predict = np.dot(user_matrix[user_row, :], movie_matrix[:, movie_col])
    
    end = time()
    if verbose:
        print('predict rate: {:.6f}'.format(predict))
        print('elapsed time: {:.4f}s'.format(end-begin))
    
    return predict

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_print_prediction_summary(df_movie,
                                user_id, 
                                movie_id, 
                                prediction,
                                verbose=False):
    '''This function is only a relatory function! It prints the prediction
    summary for the predictions under our FunkSVD engine.
    
    Source: Udacity Data Science Course - Lesson 7  - Matrix Factorization for
    Recommendations - Third Notebook - Class 18 - How are we doing w/ FunkSVD
    
    Inputs:
      - df_movie (mandatory) - the movies dataset (Pnadas dataset)
      - user_id (mandatory) - the user_id from the reviews df
      - movie_id (mandatory) - the movie_id according the movies df
      - prediction (mandatory) - the predicted rating for user_id-movie_id
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)    
    Output:
      - True, if the function runs well - this is a relatory function!
    '''
    if verbose:
        print('###function movies watched started')

    #begin = time()

    movie_name = str(df_movie[df_movie['movie_id'] == movie_id]['movie'])[5:]
    movie_name = movie_name.replace('\nName: movie, dtype: object', '')
    print("For user {} we predict a {} rating for the movie {}."\
          .format(user_id, round(prediction, 2), str(movie_name)))

    #end = time()
    #if verbose:
    #    print('elapsed time: {}s'.format(end-begin))

    return True

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_take_correlation(ls_for_user1, 
                        ls_for_user2,
                        verbose=False):
    '''This function takes two movies series from a dataset like 
    movies_to_analyze and returns the correlation between these users.

    Important Observation:
    The main dataset is normally too large to be passed into the function. So
    you need to make a pre-filter and feed this function ONLY with the data
    of the two users in focus. 
    Example of filter: for_user1=user_by_movie.loc[2] <- 2 is the ID of user1
        
    Source: Udacity Data Science Course - Lesson 6  - Identifying Reccomendations
    Forth Notebook - Class 17 - Collaborative Filtering -Recommendations with 
    MovieTweetings - Collaborative Filtering    
    
    Input:
      - for_user1 (mandatory) - raw series of movies data for User 1. It can 
        contain NaN (Pandas Series) 
      - for_user2 (mandatory) - raw series of movies data for User 2. It can 
        contain NaN (Pandas Series) 
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - the correlation between the matching ratings between the two users
    '''
    if verbose:
        print('###function take correlation coefficient started')
        
    begin = time()

    #find the movie list for each user
    ls_usr1_movie = ls_for_user1.dropna().index
    ls_usr2_movie = ls_for_user2.dropna().index

    #getting the insersections
    sim_mov = np.intersect1d(ls_usr1_movie, 
                             ls_usr2_movie, 
                             assume_unique=True)
    
    #finding the weights for the insersection movies
    #sim_mov=[1454468, 1798709, 2883512]] for user1=2 and user2=66
    sr1 = ls_for_user1.loc[sim_mov] 
    sr2 = ls_for_user2.loc[sim_mov]
    
    correlation = fn_compute_correlation(x=sr1, 
                                         y=sr2,
                                         corr_type='intersection',
                                         verbose=verbose)    
    end = time()
    
    if verbose:
        print('elapsed time: {:.4f}s'.format(end-begin))
    
    return correlation

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_take_euclidean_dist(ls_for_user1, 
                           ls_for_user2,
                           verbose=False):
    '''This function takes two movies series from a dataset like 
    movies_to_analyze and returns the correlation between these users.

    Important Observation:
    The main dataset is normally too large to be passed into the function. So
    you need to make a pre-filter and feed this function ONLY with the data
    of the two users in focus. 
    Example of filter: for_user1=user_by_movie.loc[2] <- 2 is the ID of user1
        
    Source: Udacity Data Science Course - Lesson 6  - Identifying Reccomendations
    Forth Notebook - Class 17 - Collaborative Filtering -Recommendations with 
    MovieTweetings - Collaborative Filtering    
    
    Input:
      - for_user1 (mandatory) - raw series of movies data for User 1. It can 
        contain NaN (Pandas Series) 
      - for_user2 (mandatory) - raw series of movies data for User 2. It can 
        contain NaN (Pandas Series) 
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - the euclidean distance between the two users
    '''
    if verbose:
        print('###function take euclidean distance started')
        
    begin = time()
    
    #find the movie list for each user
    try:
        ls_usr1_movie = ls_for_user1.dropna().index
    except AttributeError:
        if verbose:
            print('you need to give a dataset row, see the function documentation')
        return False
    try:
        ls_usr2_movie = ls_for_user2.dropna().index
    except AttributeError:
        if verbose:
            print('you need to give a dataset row, as for_user2=user_by_movie.loc[66]')
        return False
    
    #getting the insersections
    ls_sim_mov = np.intersect1d(ls_usr1_movie, 
                                ls_usr2_movie, 
                                assume_unique=True)
    
    #finding the weights for the insersection movies
    #sim_mov=[1454468, 1798709, 2883512]] for user1=2 and user2=66
    sr1 = ls_for_user1.loc[ls_sim_mov] 
    sr2 = ls_for_user2.loc[ls_sim_mov]
        
    euclidean_distance = fn_calculate_distance(x=sr1, 
                                               y=sr2,
                                               dist_type='euclidean',
                                               verbose=verbose) 
    end = time()
    
    if verbose:
        print('elapsed time: {}s'.format(end-begin))
    
    return euclidean_distance

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_validation_comparison(df_train, 
                             df_val,
                             user_matrix, 
                             movie_matrix,
                             num_pred=0,
                             relatory=True,
                             verbose=False,
                             graph=False):
    '''This is a relatory function only! It prints the comparision for valitate
    the prediction made, row by row of val_df.
    
    Source: Udacity Data Science Course - Lesson 7  - Matrix Factorization for
    Recommendations - Third Notebook - Class 18 - How are we doing w/ FunkSVD
    
    Input:
      - df_train (mandatory) - your train data dataset (Pandas dataset)
      - df_val (mandatory) - the validation dataset created in the third cell 
        above (Pandas dataset)
      - user_matrix (mandatory) - user by latent factor matrix
      - movie_matrix (mandatory) - latent factor by movie matrix    
      - num_pred (mandatory) - (int) the number of rows (going in order) 
        you would like to make predictions for
      - relatory(optional) - if you want only a text relatory (default=True)
        (relatory=False) will give full processing 
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
      - graph (optional) - if you want to plot a heat graph, please check it
        (default=False) - works only for full mode (relatory=False)
    Output:
      - return True if erverything goes well - this is a relatory function!
        if (relatory=True)
      - return full components if (relatory=False)
        * rmse - root-mean-square error (accuracy measure)
        * ind_rated - relation between counts of rated/user
        * actual_v_pred
        * ls_pred
        * ls_act
    '''
    if verbose:
        print('###validation comparison function started')
        
    begin=time()
    val_user = np.array(df_val['user_id'])
    val_movie = np.array(df_val['movie_id'])
    val_rating = np.array(df_val['rating'])

    if relatory: #relatory only
        if graph:
            print('graph for relatory is not supported yet!')
        if num_pred < 1:
            if verbose:
                print('no predictions asked')
            return False
        else:
            for idx in range(num_pred): #call for our subfunction
                #if verbose:
                #    print('*calling predict rating subfunction')
                pred = fn_predict_rating(
                           df_train=df_train,
                           user_matrix=user_matrix, 
                           movie_matrix=movie_matrix, 
                           user_id=val_user[idx],
                           movie_id=val_movie[idx],
                           verbose=False) #I don´t want verbosity in relat
                if not pred:
                    if verbose:
                        print('system halted')
                    return False
        
                print('Our rating actual: {} → for user {} on movie {}'\
                      .format(val_rating[idx],
                              val_user[idx], 
                              val_movie[idx]))
                print('         predited: {}'.format(round(pred)))
        return True

    else: #full mode              
        sse = 0
        num_rated = 0
        ls_pred, ls_act = [], []
        actual_v_pred = np.zeros((10,10))
        for idx in range(len(val_user)):
            try:
                pred = fn_predict_rating(
                           df_train=df_train,
                           user_matrix=user_matrix, 
                           movie_matrix=movie_matrix, 
                           user_id=val_user[idx], 
                           movie_id=val_movie[idx],
                           verbose=verbose)
                sse += (val_rating[idx] - pred)**2
                num_rated += 1
                ls_pred.append(pred)
                ls_act.append(val_rating[idx])
                actual_v_pred[11 - int(val_rating[idx] - 1), 
                              int(round(pred) - 1)] += 1
        
            except:
                continue 
        
        #rmsd - root-mean-square deviation
        #second sample moment - measure for accuracy
        #allways positive
        #rmse - root-mean-square error
        #residuals - differences values predicted x observed
        rmse = np.sqrt(sse / num_rated)
        ind_rated = num_rated / len(val_user)
    
        output = rmse, ind_rated, actual_v_pred, ls_pred, ls_act
        
        if graph:
            #https://www.delftstack.com/howto/seaborn/size-of-seaborn-heatmap/
            sns.set(rc = {'figure.figsize':(10,7)})
            sns.heatmap(actual_v_pred)
            mpyplots.xticks(np.arange(10), np.arange(1,11))
            mpyplots.yticks(np.arange(10), np.arange(1,11))
            mpyplots.xlabel("Predicted Values")
            mpyplots.ylabel("Actual Values")
            mpyplots.title("Actual vs. Predicted Values");
            
            mpyplots.figure(figsize=(8,8))
            #https://stdworkflow.com/67/attributeerror-rectangle-object-has-no-property-normed-solution
            #density=True for frequency distribution 
            #(not probability density distribution)
            mpyplots.hist(ls_act, 
                          density=True, 
                          stacked=True, 
                          alpha=.5, 
                          label='actual');
            mpyplots.hist(ls_pred, 
                          density=True, 
                          stacked=True, 
                          alpha=.5, 
                          label='predicted');
            mpyplots.legend(loc=2, prop={'size': 15});
            mpyplots.xlabel('Rating');
            mpyplots.title('Actual vs. Predicted Rating');
    
        end = time()
        if verbose:
            print('rmse (root-mean-square error): {:.4f} of accuracy - '.format(rmse)) 
            print('rated: {:.2f}%'.format(ind_rated*100.))
            print('elapsed time: {:.4f}s'.format(end-begin))

        return output
    
###Stats functions##############################################################
#########1#########2#########3#########4#########5#########6#########7#########8
def fn_experiment_size(p_null, 
                       p_alt, 
                       alpha=0.05, 
                       beta=0.20,
                       verbose=False):
    '''This function takes a size of effect and returns the minimum number of 
    samples needed to achieve the desired power.
    
    Source: Udacity Data Science Course - Lesson 4 - Statistical Considerations
    into testing - Second Notebook - Experiment Size - Analytic Solution.
    
    Inputs:
      - p_null (mandatory) - null hypothesis success rate (base) - (numpy Float)
      - p_alt (mandatory) - success rate (desired) - what we want to detect -
        (numpy Float)
      - alpha (optional) - Type-I (false positive) rate of error - (numpy Float -
        default=5%)
      - beta (optional) - Type-II (false negative) rate of error - (numpy Fload -
        default=20%)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - n - required number of samples for each group, in order to obtain the 
        desired power (it is considered that the share for each group is 
        equivalent)
    '''
    if verbose:
        print('###function experiment size started - Analytic solution')
        
    begin = time()

    #takes z-scores and st dev -> 1 observation per group!
    z_null = stats.norm.ppf(1-alpha)
    z_alt  = stats.norm.ppf(beta)
    sd_null = np.sqrt(p_null * (1-p_null) + p_null * (1-p_null))
    sd_alt  = np.sqrt(p_null * (1-p_null) + p_alt  * (1-p_alt) )
    
    #calculate the minimum sample size
    p_diff = p_alt - p_null
    num = ((z_null*sd_null - z_alt*sd_alt) / p_diff) ** 2
    
    num_max = np.ceil(num)
    
    end = time()
    
    if verbose:
        print('elapsed time: {:.4f}s'.format(end-begin))
        print('experiment size:', num_max)

    return num_max

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_invariant_analytic(df,
                          p=0.5,
                          value=0.0,
                          verbose=False):
    '''This function checks a invariant metric by analytic approach. One example
    of a invariant is if the division between two different webpages (one for H0
    and the other for H1) is similar.
    
    Source: Udacity Data Science Course - Lesson 4 - Statistical Considerations 
    into testing - Class 3 - First Notebook - Statistical Significance.
    
    Inputs:
      - df (mandatory) - dataset containing the binary data to be checked
      - p (optional) - probability (default=50%)
      - value (optional) - central value (default=0)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - It is a report function only! Returns True if everything runned well
    '''
    if verbose:
        print ('###function for invariant check started - analytic approach')
        
    begin = time()
    
    #number of trials
    num_observation = df.shape[0]
    if verbose:
        print('number of observations:', num_observation)
    
    #number of successes
    num_control = df[df['condition'] == 0].shape[0] #data.groupby('condition').size()[0]
    if verbose:
        print('number of cases with success:', num_control)
        
    #z-score and p-value
    st_dev = np.sqrt(num_observation * p * (1-p)) #formula for binomial distribution
    if verbose:
        print('Standard Deviation for Binomial Distribution: {:.1f}'.format(st_dev))
    
    z_score = ((num_control + 0.5) - p * num_observation) / st_dev
    if verbose:
        print('z-score: {:.4f}'.format(z_score))
        
    #cumulative distribution function
    cdf = stats.norm.cdf(0)
    if verbose:
        print('cumulative values over {} is {:.4f} ({:.2f}%)'\
              .format(value, cdf, cdf*100))
    
    #analytic p-value
    p_value = 2 * stats.norm.cdf(z_score)
    
    end = time()
    if verbose:
        print()
        print('analytic p-value: {:.4f} ({:.2f}%)'.format(p_value, p_value*100))
        print('*a low value means that H0 don´t have a good explanatory power')
        print('elapsed time: {:.4f}s'.format(end-begin))
        
    return True
        
#########1#########2#########3#########4#########5#########6#########7#########8
def fn_invariant_simulated(df,
                           p=0.5,
                           num_trials=200_000,
                           verbose=False):
    '''This function checks a invariant metric by a simulation approach.
    
    Source: Udacity Data Science Course - Lesson 4 - Statistical Considerations 
    into testing - Class 3 - First Notebook - Statistical Significance.
    
    Inputs:
      - df (mandatory) - dataset containing the binary data to be checked
      - p (optional) - probability (default=50%)
      - num_trials - number of trials for randomly simulating the experiment
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - It is a report function only! Returns True if everything runned well
    '''
    if verbose:
        print ('###function for invariant check started - simulation approach')
        
    begin = time()
    
    #number of trials
    num_observation = df.shape[0]
    if verbose:
        print('number of observations:', num_observation)        
        
    #number of successes
    num_control = df[df['condition'] == 0].shape[0] #data.groupby('condition').size()[0]
    if verbose:
        print('number of cases with success:', num_control)

    #simulate outcomes under null, compare to observed outcome
    samples = np.random.binomial(num_observation, p, num_trials)
        
    if verbose:
        print()
        print('*simulation part')
        print('simulated samples: {}'.format(len(samples)))
            
    #number of samples below control number
    samples_below = sum(samples <= num_control)
    if verbose:
        print('number of cases below control:', samples_below) 
    
    #samples above control number
    samples_above = sum(samples >= (num_observation-num_control))
    
    #simulated p-value
    p_value = np.logical_or(samples <= num_control, samples >= (num_observation-num_control)).mean()
    
    end = time()
    
    if verbose:
        print()
        print('simulated p_value: {:.4f} ({:.2f}%)'\
              .format(p_value, p_value*100))
        print('*a low value means that H0 don´t have a good explanatory power')
        print('elapsed time: {:.4f}s'.format(end-begin))
        
    return True
        
#########1#########2#########3#########4#########5#########6#########7#########8
def fn_peeking_correction(alpha=0.05, 
                          p_success=0.5, 
                          num_trials=1000, 
                          num_blocks=2, 
                          num_sims=10000,
                          verbose=False):
    '''This function make a estimative of the individual error rate necessary
    to limit the Type I error (false positive) rate, if an early stopping decision
    is made. It uses a simulation, to predict if significant result could exist
    when peeking ahead.
    
    Source: Udacity Data Science Course - Lesson 4 - Statistical Considerations
    into testing - Class 14 - Fifth Notebook - Early Stopping - with Multiple 
    Corrections.
    
    Inputs:
      - alpha (optional) - overall Type I error (false positive) rate that was 
        desired for the experiment - (numpy Float - default=5%) 
      - p_success (optional): probability of obtaining a success on an individual
        trial - (numpy Float - default=-50%)
      - num_trials (optional): number of trials that runs in a full experiment
        (Integer - default=10000)
      - num_blocks (optional): number of times that a a data is looked for
        (including the end) - (Integer - default=2)
      - num_sims (optional) - number of times that the simulated experiments run
        (Integer - default=10000)
    Output:
        alpha_individual: Individual error rate required to achieve overall error 
        rate
    '''
    if verbose:
        print('###function peeking correction started')

    begin=time()

    #data generation
    trials_per_block = np.ceil(num_trials / num_blocks).astype(int)
    try:
        data = np.random.binomial(trials_per_block, 
                                  p_success, 
                                  [num_sims, num_blocks])
    except ValueError:
        print('something went wrong doing binomials - p seems to be invalid!')
        return (trials_per_block, p_success, [num_sims, num_blocks])
    
    #data standardization
    data_cumsum = np.cumsum(data, axis = 1)
    block_sizes = trials_per_block * np.arange(1, num_blocks+1, 1)
    block_means = block_sizes * p_success
    block_sds   = np.sqrt(block_sizes * p_success * (1-p_success))
    data_zscores = (data_cumsum - block_means) / block_sds
    
    #the necessary individual error rate
    max_zscores = np.abs(data_zscores).max(axis = 1)
    z_crit_ind = np.percentile(max_zscores, 100 * (1 - alpha))
    alpha_individual = 2 * (1 - stats.norm.cdf(z_crit_ind))
 
    end = time()
    
    if verbose:
        print('probabilities - alpha individual: {:.4f} ({:.2f}%)'\
              .format(alpha_individual, alpha_individual*100))
        print('elapsed time: {:.4f}s'.format(end-begin))

    return alpha_individual

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_peeking_simulation(alpha=0.05, 
                          p=0.5, 
                          num_trials=1000,
                          num_blocks=2,
                          num_sims=10_000,
                          verbose=False):
    '''This function aims to simulate the rate of Type I error (false positive) 
    produced by the early stopping decision. It is based on a significant result
    when peeking ahead.
    
    Source: Udacity Data Science Course - Lesson 4 - Statistical Considerations
    into testing - Class 15 - Fifth Notebook - Early Stopping.
    
    Inputs:
        - alpha (optional) - Type I error rate that was supposed
        - p (optional) - probability of individual trial success
        - num_trials (optional) - number of trials in a full experiment
        - num_blocks (optional) - number of times data is looked at (including end)
        - num_sims: Number of simulated experiments run
    Output:
        p_sig_any: proportion of simulations significant at any check point, 
        p_sig_each: proportion of simulations significant at each check point
    '''
    if verbose:
        print('###function peeking for early stopping started - Simulating {} times'\
              .format(num_sims))
        
    begin=time()
    
    #generate the data
    trials_per_block = np.ceil(num_trials / num_blocks).astype(int)
    data = np.random.binomial(trials_per_block, p, [num_sims, num_blocks])
    
    #put the data under a standard
    data_cumsum = np.cumsum(data, axis=1) #cumsum is s summation 
    block_sizes = trials_per_block * np.arange(1, num_blocks+1, 1)
    block_means = block_sizes * p
    block_sds = np.sqrt(block_sizes * p * (1-p))
    data_zscores = (data_cumsum - block_means) / block_sds
    
    #results
    z_crit = stats.norm.ppf(1-alpha/2) #norm is a normal distribution
    sig_flags = np.abs(data_zscores) > z_crit
    p_sig_any = (sig_flags.sum(axis=1) > 0).mean()
    p_sig_each = sig_flags.mean(axis=0)
    
    tuple = (p_sig_any, p_sig_each)
    
    end = time()
    
    if verbose:
        print('probabilities - signal(any): {:.4f} ({:.2f}%), signal(each): {}'\
              .format(p_sig_any, p_sig_any*100, p_sig_each))
        print('elapsed time: {:.4f}s'.format(end-begin))
    
    return tuple

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_power(p_null, 
             p_alt, 
             num, 
             alpha=0.05, 
             plot=False,
             verbose=False):
    '''This function takes an alpha rate and computes the power of detecting the 
    difference in two populations.The populations can have different proportion 
    parameters.
    
    Source: Udacity Data Science Course - Lesson 4 - Statistical Considerations
    into testing - Class 5 - Second Notebook - Experiment Size - By Trial and 
    Error.
    
    Inputs:
      - p_null (mandatory) - rate of success (base) under the Null hypothesis
        (numpy Float) 
      - p_alt (mandatory) -  rate of sucess (desired) must be larger than the
        first parameter - (numpy Float)
      - num (mandatory) - number of observations for each group - (integer)
        alpha (optional) - rate of Type-I error (false positive-normally the
        more dangerous) - (numpy Float - default 5%)
      - plot (optional) - if you want to plot the distribution - (Boolean, 
        default=False)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
        power - the power for detection of the desired difference under the 
        Null Hypothesis.
    '''
    if verbose:
        print('###function power of an Hypothesis started - by Trial  & Error')
        
    begin = time()
    
    #the idea: start with the null hypothesis. Our main target is to find 
    #Type I errors (false positive) trigger (critical value is given by
    #Alpha parameter - normally 5%).
    
    #se_null → standard deviation for the difference in proportions under the
    #null hypothesis for both groups
    #-the base probability is given by p_null
    #-the variance of the difference distribution is the sum of the variances for
    #-the individual distributions
    #-for each group is assigned n observations.
    se_null = np.sqrt((p_null * (1-p_null) + p_null * (1-p_null)) / num)
    #null_dist → normal continuous random variable (form Scipy doc)
    null_dist = stats.norm(loc=0, scale=se_null)

    #p_crit: Compute the critical value of the distribution that would cause us 
    #to reject the null hypothesis. One of the methods of the null_dist object 
    #will help you obtain this value (passing in some function of our desired 
    #error rate alpha). The power is the proportion of the distribution under 
    #the alternative hypothesis that is past that previously-obtained critical value.
    p_crit = null_dist.ppf(1-alpha) #1-alpha=95%
    
    #se_alt: Now it's time to make computations in the other direction. 
    #This will be standard deviation of differences under the desired detectable 
    #difference. Note that the individual distributions will have different variances 
    #now: one with p_null probability of success, and the other with p_alt probability of success.
    se_alt  = np.sqrt((p_null * (1-p_null) + p_alt  * (1-p_alt)) / num)

    #alt_dist: This will be a scipy norm object like above. Be careful of the 
    #"loc" argument in this one. The way the power function is set up, it expects 
    #p_alt to be greater than p_null, for a positive difference.
    alt_dist = stats.norm(loc=p_alt-p_null, scale=se_alt)

    #beta → Type-II error (false negative) - I fail to reject the null for some
    #non-null states
    beta = alt_dist.cdf(p_crit)    
    
    if plot:
        plot = fn_plot(first_graph=null_dist, 
                       second_graph=alt_dist,
                       aux=p_crit,
                       type='htest',
                       verbose=verbose)
        
    power = (1 - beta)
    end = time()
    
    if verbose:
        print('hypotesis power coefficient: {:.4f} ({:.2f}%)'.format(power, power*100))
        print('*a low value means that H0 don´t have a good explanatory power')
        print('elapsed time: {:.4f}s'.format(end-begin))
        
    return power

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_quantile_ci(data, 
                   q, 
                   c=0.95, 
                   num_trials=1000,
                   plot=False,
                   verbose=False):
    '''This function takes a quartile for a data and returns a confidence 
    interval, using Bootstrap method.
    
    Source: Udacity Data Science Course - Lesson 4 - Statistical Considerations
    into testing - Class 8 - Third Notebook - Non-Parametric Tests - Bootstrapping 
    Confidence Intervals using Quantiles.
    
    Inputs:
      - data (mandatory) - a series of numpy Float data to be processed - it
        can be a Pandas Series - (numpy Array)
      - q (mandatory) - quantile to be estimated - (numpy Array - between 0 and 1)
      - c (optional) - confidence interval - (float, default: 95%)
      - num_trials (optional) - the number of samples that bootstrap will perform
        (default=1000)
      - plot (optional) - if you want to plot the distribution - (Boolean, 
        default=False)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - ci - upper an lower bound for the confidence interval - (Tuple of numpy Float)
    '''
    if verbose:
        print("###function quantile confidence interval started - Bootstrapping method")
        
    begin=time()
    
    if plot:
        if verbose:
            print('initial histogram distribution graph')
        plot = fn_plot(first_graph=data, 
                       type='hist',
                       verbose=verbose)

    #sample quantiles for bootstrap
    num_points = data.shape[0]
    sample_qs = []
    
    #loop for each bootstrap element
    for _ in range(num_trials):
        #random sample for the data
        sample = np.random.choice(data, num_points, replace=True) #with replacement
        
        #desired quantile
        sample_q = np.percentile(sample, 100 * q)
        
        #append to the list of sampled quantiles
        sample_qs.append(sample_q)
        
    #confidence interval bonds
    lower_limit = np.percentile(sample_qs, (1 - c) / 2 * 100)
    upper_limit = np.percentile(sample_qs, (1 + c) / 2 * 100)
    
    tuple = (lower_limit, upper_limit)
    end = time()
    
    if verbose:
        print('confidence interval - lower: {:.4f} ({:.2f}%) upper: {:.4f} ({:.2f}%)'\
              .format(lower_limit, lower_limit*100, upper_limit, upper_limit*100))
        print('elapsed time: {:.4f}s'.format(end-begin))

    return (lower_limit, upper_limit)

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_quantile_permutation_test(x, 
                                 y, 
                                 q, 
                                 alternative='less',
                                 num_trials=10_000,
                                 plot=False,
                                 verbose=False):
    '''this function takes a vector of independent feature, another of dependent
    feature and calculates a confidence interval for a quantile of a dataset.
    It uses a Bootstrap method.
    
    Source: Udacity Data Science Course - Lesson 4 - Statistical Considerations
    into testing - Class 8 - Third Notebook - Non-Parametric Tests - Bootstrapping
    Confidence intervals using Permutation Test.
    
    Inputs:
      - x (mandatory) - a vector containing zeroes and 1 values, for the 
        independent (to be grouped) feature - (Boolean)
      - y (mandatory) - a vector containing zeroes and 1 values, for the 
        dependent (output) feature
      - q (mandatory) - a verctor containing zeroes and 1 valures for the output
        quantile
      - alternative (optional) - please inform the type of test to be performed
        (possible: 'less' and 'greater') - (default='less')
      - num_trials (optional) number of permutation trials to perform 
      - plot (optional) - if you want to plot the distribution - (Boolean, 
        default=False)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - the estimated p-value of the test (numpy Float)
    '''
    if verbose:
        print("###function quantile permutation test - Bootstrapping method")
        
    begin=time()
    
    if plot:
        if verbose:
            print('first showing histogram graphs for both condition 0 and condition 1')
        df_plot = pd.concat([y, x], axis=1, join="inner") #recreate dataset
        plot = fn_plot(first_graph=df_plot[df_plot['condition'] == 0]['time'], 
                       second_graph=df_plot[df_plot['condition'] == 1]['time'],
                       aux=df_plot['time'],
                       type='2hist',
                       verbose=verbose)      
          
    #initialize list for bootstrapped sample quantiles
    sample_diffs = []
    
    #loop on trials
    for _ in range(num_trials):
        #permute the grouping labels
        labels = np.random.permutation(y)
        
        #difference in quantiles
        cond_q = np.percentile(x[labels == 0], 100 * q)
        exp_q  = np.percentile(x[labels == 1], 100 * q)
        
        #add to the list of sampled differences
        sample_diffs.append(exp_q - cond_q)
    
    #observed statistic for the difference
    cond_q = np.percentile(x[y == 0], 100 * q)
    exp_q  = np.percentile(x[y == 1], 100 * q)
    obs_diff = exp_q - cond_q
    
    #p-value for the result
    if alternative == 'less':
        hits = (sample_diffs <= obs_diff).sum()
    elif alternative == 'greater':
        hits = (sample_diffs >= obs_diff).sum()
    
    p_value = hits / num_trials
    end = time()
    
    if verbose:
        print('estimated p-value for the test: {:.4f} ({:.2f}%)'\
              .format(p_value, p_value*100))
        print('elapsed time: {:.3f}s'.format(end-begin))
    
    return p_value

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_ranked_sum(x, 
                  y,
                  z=pd.Series([]),
                  alternative='two-sided',
                  plot=False,
                  verbose=False):
    '''This function returns a p-value for a ranked-sum test. It is presumed 
    that there are no ties.
    
    Source: Udacity Data Science Course - Lesson 5 - Statistical Considerations
    into testing - Class 10 - Forth Notebook - More Non-Parametric Tests - 
    Mann-Whitney.
    
    Inputs:
      - x (mandatory) - a vector of numpy Float, as the first entry
      - y (mandatory)  - a vector of numpy Float, as the second entry
      - z (optional) - a vector dimension (data['time'], for plotting the graph
        - (default=empty dataset) - you don´t need to inform, if you don´t intend
        to show the histograms graph
      - alternative (optional) - the test to be performed (options:'two-sided', 
        'less', 'greater') (default='two-sided')
      - plot (optional) - if you want to plot the distribution - (Boolean, 
        default=False)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - an estimative for p-value for the ranked test
    '''
    if verbose:
        print('###function ranked sum started for ', alternative)
        
    begin=time()
    
    if plot:
        if verbose:
            print('first showing histogram graphs for both condition 0 and condition 1')
        if z.empty:
            if verbose:
                print('cannot display the graph, z parameter is missing')            
        else:
            plot = fn_plot(first_graph=x, 
                           second_graph=y,
                           aux=z,
                           type='2hist',
                           verbose=verbose)      
    
    #definining initial u as 0
    u = 0
    for i in x:
        wins = (i > y).sum()
        ties = (i == y).sum()
        u += wins + 0.5 * ties
    
    #computing z-score
    num_1 = x.shape[0]
    num_2 = y.shape[0]
    mean_u = num_1 * num_2 / 2
    stdev_u = np.sqrt( num_1 * num_2 * (num_1 + num_2 + 1) / 12 )
    z = (u - mean_u) / stdev_u
    
    #rules for the p-value, according to the test
    if alternative == 'two-sided':
        p_value = 2 * stats.norm.cdf(-np.abs(z))
    if alternative == 'less':
        p_value = stats.norm.cdf(z)
    elif alternative == 'greater':
        p_value = stats.norm.cdf(-z)
        
    end = time()
    
    if verbose:
        print('estimated p-value for the ranked sum test: {:.4f} ({:.2f}%)'\
              .format(p_value, p_value*100))
        print('elapsed time: {:.4f}s'.format(end-begin))
    
    return p_value

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_sign_test(x, 
                 y,
                 z=pd.Series([]),
                 alternative='two-sided',
                 plot=False,
                 verbose=False):
    '''This function returns a p-value for a sign test. It is presumed
    that there are no ties.
    
    Source: Udacity Data Science Course - Lesson 4 - Statistical Considerations
    into testing - Class 10 - Forth Notebook - More Non-Parametric Tests.
    
    Inputs:
      - x (mandatory) - a vector of numpy Float, as the first entry
      - y (mandatory) - a vector of numpy Float, as the second entry
      - z (optional) - a vector dimension (data['time'], for plotting the graph
        - (default=empty dataset) - you don´t need to inform, if you don´t intend
        to show the histograms graph
      - alternative (optional, options: {'two-sided', 
        'less', 'greater'}) - the test to be performed (, default='two-sided')
        * two-sided -> to test for both tails of the normal distribution curve
        * less -> to test for the left tail of the normal distribution curve
        * greater -> to test for the right tail of the normal distribution curve
      - plot (optional) - if you want to plot the distribution - (Boolean, 
        default=False)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - an estimative for p-value for the sign test
    '''
    if verbose:
        print('###function sign test started for', alternative)
        
    begin=time()
    
    if plot:
        if verbose:
            print('first showing signal test plots for both conditions')
        if z.empty:
            if verbose:
                print('cannot display the graph, z parameter is missing')            
        else:
            plot = fn_plot(first_graph=x,
                           second_graph=y,
                           aux=z,
                           type='stest',
                           verbose=True)      
   
    # compute parameters
    num = x.shape[0] - (x == y).sum()
    k = (x > y).sum() - (x == y).sum()

    # compute a p-value
    if alternative == 'two-sided':
        p_value = min(1, 2 * stats.binom(num, 0.5).cdf(min(k, num-k))) #cdf is cumulative distribution function
    if alternative == 'less':
        p_value = stats.binom(num, 0.5).cdf(k)
    elif alternative == 'greater':
        p_value = stats.binom(num, 0.5).cdf(num-k)

    end = time()
    
    if verbose:
        print('estimated p_value for sign test: {:.4f} ({:.2f}%)'.format(p_value, p_value*100))
        print('elapsed time: {:.4f}s'.format(end-begin))
   
    return p_value

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_variant_analytic(df,
                        p=0.5,
                        value=0.0,
                        verbose=False):
    '''This function checks a variant metric by analytic approach. One example
    of a variant is if the migration to a new webpage format generates more
    sales.
    
    Source: Udacity Data Science Course - Lesson 4 - Statistical Considerations 
    into testing - Class 3 - First Notebook - Statistical Significance.
    
    Inputs:
      - df (mandatory) - dataset containing the binary data to be checked
      - p (optional) - probability (default=50%)
      - value (optional) - central value (default=0)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
    '''
    if verbose:
        print ('###function for variant check started - analytic approach')
        
    begin = time()
    
    #number of trials
    num_observation = df.shape[0]
    num_condition = df.groupby('condition').size()
    if verbose:
        print('{} observations: {} on page H0 (control) and {} on page H1 (experiment)'\
              .format(num_observation, num_condition[0], num_condition[1]))

    #means on metric
    num_clicks= df[df['click'] == 1].groupby('condition').size()
    p_click = df.groupby('condition').mean()['click']
    diff = (((p_click[1] - p_click[0]) / p_click[0])*100)
    if verbose:
        print('user clicked on buy: {} ({:.1f}%) page H0 and {} ({:.1f}%) page H1'\
              .format(num_clicks[0], p_click[0]*100, num_clicks[1], p_click[1]*100))
        print('  - relative difference for page H1: {:.1f}%'.format(diff))
        
    #H0 -> trials & overall 'positive' rate under H0
    n_control = df.groupby('condition').size()[0]
    n_exper = df.groupby('condition').size()[1]
    p_null = df['click'].mean()

    #standard error
    std_error = np.sqrt(p_null * (1-p_null) * (1/n_control + 1/n_exper))
    if verbose:
        print('Standard Error: {:.1f}'.format(std_error))

    #z-score and p-value
    z_score = (p_click[1] - p_click[0]) / std_error
    if verbose:
        print('z-score: {:.4f}'.format(z_score))

    p_value = 1-stats.norm.cdf(z_score)
    
    end = time()
    
    if verbose:
        print()
        print('analytic p-value: {:.4f} ({:.2f}%)'.format(p_value, p_value*100))
        print('*a low value means that H0 don´t have a good explanatory power')
        print('elapsed time: {:.4f}s'.format(end-begin))
        
    return True
              
#########1#########2#########3#########4#########5#########6#########7#########8
def fn_variant_simulated(df,
                         p=0.5,
                         num_trials=200_000,
                         verbose=False):
    '''This function checks a variant metric by simulation approach.
    
    Source: Udacity Data Science Course - Lesson 4 - Statistical Considerations 
    into testing - Class 3 - First Notebook - Statistical Significance.
    
    Inputs:
      - df (mandatory) - dataset containing the binary data to be checked
      - p (optional) - probability (default=50%)
      - num_trials - number of trials for randomly simulating the experiment
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - It is a report function only! Returns True if everything runned well
    '''
    if verbose:
        print ('###function for variant check started - simulation approach')
        
    begin = time()
    
    #number of trials
    num_observation = df.shape[0]
    num_condition = df.groupby('condition').size()
    num_control = num_condition[0]
    num_experiment = num_condition[1]
    if verbose:
        print('{} observations: {} on page H0 (control) and {} on page H1 (experiment)'\
              .format(num_observation, num_control, num_experiment))
        
    #'positive' rate under null
    p_null = df['click'].mean()
    #if verbose:
    #   print('p-Null: {:.4f}'.format(p_null))
        
    #means on metric
    num_clicks= df[df['click'] == 1].groupby('condition').size()
    p_click = df.groupby('condition').mean()['click']
    diff = (((p_click[1] - p_click[0]) / p_click[0])*100)
    if verbose:
        print('user clicked on buy: {} ({:.1f}%) page H0 and {} ({:.1f}%) page H1'\
              .format(num_clicks[0], p_click[0]*100, num_clicks[1], p_click[1]*100))
        print('  - relative difference for page H1: {:.1f}%'.format(diff))

    #simulate outcomes under null, compare to observed outcome
    ctrl_clicks = np.random.binomial(num_control, p_null, num_trials)
    exp_clicks = np.random.binomial(num_experiment, p_null, num_trials)
    samples = exp_clicks / num_experiment - ctrl_clicks / num_control
    
    if verbose:
        print()
        print('*simulation part')
        print('simulated clicks on H0 (control): {} and on H1 (experiment): {}'\
              .format(len(ctrl_clicks), len(exp_clicks)))
        print('samples:', len(samples))
        
    #simulated p-value:
    p_value = (samples >= (p_click[1] - p_click[0])).mean()

    #if verbose:
    #    print('p-value: {:.4f} ({:.2f}%)'.format(p_value, p_value*100)) 

    end = time()
    
    if verbose:
        print()
        print('simulated p_value: {:.4f} ({:.2f}%)'\
              .format(p_value, p_value*100))
        print('*a low value means that H0 don´t have a good explanatory power')
        print('elapsed time: {:.4f}s'.format(end-begin))
        
    return True

###General purpose functions####################################################
#########1#########2#########3#########4#########5#########6#########7#########8
def fn_onsquare(x, 
                y, 
                verbose=False):
    '''This function calculates correlation based on on basis O(n^2).
    it fits for Kendall´s Tau.
    Inputs:
      - x vector for the first variable (numpy Array)
      - x vector for the second variable (numpy Array)
    Output:
      - a O(nLog(n)) computing time correlation.
    '''
    if verbose:
        print('*O(n^2) based correlation started')
    
    #initial parameters
    num = len(x) 
    sum_val = 0

    #loop calculating for mean values 
    for i, (x_i, y_i) in enumerate(zip(x, y)):        
        for j, (x_j, y_j) in enumerate(zip(x, y)):
            if i < j:
                sum_val += np.sign(x_i - x_j) * np.sign(y_i - y_j)
                        
    correlation = 2 * sum_val / (num * (num - 1))

    return correlation

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_ologn(x, 
             y, 
             verbose=False):
    '''This function calculates correlation based on on basis O(nLog(n)).
    it fits for Pearson and Spearman.
    Inputs:
      - x vector for the first variable (numpy Array)
      - x vector for the second variable (numpy Array)
    Output:
      - a O(nLog(n)) computing time correlation.
    '''
    if verbose:
        print('*O(nLog(n)) based correlation started')
    
    #calculating 
    x_diff = x - np.sum(x) / len(x) #x - nean x
    y_diff = y - np.sum(y) / len(y) #y - mean y
    cov_xy = np.sum(x_diff * y_diff)
    var_x = np.sqrt(np.sum(x_diff ** 2))
    var_y = np.sqrt(np.sum(y_diff ** 2))
    
    correlation = cov_xy / (var_x * var_y)
    
    return correlation

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_plot(first_graph, 
            second_graph=pd.Series([]),
            aux=pd.Series([]),
            type='none',
            verbose=False):
    '''This is a general function for plotting and embeleshing graphs. It was
    first designed fot the distribution for Power, and then generallized for
    other graphs on the scope of this project.
    
    Inputs:
      - first_graph (mandatory) - data (minimal) for the graph 
      - second_graph (optional) - (default=empty dataset)
      - aux (optional) - (default=empty dataset)
      - type (optional) - possible: {'none', 'htest', 'hist', '2hist', 'stest'}
        (default='none')
        * htest -> two plots, H0 and H1, in one axis, with legends
          x = p-values
          y = sum probabilities for each graph
          + vertical line for p_crit value for H0 explanatory power
          + fills for showing the explanatory power
          first_graph - H0 series of data (normal continous random variable)
          contains y_null for shaping H0
          second_graph - H1 series of data (normal continous random variable)
          contains y_alt for shaping H1
          aux - p_crit (1-alpha) - the critical point thal over it, H0 does not
          explain the phenomena 
        * hist -> one histogram in one axis, no legends
          x = time
          y = counts for each bin, given n bins (automatically calculated)
          first_graph - series of time-values 
          second_graph - not used
          aux - not used
        * 2hist -> two histograms (control, experiment) in one axis, with legends
          x = time
          y = counts for each bin, for two partially superposed histograms
          first_graph - control data
          second_graph - experiment data
          aux -
        * stest -> two plots (control, experiment) in one axis, with legends
          x = time (days of experiment)
          y = success rate for each experiment
          first_graph - success rate for control
          second_graph - success rate for experiment
          aux - series of days when the experiment was running
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output - True, if everything goes well - this is a plot function only!
    '''
    if verbose:
        print('###function plot started')        
    
    #making the plot
    mstyles.use('seaborn-darkgrid')#ggplot') #dark_background')
    fig_zgen = mpyplots.figure() #creating the object    
    axis_zgen = fig_zgen.add_axes([0,0,1,1]) #X0 y0 width height
    
    if type == 'htest': #histogram for h0 h1 test
        #assertions
        #assert second_graph exists
        assert aux > 0. #aux receives p_critical
        if verbose:
            print('*plotting hypothesis test')
        #preprocessing
        low_bound = first_graph.ppf(.01) #null hypothesis distribution
        high_bound = second_graph.ppf(.99) #alternative hypothesis distribution
        x = np.linspace(low_bound, high_bound, 201)
        y_null = first_graph.pdf(x) #null
        y_alt = second_graph.pdf(x) #alternative
        #plotting
        axis_zgen.plot(x, y_null)
        axis_zgen.plot(x, y_alt)
        axis_zgen.vlines(aux, 
                         0, 
                         np.amax([first_graph.pdf(aux), second_graph.pdf(aux)]),
                         linestyles = '--', color='red')
        axis_zgen.fill_between(x, y_null, 0, where = (x >= aux), alpha=0.5)
        axis_zgen.fill_between(x, y_alt , 0, where = (x <= aux), alpha=0.5)
        axis_zgen.legend(labels=['null hypothesis','alternative hypothesis'], fontsize=12)
        title = 'Hypothesis Test'
        x_label = 'difference'
        y_label = 'density'
        
    elif type == 'hist': #time count histogram
        #assertions
        #asserts second graph is False
        #asserts aux is False
        if verbose:
            print('*showing data histogram')
        n_bins = np.arange(0, first_graph.max()+400, 400)
        mpyplots.hist(first_graph, 
                      bins = n_bins)
        title = 'Time Histogram'
        x_label = 'time'
        y_label = 'counts'
    
    elif type == '2hist':
        #assertions
        #assert second_graph
        #assert aux == data['time'] #aux receives data['time']
        counts1 = first_graph
        counts2 = second_graph
        if verbose:
            print('*showing test (control and experiment) histograms')
        #plotting
        borders = np.arange(0, aux.max()+400, 400)
        mpyplots.hist(counts1, alpha=0.5, bins=borders)
        mpyplots.hist(counts2, alpha=0.5, bins=borders)
        axis_zgen.legend(labels=['control', 'experiment'], fontsize=12)
        title = 'Time Histogram'
        x_label = 'time'
        y_label = 'counts'
        
    elif type == 'stest':
        #assertions
        #assert second_graph
        #assert aux == data['day'] 
        if verbose:
            print('*plotting signal test (control and experiment) graphs')
        #preprocessing
        x=aux
        y_control=first_graph
        y_experiment=second_graph
        #plotting
        axis_zgen.plot(x, y_control)
        axis_zgen.plot(x, y_experiment)      
        axis_zgen.legend(labels=['control', 'experiment'], fontsize=12)
        title = 'Signal Test'
        x_label = 'day of experiment'
        y_label = 'success rate'
        
    elif type == 'none':
        if verbose:
            print('*no graph type has been choosen, nothing was plotted')
        return False
       
    else:
        raise Exception('type of graph invalid or not informed')
    
    fig_zgen.suptitle(title, fontsize=14, fontweight='bold')
    mpyplots.xlabel(x_label, fontsize=14)
    mpyplots.ylabel(y_label, fontsize=14)
    mpyplots.show()
    
    return True

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_read_data(filepath,
                 index=False,
                 index_col='id',
                 remove_noisy_cols=False,
                 dtype=None,
                 verbose=False):
    '''This function reads a .csv file. It was first designed for the first
    project of this course and then generallized for other imports. The idea
    is to make the first dataset testing and modification on it.
    
    Inputs:
      - filepath (mandatory) - String containing the full path for the data to
        oppened
      - index_col (optional) - String containing the name of the index column
        (default='id')
      - remove_noisy_cols - if you want to remove noisy (blank) columns
        (default=False)
      - verbose (optional) - if you needed some verbosity, turn it on - Boolean 
        (default=False)
    Output:
      - Pandas Dataframe with the data
    *this function came from my library udacourse2.py and was adapted for this
    project
    '''
    if verbose:
        print('###function read data from .csv file started')
        
    begin=time()
    
    #reading the file
    df = pd.read_csv(filepath_or_buffer=filepath,
                     dtype=dtype)
    if remove_noisy_cols:
        del df['Unnamed: 0']
    if index:
        try:
            df = df.set_index(index_col)
            if verbose:
                print('index name set as', index_col)
        except KeyError:
            df.index_name = index_col
    else:
        df = df.reset_index(drop=True)
        if verbose:
            print('index reset and drop')
    
    #if verbose:
    #    print('file readed as Dataframe')

    #testing if Dataframe exists
    #https://stackoverflow.com/questions/39337115/testing-if-a-pandas-dataframe-exists/39338381
    if df is not None: 
        if verbose:
            print('dataframe created from {} has {} lines and {} columns'\
                  .format(filepath, df.shape[0], df.shape[1]))
            #print(df.head(5))
    else:
        raise Exception('something went wrong when acessing .csv file', filepath)
    
    #setting a name for the dataframe (I will cound need to use it later!)
    ###https://stackoverflow.com/questions/18022845/pandas-index-column-title-or-name?rq=1
    #last_one = filepath.rfind('/')
    #if last_one == -1: #cut only .csv extension
    #    df_name = filepath[: -4] 
    #else: #cut both tails
    #    df_name = full_path[last_one+1: -4]   
    #df.index.name = df_name
    #if verbose:
    #    print('dataframe index name set as', df_name)
              
    end = time()
    
    if verbose:
        print('elapsed time: {:.4f}s'.format(end-begin))
              
    return df

###DEPRECATED###################################################################
#########1#########2#########3#########4#########5#########6#########7#########8
def fn_movie_liked(item, 
                   min_rating=7,
                   sort=True,
                   verbose=False):
    '''This function takes all the items for one user and return the best rated
    items.
    
    Source: Udacity Data Science Course - Lesson 6  - Identifying Reccomendations
    Forth Notebook - Class 17 - Collaborative Filtering - Recommendations with 
    MovieTweetings - Collaborative Filtering

    Inputs:
      - item (mandatory) - a dataset filtered for an individual user.
        e.g. (for user 66): user_item['user_id'] == 66  
      - min_rating (optional) - the trigger point to consider an item to be 
        considered "nice" for an user (integer, default=7)
      - sort (optional) - if you want to show the most rated items first
        (Boolean, default=True)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - movie_liked - an array of movies the user has watched and liked
    '''
    raise Exception('function deprecated, use fn_movie_liked2')
    
    try:
        user = item.iloc[0]
    except TypeError:
        if verbose:
            print('you must inform a filtered dataset, see documentation')
        return False
    except IndexError:
        if verbose:
            print('an empty dataset was informed, this user does not exist')
        return False

    if verbose:
        print('###function movies liked started for user', user)

    begin = time()
    movie_liked = item[item['rating'] > 7]
    
    if sort:
        if verbose:
            print('*sorting the list - best rated first')
        movie_liked = movie_liked.sort_values(by='rating', ascending=False)

    movie_liked = np.array(movie_liked['movie_id'])    
    end = time()
    
    if verbose:
        print('elapsed time: {:.4f}s'.format(end-begin))
    
    return movie_liked

#########1#########2#########3#########4#########5#########6#########7#########8
#for reloading this library on Jupyter, just run this lines in a code cell:
#from importlib import reload 
#import udacourse3
#udacourse3 = reload(udacourse3)

#source
#https://stackoverflow.com/questions/27779677/how-to-format-elapsed-time-from-seconds-to-hours-minutes-seconds-and-milliseco
#hours, rem = divmod(end-start, 3600)
#minutes, seconds = divmod(rem, 60)
#print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

#########1#########2#########3#########4#########5#########6#########7#########8
def Xfn_validation_comparison(df_train,
                             user_matrix,
                             movie_matrix,
                             val_df, 
                             num_pred,
                             relatory=True,
                             verbose=False):
    if verbose:
        print('###function validation comparison started')

    begin = time()
    val_user = np.array(val_df['user_id'])
    val_movie = np.array(val_df['movie_id'])
    val_rating = np.array(val_df['rating'])
    
    for idx in range(num_pred):
        #call for our subfunction
        #if verbose:
        #    print('*calling predict rating subfunction')
        pred = fn_predict_rating(df_train=df_train,
                                 user_matrix=user_matrix, 
                                 movie_matrix=movie_matrix, 
                                 user_id=val_user[idx],
                                 movie_id=val_movie[idx],
                                 verbose=False)
        if not pred:
            if verbose:
                print('system halted')
            return False
        
        print('Our rating actual: {} → for user {} on movie {}'\
              .format(val_rating[idx],
                      val_user[idx], 
                      val_movie[idx]))
        print('         predited: {}'.format(round(pred)))
        
    #end = time()
    #if verbose:
    #    print('elapsed time: {}s'.format(end-begin))
    return True
