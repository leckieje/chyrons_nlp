# chyron archive web address: https://archive.org/services/third-eye.php
import pandas as pd 
import numpy as np
import datetime

# pull chyrons from web as dataframes
def get_dfs(start_y=2017, start_m=9, start_d=7, end_today=True, end_y=None, end_m=None, end_d=None):
    
    first = datetime.date(start_y, start_m, start_d)
    delta = datetime.timedelta(weeks=1)
    last = first + delta
    dfs = []
    iters = 0
    
    if end_today:
        stop = datetime.date.today()
    else:
        stop = datetime.date(end_y, end_m, end_d)
    
    while first < stop:
        if iters > 210:
            break
        else:
            print(iters)
            first_str = first.strftime('%m/%d/%y')
            last_str = last.strftime('%m/%d/%y')
            url = f'https://archive.org/services/third-eye.php?dayL={first_str}&dayR={last_str}'
            df = pd.read_csv(url, delimiter='\t')
            dfs.append(df)

            first = last + datetime.timedelta(days=1)
            last = first + delta
            
            if last > stop:
                last = stop
                
            iters += 1
        
    return dfs

# cleans dfs
def clean_dfs(df, csv=True, concat=False, csv_name='chyron_next'):
    
    # stack dfs from list
    if concat:
        df = pd.concat(df, ignore_index=True)
        
    # drop duplicates
    df.drop_duplicates(subset=['text'], keep='first', inplace=True, ignore_index=True)
    
    # count words in chyron 
    df['num_words'] = df['text'].apply(lambda x: len(x.split()) if isinstance(x, str) else 'NaN')
    
    # save to local csv          
    if csv:
        df.to_csv(f'/Users/jonleckie/Desktop/DSI_all/capstones/capstone_two/chyrons/{csv_name}.csv')    
    
    return df
