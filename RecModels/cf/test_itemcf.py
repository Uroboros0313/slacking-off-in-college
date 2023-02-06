from sklearn.model_selection import train_test_split

from utils import read_dataframe
from model import ItemCF

DATA_FILE = '../data/ml-100k/u.data'
TRAIN_RATIO = 0.7
TEST_RATIO = 0.3
SEED = 2022
N = 10          
PREC = 0
REC = 0
COVER = 0
REC_COLS = ['user', 'item', 'rating', 'timestamp']   



df = read_dataframe(DATA_FILE, sep='\t', names=REC_COLS)
del df['timestamp']
df.drop_duplicates(subset=['user', 'item'], keep='first', inplace=True)
train, test =\
    train_test_split(df, test_size=TEST_RATIO, random_state=SEED, shuffle=True)
    

if __name__ == '__main__':
    recommender = ItemCF(train)
    recommender.init_data()
    recommender.cal_similarity()
    test_users = test['user'].unique()
    all_res = recommender.cal_recommend(test_users)