from datetime import date, datetime, timedelta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
import tweepy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from yahoo_fin.stock_info import get_data


# =============================================================================
# # =============================================================================
# # Przygotowanie danych
# # =============================================================================
# =============================================================================

# =============================================================================
# Pobranie tweetow Elona Muska
# =============================================================================

BEAER_TOKEN="..."

client = tweepy.Client(bearer_token=BEAER_TOKEN)

# ID konta Elona Muska
client.get_user(username="elonmusk")


# Pobranie tweetow
tweets = []
for tweet in tweepy.Paginator(client.get_users_tweets, id='44196397',tweet_fields=["created_at,conversation_id"], start_time = "2021-09-01T00:00:00.00Z",
                                    max_results=100).flatten(limit=6400):
    tweets.append([tweet['id'],tweet["created_at"],tweet['text'],tweet['conversation_id']])


tweets_and_replies = pd.DataFrame(tweets)
tweets_and_replies.rename(columns = {0:'Id', 1:'Date', 2:'Text', 3:'ConversationID'}, inplace = True)
tweets_and_replies.to_csv(r"...\tweets_and_replies.csv")

# =============================================================================
# Przygotowanie tweetow
# =============================================================================

Musk_tweets = pd.read_csv(r"...\tweets_and_replies.csv")
Musk_tweets = Musk_tweets.rename(columns = {"Date":"datetime"})
Musk_tweets['date'] = pd.to_datetime(Musk_tweets['datetime']).dt.date
Musk_tweets = Musk_tweets.drop_duplicates()

# Ograniczenie zbioru do konkretnych dat
Musk_tweets = Musk_tweets[(Musk_tweets['datetime'] >= '2021-09-15') & (Musk_tweets['datetime'] < '2022-09-15')]


# Ujednolicenie stref czasowych między danymi z Yahoo (UTC-4) oraz z Twittera (UTC+0) 
Musk_tweets['hour'] =  pd.to_datetime(Musk_tweets['datetime']).dt.hour
Musk_tweets['month'] =  pd.to_datetime(Musk_tweets['datetime']).dt.month
Musk_tweets['date_stock'] = np.where(Musk_tweets['hour']<20, Musk_tweets['date'], Musk_tweets['date'] - timedelta(days=-1))


# Usunięcie zbędnych znaków, emoji oraz wyrazów
def remove_links(tweet):
    '''Takes a string and removes web links from it'''
    tweet = re.sub(r'http\S+', '', tweet) 
    tweet = re.sub(r'bit.ly/\S+', '', tweet) 
    tweet = tweet.strip('[link]')
    return tweet


def remove_users(tweet):
    '''Takes a string and removes retweet and @user information'''
    tweet = re.sub('(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet) 
    tweet = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)
    return tweet


my_stopwords = nltk.corpus.stopwords.words('english')
word_rooter = nltk.stem.snowball.PorterStemmer(ignore_stopwords=False).stem
lemmatizer = WordNetLemmatizer()
my_punctuation = '!"$%&\'()*+,-,-./:;<=>?[\\]^_`{|}~•“”’…@#'
emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                u"\U0001f926-\U0001f937"
                u'\U00010000-\U0010ffff'
                u"\u200d"
                u"\u2640-\u2642"
                u"\u2600-\u2B55"
                u"\u23cf"
                u"\u23e9"
                u"\u231a"
                u"\u3030"
                u"\ufe0f"
    "]+", flags=re.UNICODE)


def clean_tweet(tweet, bigrams=False, token = False, lemma = False):
    tweet = remove_users(tweet)
    tweet = remove_links(tweet)
    tweet = tweet.lower() 
    tweet = re.sub('['+my_punctuation + ']+', ' ', tweet) 
    tweet = re.sub('\s+', ' ', tweet) 
    tweet = re.sub('([0-9]+)', '', tweet) 
    tweet = re.sub('amp','',tweet)
    tweet = emoji_pattern.sub(r'', tweet) 
    tweet_token_list = [word for word in tweet.split(' ')
                            if word not in my_stopwords] 

    if token:
        tweet_token_list = [word_rooter(word) if '#' not in word else word
                            for word in tweet_token_list]
    if lemma:
        tweet_token_list = [lemmatizer.lemmatize(word) if '#' not in word else word
                            for word in tweet_token_list]
    if bigrams:
        tweet_token_list = tweet_token_list+[tweet_token_list[i]+'_'+tweet_token_list[i+1]
                                            for i in range(len(tweet_token_list)-1)]
    tweet = ' '.join(tweet_token_list)
    return tweet


# Uzyskanie oczyszczonych tweetow
Musk_tweets['clean_tweet'] = Musk_tweets.Text.apply(clean_tweet)
Musk_tweets['clean_tweet_token'] = Musk_tweets.Text.apply(clean_tweet,token=True)
Musk_tweets['clean_tweet_lemma'] = Musk_tweets.Text.apply(clean_tweet,lemma=True)


# Uzyskanie wartosci VADER oznaczającej nacechowanie emocjonalne tweetow
analyser = SentimentIntensityAnalyzer()
for j in ['clean_tweet_lemma']:
    i=0 
    compval = [ ] 
    
    while (i<len(Musk_tweets)):
        k = analyser.polarity_scores(Musk_tweets.iloc[i][j])
        compval.append(k['compound'])
        i = i+1
        
    compval = np.array(compval)
    Musk_tweets['VADER_text'] = compval

Musk_tweets['date'] = pd.to_datetime(Musk_tweets['datetime']).dt.date


# Stworzenie backupu, ponieważ były testowane różne podejscia
Musk_tweets_bck = Musk_tweets
#Musk_tweets = Musk_tweets_bck
#Musk_tweets = Musk_tweets[(Musk_tweets['VADER_text']>=0.4) | (Musk_tweets['VADER_text']<=-0.4)]

#Musk_tweets = Musk_tweets[(Musk_tweets['VADER_text']>=0.4)]

#Musk_tweets=Musk_tweets[Musk_tweets['VADER_text']!=0]

# Musk_tweets['is_tesla'] = [1 if 'tesla' in x else 0 for x in Musk_tweets['clean_tweet_lemma']]
# Musk_tweets = Musk_tweets[Musk_tweets['is_tesla']==1]


# Uzyskanie wartosci nacechowania emocjonalnego per dzień
Musk_tweets_gr = Musk_tweets[['date_stock','VADER_text']]
Musk_tweets_gr.rename(columns = {'date_stock':'date'}, inplace = True)

Musk_tweets_gr = Musk_tweets_gr.groupby('date').agg({'date':'size','VADER_text':'mean'}).rename(
                columns={'date':'tweet_count','VADER_text':'VADER_mean'}).reset_index()


# =============================================================================
# Przygotowanie danych giełdowych pobranych z Yahoo finance
# =============================================================================


tesla_fin = get_data("tsla", start_date="09/15/2021", end_date="09/15/2022", index_as_date=True, interval="1d")
#tesla_fin.to_csv(r"...\tesla_fin.csv")
#tesla_fin = pd.read_csv(r"...\tesla_fin.csv")

# Uzupełnienie brakujących obserwacji (weekendy, swieta)
def stock_fill(df):
    dates = pd.date_range(start=df.index.min(), end=df.index.max())
    df = df.reindex(dates)
    df = df.interpolate()
    df['ticker'] = df.ticker.fillna(method='ffill')
    return df

tsla = stock_fill(tesla_fin)

# Utworzenie zmiennych opóźnionych oraz zmiennej zależnej
tsla['date'] = tsla.index
tsla['date'] = pd.to_datetime(tsla['date']).dt.date
tsla['close_1'] = tsla['close'].shift(1)
tsla['close_2'] = tsla['close'].shift(2)
tsla['open_1'] = tsla['open'].shift(1)
tsla['high_1'] = tsla['high'].shift(1)
tsla['low_1'] = tsla['low'].shift(1)
tsla['volume_1'] = tsla['volume'].shift(1)
tsla.drop(index=tsla.index[0], axis=0, inplace=True)
tsla['move'] = [1 if tsla.iloc[i]['close_1'] < tsla.iloc[i]['close'] else 0 for i in range(0,len(tsla))]


# Połaczenie danych z Twittera i z Yahoo finance 
df = tsla.merge(Musk_tweets_gr, on = "date")
df = df.drop(columns=['ticker'])


# Utworzenie zmiennych rolowanych
df.set_index('date', inplace=True)
df['rol_vader2'] = df['VADER_mean'].rolling(2).mean()
df['rol_vader3'] = df['VADER_mean'].rolling(3).mean()

df['rol_tweet_count2'] = df['tweet_count'].rolling(2).sum()
df['rol_tweet_count3'] = df['tweet_count'].rolling(3).sum()


# =============================================================================
# # =============================================================================
# # Analiza sentymentu
# # =============================================================================
# =============================================================================


# =============================================================================
# Wizualizacja
# =============================================================================

# Chmura słów
from wordcloud import WordCloud

#plt.figure(figsize=(10, 10)) 
long_string = ','.join(list(Musk_tweets['clean_tweet_lemma'].values))
wordcloud = WordCloud(background_color="black", max_words=100, contour_width=5,
                      contour_color='steelblue',width=600,height=400, min_word_length=2)
wordcloud.generate(long_string)
wordcloud.to_image()


# Wordcloud tylko pozytywne tweety
musk_tweets_positive = Musk_tweets[['clean_tweet_lemma','VADER_text']][Musk_tweets['VADER_text']>=0.4]
long_string = ','.join(list(musk_tweets_positive['clean_tweet_lemma'].values))
wordcloud = WordCloud(background_color="limegreen", max_words=100, contour_width=5,
                      contour_color='steelblue',width=600,height=400, min_word_length=2,colormap='PuOr')
wordcloud.generate(long_string)
wordcloud.to_file(r"...\wordcloud_pozytywne.png")
wordcloud.to_image()


# Wordcloud tylko negatywne tweety
musk_tweets_negative = Musk_tweets[['clean_tweet_lemma','VADER_text']][Musk_tweets['VADER_text']<=-0.4]
long_string = ','.join(list(musk_tweets_negative['clean_tweet_lemma'].values))
wordcloud = WordCloud(background_color="indianred", max_words=100, contour_width=5,
                      contour_color='steelblue',width=600,height=400, min_word_length=2,colormap='coolwarm')
wordcloud.generate(long_string)
wordcloud.to_file(r"...\wordcloud_negatywne.png")
wordcloud.to_image()

# Liczba slow 
from collections import Counter
split_it = long_string.split()
Countera = Counter(split_it)
  

# Najczesniej występujące słowa
most_occur = Countera.most_common(24)
print(most_occur)


# Chmura słów w kształcie logo Twittera
from imageio import imread

twitter_mask = imread(r"...\twitter_mask.png")

wordcloud = WordCloud(
                    background_color='white',
                    width=800,
                    height=500,
                    mask=twitter_mask,
                    contour_color='deepskyblue',
                    min_word_length=2,colormap='Blues_r').generate(long_string)
plt.figure(figsize=(10, 10)) 
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig(r"...\my_twitter_wordcloud.png", dpi=1000)
plt.show()


# Podział tweetów na 3 grupy sentymentu na podstawie Vader score- tweety pozytywne, negatywne i neutralne
i = 0
v=0.4 #VADER score
predicted_value = [ ] 
df_U = Musk_tweets


while(i<len(df_U)):
    if ((df_U.iloc[i]['VADER_text'] >= v)):
        predicted_value.append('Pozytywny')
        i = i+1
    elif ((df_U.iloc[i]['VADER_text'] > -v) & (df_U.iloc[i]['VADER_text'] < v)):
        predicted_value.append('Neutralny')
        i = i+1
    elif ((df_U.iloc[i]['VADER_text'] <= -v)):
        predicted_value.append('Negatywny')
        i = i+1
        
df_U['sentiment'] = predicted_value

# Podsumowanie grup nacechowania
df_U_sen = df_U.groupby('sentiment').count()['clean_tweet_lemma'].reset_index().sort_values(by='clean_tweet_lemma',ascending=False)
df_U_sen.style.background_gradient(cmap='Purples')

df_U_sen


# Wykres liczebnosci grup nacechowania w kształcie odwróconego stożka
from plotly import graph_objs as go
import kaleido

fig = go.Figure(go.Funnelarea(
    text =df_U_sen.sentiment,
    values = df_U_sen.clean_tweet_lemma,
    #title = {"position": "top center", "text": "Rozkład sentymentu"},
    showlegend=False,
    marker = {"colors": [ "deepskyblue", "limegreen", "indianred"]},
    textfont = { "color": "black"}
    ))
fig.show()

fig.write_image(r"...\rozklad_sentymentu.png")



# =============================================================================
# Weryfikacja hashtagow, retweetow i oznaczeń
# =============================================================================

def find_retweeted(tweet):
    '''This function will extract the twitter handles of retweed people'''
    return re.findall('(?<=RT\s)(@[A-Za-z]+[A-Za-z0-9-_]+)', tweet)

def find_mentioned(tweet):
    '''This function will extract the twitter handles of people mentioned in the tweet'''
    return re.findall('(?<!RT\s)(@[A-Za-z]+[A-Za-z0-9-_]+)', tweet)  

def find_hashtags(tweet):
    '''This function will extract hashtags'''
    return re.findall('(#[A-Za-z]+[A-Za-z0-9-_]+)', tweet) 


Musk_tweets['retweeted'] = Musk_tweets.Text.apply(find_retweeted)
Musk_tweets['mentioned'] = Musk_tweets.Text.apply(find_mentioned)
Musk_tweets['hashtags'] = Musk_tweets.Text.apply(find_hashtags)



# =============================================================================
# Korelacja 
# =============================================================================

df_model_corr = df[['move','rol_vader3','rol_tweet_count3','volume_1','open_1']]
df_model_corr = df_model_corr.rename(columns={'move': 'Zachowanie giełdy', 'rol_vader3': 'Wartość sentymentu', 'rol_tweet_count3': 'Średnia liczba tweetów', 'volume_1':'Wolumen akcji', 'open_1': 'Cena otwarcia'})

fig, ax = plt.subplots(figsize=(6, 6))

corrMatrix = df_model_corr.corr()
sns.heatmap(corrMatrix, annot=False)

plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.xticks(rotation=45, ha='right')

plt.show()

# Statystyki opisowe
stat_opis = df_model_corr.describe().T

# =============================================================================
# Ogolna wizualizacja
# =============================================================================
#Liczba tweetow
Musk_tweets_no_tweets = Musk_tweets[['datetime']]
Musk_tweets_no_tweets['datetime'] = pd.to_datetime(Musk_tweets_no_tweets['datetime'])

Musk_tweets_no_tweets_gr = Musk_tweets_no_tweets.groupby(Musk_tweets_no_tweets['datetime'].dt.to_period('M')).count()

Musk_tweets_no_tweets_gr.plot(kind='bar', figsize=(6.5, 6.5), legend = None)
plt.xlabel("")
#plt.title("Liczba tweetów Elona Muska")
plt.xticks(rotation=45, ha='right')
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
plt.savefig(r"...\liczba_tweetow.png", dpi=1000)

# Cena na gieldzie
x_plot = tesla_fin.index
y_plot = tesla_fin['close']


fig, ax = plt.subplots(figsize=(10, 8))
plt.plot(x_plot, y_plot)
plt.xticks(rotation=45, ha='right')
ax.yaxis.set_major_formatter('{x:1.0f}$')
#plt.legend('Liczba tweetów', ncol=2, loc='upper left')
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
plt.show()
plt.savefig(r"...\tesla_stock.png", dpi=1000)

# =============================================================================
# # =============================================================================
# # Model
# # =============================================================================
# =============================================================================

from sklearn.model_selection import train_test_split,cross_validate, cross_val_score#, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


## Podzial na zmienne ciągłe i dyskretne, a także weryfikacja modeli tylko ze zmiennymi giełdowymi i ze zmiennymi z Twittera
## Proces weryfikacji dopasowania odpowiednich zmiennych do badanego zjawiska został pominięty

#df_t=df[df['VADER_score']!=0]
df_model = df[['move','rol_vader3','rol_tweet_count3','volume_1','open_1']]
#df_model = df[['move','volume_1','open_1']]
#df_model = df[['move','rol_vader3','rol_tweet_count3']]
df_model = df_model.dropna()
cat_var = []
num_var = ['rol_vader3','rol_tweet_count3','volume_1','open_1']
#num_var = ['volume_1','open_1']
#num_var = ['rol_vader3','rol_tweet_count3']
        
X = df_model.drop(["move"], axis=1)
y = df_model["move"]

# Normalizacja zmiennych
scaler = MinMaxScaler()
X[num_var] = scaler.fit_transform(X[num_var])


# Podzial na probę treningową i testową
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=35)

pd.options.display.float_format = '{:.4f}'.format


# Wybór modeli, miar oraz liczby k-fold do walidacji krzyżowej
models = {
    "LogisticRegression":{"model":LogisticRegression() },
    'SVM':{"model":SVC( kernel='rbf') },
    "DecisionTreeClassifier":{"model":DecisionTreeClassifier() },
    "RandomForestClassifier":{"model":RandomForestClassifier() }
    #"XGBClassifier":{"model": XGBClassifier(verbosity = 0) }
}

models_results = []
test_results=[]
scorings = ['accuracy','f1', 'roc_auc']
k = 5

# Petla, która wyswietla podsumowane wyniki dla wszystkich estymatorów
#print(f"For {k}-fold CV:\n")
for name, m in models.items():

    model = m['model']
    print(f"{name:3}:\nTrain")
    for scoring in scorings:
        result = cross_validate(model.fit(X_train,y_train), X_train,y_train, cv = k, scoring=scoring)
        
        score = result['test_score']
        mean_score=sum(score)/len(score)
        mean_fit_time = round( sum(result['fit_time']) / len(result['fit_time']), 3)

        m['mean_score'] = mean_score
        m['Training time (sec)'] = mean_fit_time
        m['scoring'] = scoring
        
        lst = [name,m['scoring'], m['mean_score'],m['Training time (sec)']]
        models_results.append(lst)
        
        #print(f" Scoring: {scoring} \n Scores: {score} \n Mean_{scoring}: {mean_score} \n Mean training time {mean_fit_time} sec\n")
        print(f" {scoring}: {round((mean_score),3)}")
    
    preds = model.predict(X_test)
    
    tr={'Model': name,'Accuracy':round(accuracy_score(y_test, preds),3)
        ,'F1 score':round(f1_score(y_test, preds),3)
        , 'ROC_AUC': round(roc_auc_score(y_test, preds),3)}
    test_results.append(tr)
    
    print( "\nTEST"
          ,"\n accuracy: ", round(accuracy_score(y_test, preds),3)
          ,"\n f1: ", round(f1_score(y_test, preds),3)
          ,"\n roc_auc:", round(roc_auc_score(y_test, preds),3) )
 
    # Wyswietlanie confusion matrix
    # cm = confusion_matrix(y_test, preds)
    # sns.heatmap(cm,annot=True, linewidths=.5)
    # plt.show()


# Wybrany model to regresja logistyczna

# =============================================================================
# Istotnosc zmiennych
# =============================================================================

# Weryfikacja modelu bardziej ekonometrycznie
import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result_sm=logit_model.fit()
print(result_sm.summary())
#odds ratio
np.exp(result_sm.params)


from statsmodels.stats.weightstats import ztest as ztest
ztest(X,y, value=0) 


# Rozne sprawdzenia
model = LogisticRegression()

# fit the model
model.fit(X, y)
preds_test = model.predict
importance = model.coef_[0]
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
    
# plot feature importance
# pyplot.bar([x for x in range(len(importance))], importance)
# pyplot.show()

odds = np.exp(model.coef_[0])
pd.DataFrame(odds, 
             X.columns, 
             columns=['coef'])\
            .sort_values(by='coef', ascending=False)


from sklearn import svm
model = svm.SVC(kernel='linear')

# fit the model
model.fit(X, y)


importance = model.coef_[0]
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))


from sklearn.inspection import permutation_importance
model = SVC()
# fit the model
model.fit(X, y)
# get importance
importance = permutation_importance(model, X, y)
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))


# Model jest ok

# =============================================================================
# Pozostałe rzeczy
# =============================================================================


# Minimalna i maksymalna cena na gieldzie w badanym okresie
tesla_fin[tesla_fin['close']==min(tesla_fin['close'])]
tesla_fin[tesla_fin['close']==max(tesla_fin['close'])]


tesla_fin['month'] =  pd.to_datetime(tesla_fin['index']).dt.month
tesla_fin_min_loc = tesla_fin.loc['2022-4-1':'2022-6-30']
tesla_fin_min_loc[tesla_fin_min_loc['close']==max(tesla_fin_min_loc['close'])]



# Tweety negatywne, które odnoszą sie do Tesli
Musk_tweets_negative = Musk_tweets[Musk_tweets['VADER_text'] <= -0.4]
Musk_tweets_negative['is_tesla'] = [1 if 'tesla' in x else 0 for x in Musk_tweets_negative['clean_tweet_lemma']]


# Tweety, które odnoszą się do Twittera
Musk_tweets_twitter = Musk_tweets
Musk_tweets_twitter['is_twitter'] = [1 if 'twitter' in x else 0 for x in Musk_tweets_twitter['clean_tweet_lemma']]
Musk_tweets_twitter = Musk_tweets_twitter[Musk_tweets_twitter['is_twitter']==1]
Musk_tweets_twitter['VADER_text'].describe()


# Histogram statystyki Vader 
plt.hist(Musk_tweets['VADER_text'],100)
plt.show() 


# Liczba tweetow, gdzie Vader = 0
Musk_tweets[Musk_tweets['VADER_text'] == 0].count()


# Weryfikacja scatterplotów
plt.scatter(df['close'], df['open'])
plt.show()
