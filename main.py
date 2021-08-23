from newsapi import NewsApiClient
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import warnings
warnings.filterwarnings("ignore")

api = NewsApiClient(api_key='###API-KEY####')
res = api.get_everything(q='Russia', language='en', to="2021-08-21", from_param="2021-07-21",page=5)
full_data = pd.io.json.json_normalize(res,'articles')

content = ""
#Объединим все заголовки
for object in full_data["content"]:
    content += str(object).lower()

#скачаем стоп слова
stopWords = stopwords.words('english')

#Добавим слово chars, так как оно присутствует в ответе, и в данный момент служебное
stopWords.append('chars')

#Проведем токенезацию слов
wordTokens = word_tokenize(content)

#Создадим список без стоп-слов
clear_text = []
for word in wordTokens:
    if word not in stopWords:
        clear_text.append(word)

#Уберем слова, короче 2х символов, врят-ли нам нужны всякие предлоги и т.д.
finall_clear_text = []

for word in clear_text:
    if len(word) > 2:
        finall_clear_text.append(word)

#Составляем словарь запросов
fdist = FreqDist(finall_clear_text)

#Берем топ 50 слов
top_50 = fdist.most_common(50)

wordCloud = WordCloud(width=900,height=500, max_words=50, background_color="white", relative_scaling=1,normalize_plurals=False).generate_from_frequencies(fdist)

plt.imshow(wordCloud, interpolation='bilinear')
plt.axis("off")
plt.show()
