import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import json
import pandas as pd
import Task2
import Task1

London_tweet = []
for line in open('geoLondonJan', 'r', encoding="utf-8"):
    line_json = json.loads(line)
    long = line_json['coordinates']['coordinates'][0]
    lat = line_json['coordinates']['coordinates'][1]
    text = line_json['text']
    new_dict = {'location': [lat, long], 'text': text}
    London_tweet.append(new_dict)

l = len(London_tweet)
tweet_list = [London_tweet[i]['text'] for i in range(l)]


model = Task2.model
n = model.tweetlist_Scoring(tweet_list)
print("length: ", len(n))


for i in range(l):
    London_tweet[i]['Score'] = n[i]

## the location with low quality
lq = [London_tweet[i] for i in range(l) if London_tweet[i]['Score'] < 0]
print("numbers of low quality: ", len(lq))

## the location with high quality
hq = [London_tweet[i] for i in range(l) if London_tweet[i]['Score'] > 0]
print("numbers of high quality: ", len(hq))

## the number of tweets with low quality
lq_dict_coor = {}
for i in range(len(lq)):
    row = np.floor((London_tweet[i]['location'][0] - 51.261318) / Task1.latOffset)
    col = np.floor((London_tweet[i]['location'][1] + 0.563) / Task1.longOffset)
    lq_dict_coor[(row, col)] = lq_dict_coor.get((row, col), 0) + 1

print(max(lq_dict_coor))

lq_df_coor = pd.DataFrame(0, columns=list(range(59)), index=list(range(48)))
for key, value in lq_dict_coor.items():
    lq_df_coor.iloc[int(key[0]), int(key[1])] = value
### heatmap
#### the max range of the heatmap is 50
sns.set_context({"figure.figsize":(10,8)})
sns.heatmap(data=lq_df_coor,square=True, cmap="Oranges", vmax = 50)
plt.title("Location heatmap for low quality tweets")
plt.savefig("Task3-lq_heatmap_50.png")
plt.show()
#### no max range
sns.set_context({"figure.figsize":(10,8)})
sns.heatmap(data=lq_df_coor,square=True, cmap="Oranges")
plt.title("Location heatmap for low quality tweets")
plt.savefig("Task3-lq_heatmap.png")
plt.show()
### histagram
fig, axis = plt.subplots(figsize =(10, 5))
axis.hist(list(lq_dict_coor.values()), bins = list(range(0,350,5)))
plt.xlabel("Number of tweets")
plt.ylabel("Number of grids")
plt.title("Distribution of low quality tweet numbers")
plt.savefig("Task3-lq_histogram.png")
plt.show()

## the number of tweets with high quality
hq_dict_coor = {}
for i in range(len(hq)):
    row = np.floor((London_tweet[i]['location'][0] - 51.261318) / Task1.latOffset)
    col = np.floor((London_tweet[i]['location'][1] + 0.563) / Task1.longOffset)
    hq_dict_coor[(row, col)] = hq_dict_coor.get((row, col), 0) + 1

print(max(hq_dict_coor))

hq_df_coor = pd.DataFrame(0, columns=list(range(59)), index=list(range(48)))
for key, value in hq_dict_coor.items():
    hq_df_coor.iloc[int(key[0]), int(key[1])] = value
### heatmap
#### the max range of the heatmap is 50
sns.set_context({"figure.figsize":(10,8)})
sns.heatmap(data=hq_df_coor,square=True, cmap="Oranges", vmax = 50)
plt.title("Location heatmap for high quality tweets")
plt.savefig("Task3-hq_heatmap_50.png")
plt.show()
#### no max range
sns.set_context({"figure.figsize":(10,8)})
sns.heatmap(data=hq_df_coor,square=True, cmap="Oranges")
plt.title("Location heatmap for high quality tweets")
plt.savefig("Task3-hq_heatmap.png")
plt.show()
### histogram
fig, axis = plt.subplots(figsize =(10, 5))
axis.hist(list(hq_dict_coor.values()), bins = list(range(0,300,5)))
plt.xlabel("Number of tweets")
plt.ylabel("Number of grids")
plt.title("Distribution of high quality tweet numbers")
plt.savefig("Task3-hq_histogram.png")
plt.show()
# ### wordcloud
file_name1 = "Task3_hq_wordcloud.png"
hq_list = [item['text'] for item in hq for key in item]
hq_dict = Task2.Filter(hq_list,Task2.stopwords).word_dict()
Task2.wordcloud(hq_dict, file_name1)

file_name2 = "Task3_lq_wordcloud.png"
lq_list = [item['text'] for item in lq for key in item]
lq_dict = Task2.Filter(lq_list,Task2.stopwords).word_dict()
Task2.wordcloud(lq_dict, file_name2)