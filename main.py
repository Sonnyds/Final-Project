import datetime

import requests
from selenium.webdriver.chrome.options import Options
import pandas as pd
from selenium import webdriver
from bs4 import BeautifulSoup as BS
import json
import os
import re
import requests_cache



types = ['action','adventure','animation','biography','comedy','crime','drama','family','fantasy','film-noir','history','horror','music','musical','mystery','romance','sci-fi','sport','thriller','war','western']
moviesPath = 'movies.xlsx'
def fetchData():
    items = []
    requests_cache.install_cache('scrape_cache', backend='filesystem')
    session = requests.Session()
    startTime = datetime.datetime.now()
    print(len(types))
    for i in range(0, len(types)):
        type = types[i]
        for j in range(0, 2):
            if (j == 0):
                # first page
                url = f'https://www.imdb.com/search/title/?title_type=feature&num_votes=25000,&genres={type}&sort=user_rating,desc&ref_=adv_prv'
            else:
                url = f'https://www.imdb.com/search/title/?title_type=feature&num_votes=25000,&genres={type}&sort=user_rating,desc&start=51&ref_=adv_nxt'
            res = session.get(url)

            soup = BS(res.text, 'lxml')
            cards = soup.findAll('div', class_='lister-item mode-advanced')
            for card in cards:
                dic = {}
                dic['name'] = card.find('span', class_='lister-item-index unbold text-primary').findNext(
                    'a').text.strip()
                dic['year'] = card.find('span', class_='lister-item-index unbold text-primary').findNext('a').findNext(
                    'span').text.strip()
                dic['scores'] = card.find('span', class_='global-sprite rating-star imdb-rating').findNext(
                    'strong').text.strip()
                certificate = card.find('span', class_='certificate')
                if (certificate != None):
                    dic['certificate'] = certificate.text.strip()
                else:
                    dic['certificate'] = ''
                dic['runtime'] = card.find('span', class_='runtime').text.strip()
                dic['genre'] = card.find('span', class_='genre').text.strip()
                dic['desciption'] = card.find('div', class_='ratings-bar').findNext('p').text.strip()
                staffs = card.find('div', class_='ratings-bar').findNext('p').findNext('p').text
                staffs = staffs.strip()
                index0 = staffs.index(':')
                index1 = staffs.index('|')
                directors = staffs[index0 + 1:index1].strip()
                directorlist = directors.split(', \n')
                directors = ','.join(directorlist)
                dic['directors'] = directors
                st_rest = staffs[index1 + 1:]
                index2 = st_rest.find(':')
                stars = st_rest[index2 + 1:].strip()
                starlist = stars.split(', \n')
                stars = ','.join(starlist)
                dic['stars'] = stars
                dic['url'] = card.find('span', class_='lister-item-index unbold text-primary').findNext('a')['href']
                print(dic)
                items.append(dic)

    endTime = datetime.datetime.now()
    last = endTime - startTime
    print('cost time :', last)
    dataframe = pd.DataFrame(items)
    dataframe.to_csv('Movies.csv', index=False)
    dataframe.to_excel('movies.xlsx', index=False)

def getFeatures():
    features = []
    data = pd.read_excel(moviesPath,sheet_name='Sheet1')
    for i in range(1,len(data)):
        features.append(data.iloc[i, 1])
        features.append(data.iloc[i, 5])
        features.append(data.iloc[i, 7])
        features.append(data.iloc[i, 8])

    new_features = list(set(features))
    new_features.sort(key=features.index)
    print(new_features)
    return new_features
def getClasses():
    classes = []
    data = pd.read_excel(moviesPath, sheet_name='Sheet1')
    for i in range(1,len(data)):
        classes.append(data.iloc[i, 0])
    return classes

if __name__ == '__main__':
    fetchData()
    features = getFeatures()
    classes = getClasses()








