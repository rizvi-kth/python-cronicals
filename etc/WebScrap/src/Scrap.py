import requests
from bs4 import BeautifulSoup
import pandas as pd

soup = BeautifulSoup(requests.get('https://www.xxl.se/cykel/cyklar/mountainbike-mtb/c/100300').text)
cards = soup.find_all('div', attrs={'class': 'cards'})

for card in cards:
    ass = card.find_all('a', attrs={'class': 'card card-prod gtm-p-link'})
    for a in ass:
        # print(a.prettify())
        # print(a.find_all('div', attrs={'class': 'description'})[0].text)
        print(a.find_all('div', attrs={'class': 'description'})[0].text)

# df = pd.read_html(str(cards))
