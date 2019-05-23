import time
from selenium import webdriver
#from .saveramdata import if_notify

WEB_DRIVER_LOCATION = 'C:/Users/A547184/Git/Repos/python-cronicals/etc/PySelenium/ChromeDrivers/chromedriver_v74.exe'
# WEB_DRIVER_LOCATION = 'C:\\tools\\chromedrivers\\chromedriver_v74.exe'
START_URL = 'https://www.xxl.se/cykel/cyklar/mountainbike-mtb/c/100300'

browser = webdriver.Chrome(WEB_DRIVER_LOCATION)
browser.set_page_load_timeout(30)
browser.implicitly_wait(30)
browser.get(START_URL)

# visa_fler = browser.find_element_by_partial_link_text("Visa")
visa_fler_xpath = '/html/body/main/div[3]/div/div/div[5]/div[2]/a'
if len(browser.find_elements_by_xpath(visa_fler_xpath)) > 0:
    browser.find_element_by_xpath(visa_fler_xpath).click()

time.sleep(5)

cards = browser.find_elements_by_class_name('gtm-p-link')
urls = []
prices = []
names = []
for card in cards:
    names.append(card.get_attribute("data-name"))
    print(card.get_attribute("data-name"))
    prices.append(card.get_attribute("data-price"))
    print(card.get_attribute("data-price"))
    urls.append(card.get_attribute("href"))
    print(card.get_attribute("href"))

print('Total MTB count: ', len(names))


browser.get(urls[4])
time.sleep(2)

# print(names[i])
# print(prices[i])
description = browser.find_element_by_xpath('/html/body/main/div[2]/div/div[2]/div[6]/div[3]').text
descriptions = description.split('\n')
weight = [des for des in descriptions if '- Vikt:' in des and len(des) < 50] or False
ram = [des for des in descriptions if '- Ram:' in des and len(des) < 50] or False

print(prices[4], weight, ram,names[4])

def try_parse(string, fail=False):
    try:
        return float(string)
    except Exception:
        return fail;

def get_weight_int(weight):
    # Process weight
    if weight:
        if isinstance(weight[0], str):
            weight[0] = weight[0].replace(',','.')
            ws = weight[0].split(' ')
            w_int = [w for w in ws if try_parse(w)] or False
            print(w_int[0])
            return float(w_int[0])
    return 0

get_weight_int(weight)