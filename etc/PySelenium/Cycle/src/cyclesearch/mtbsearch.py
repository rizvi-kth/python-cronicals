import time
import pandas as pd

from selenium import webdriver
#from .saveramdata import if_notify
from . import WEB_DRIVER_LOCATION
from . import START_URL
from ..utils import putils


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
# exit(0)

for i, url in enumerate(urls):
    browser.get(url)
    time.sleep(2)

    # print(names[i])
    # print(prices[i])
    description = browser.find_element_by_xpath('/html/body/main/div[2]/div/div[2]/div[6]/div[3]').text
    descriptions = description.split('\n')
    weight = [des for des in descriptions if '- Vikt:' in des and len(des) < 50] or False
    ram = [des for des in descriptions if '- Ram:' in des and len(des) < 50] or False

    # record = '{:d}, {:s}, {:.2f}, {:s}, {:s}, {:s} '.format(i, prices[i], putils.get_first_digit_in_string(weight), str(weight), str(ram), names[i])
    record = '{:d}, {:s}, {:.2f}, {:s}, {:s}'.format(i, prices[i], putils.get_first_digit_in_string(weight),
                                                            str(weight), str(ram))

    print(record)

    putils.write_new_price_to_file(record)

#browser.back()









# if if_notify(int(ram_price.split('kr')[0].replace(' ', ''))):

# browser.get('https://qpush.me/en/push/')
# form = browser.find_element_by_id('pushForm')
#
# name = form .find_element_by_id('in_name')
# name.send_keys('rizvi_du')
#
# code = form .find_element_by_id('in_code')
# code.send_keys('')  # 328647
#
# # //*[@id="pushForm"]/textarea
# msg = form .find_element_by_xpath('//*/textarea')
# msg.send_keys('Current price: {}'.format(ram_price))
#
# submit = form.find_element_by_id('submit')
# submit.click()

# browser.maximize_window()
time.sleep(5)
browser.quit()
