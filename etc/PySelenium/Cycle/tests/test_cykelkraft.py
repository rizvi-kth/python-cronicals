import time
from selenium import webdriver
from datetime import datetime
#from .saveramdata import if_notify

CSV_PATH = 'C:\\tmp\\cycle_alu.csv'

def write_new_price_to_file(new_record):
    with open(CSV_PATH, mode='a') as file:
        # print('Writing to file :{:d}'.format(new_price))
        file.write('{:%Y-%m-%d %H:%M:%S}, {:s}\n'.format(datetime.now(), new_record))
        

WEB_DRIVER_LOCATION = 'C:/Users/A547184/Git/Repos/python-cronicals/etc/PySelenium/ChromeDrivers/chromedriver_v74.exe'
# WEB_DRIVER_LOCATION = 'C:\\tools\\chromedrivers\\chromedriver_v74.exe'

START_URL = 'https://www.cykelkraft.se/cykel/mountainbike?e_bikes=5426&frame_type=1993' # Alu 

# Carbon 2019
# 'https://www.cykelkraft.se/cykel/mountainbike?dir=asc&e_bikes=5426&frame_type=1991&order=price&year=19415'


browser = webdriver.Chrome(WEB_DRIVER_LOCATION)
browser.set_page_load_timeout(30)
browser.implicitly_wait(30)
browser.get(START_URL)

# browser.find_element_by_partial_link_text('Nästa').click()

urls = []
prices = []
names = []
visa_fler_xpath = '//*[@id="amshopby-page-container"]/section/div/div[7]/ul[2]/li[5]/a'
while True:
       
    cards = browser.find_elements_by_class_name('product-grid-small')
    for card in cards:
        # print(card.find_element_by_class_name('promo').text)
        prices.append(card.find_element_by_class_name('product-grid-price').text)
        print(prices[-1])        
        names.append(card.find_element_by_class_name('product-grid-name').text)        
        print(names[-1])
        urls.append(card.find_elements_by_tag_name('a')[2].get_attribute('href'))
        print(urls[-1])

    if len(browser.find_elements_by_xpath(visa_fler_xpath)) > 0:    

        browser.find_element_by_xpath(visa_fler_xpath).click()
        time.sleep(10)
    else:
        break


print('Total MTB count: ', len(names))
weights = []
for i, url in enumerate(urls):
    browser.get(url)
    time.sleep(2)
    browser.find_element_by_xpath('//*[@id="specification-tab"]/a').click()
    time.sleep(2)
    
    lis = browser.find_elements_by_tag_name('li')
    wts = [l.text for l in lis if 'Vikt' in l.text ] or False
    
    print(prices[i])
    print(wts)
  
    
    # print(names[i])
    # print(prices[i])
    # description = browser.find_element_by_xpath('/html/body/main/div[2]/div/div[2]/div[6]/div[3]').text
    # descriptions = description.split('\n')
    # weight = [des for des in descriptions if '- Vikt:' in des and len(des) < 50] or False
    # ram = [des for des in descriptions if '- Ram:' in des and len(des) < 50] or False

    # record = '{:d}, {:s}, {:.2f}, {:s}, {:s}, {:s} '.format(i, prices[i], putils.get_first_digit_in_string(weight), str(weight), str(ram), names[i])
    record = '{:d}, {:s}, {:s}, {:s}'.format(i, prices[i], str(wts), names[i])

    print(record)

    write_new_price_to_file(record)

