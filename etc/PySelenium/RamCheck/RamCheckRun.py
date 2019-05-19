import time
from selenium import webdriver

# browser = webdriver.Chrome('../third/chromedriver_v74.exe')
browser = webdriver.Chrome('C:\\tools\\chromedrivers\\chromedriver_v74.exe')

browser.set_page_load_timeout(30)
browser.get('https://www.dustinhome.se/product/5010752115/ram')
# btn = driver.find_element_by_xpath('//*[@id="site-selector"]/div/div/div[2]/div[2]/div[3]/a')
btn = browser.find_element_by_link_text("Privatperson")
btn.click()
# <span class="price">1 395 kr</span>
lbl = browser.find_element_by_class_name('price')
ram_price = lbl.text
print('Current price: ', ram_price)

browser.get('https://qpush.me/en/push/')
form = browser.find_element_by_id('pushForm')

name = form .find_element_by_id('in_name')
name.send_keys('rizvi_du')

code = form .find_element_by_id('in_code')
code.send_keys('328647')

# //*[@id="pushForm"]/textarea
msg = form .find_element_by_xpath('//*/textarea')
msg.send_keys('Current price: {}'.format(ram_price))

submit = form.find_element_by_id('submit')
submit.click()

# browser.maximize_window()
time.sleep(5)
browser.quit()

