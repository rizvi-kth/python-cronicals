import time
from selenium import webdriver

driver = webdriver.Chrome('../third/chromedriver_v73.exe')
driver.set_page_load_timeout(30)
driver.get('https://www.dustinhome.se/product/5010752115/ram')
# btn = driver.find_element_by_xpath('//*[@id="site-selector"]/div/div/div[2]/div[2]/div[3]/a')
btn = driver.find_element_by_link_text("Privatperson")
btn.click()
# <span class="price">1 395 kr</span>
lbl = driver.find_element_by_class_name('price')
print('Current price: ', lbl.text)
driver.maximize_window()
time.sleep(5)
driver.quit()

