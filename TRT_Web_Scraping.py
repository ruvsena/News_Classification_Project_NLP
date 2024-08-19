import selenium.webdriver as webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from time import sleep
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys



user_agent ='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
driver_path = r"C:\\Users\\pc\\Desktop\\chromedriver.exe"
chrome_service = Service(driver_path)
chrome_options = Options()
chrome_options.add_argument(f'user-agent={user_agent}')
browser =webdriver.Chrome(service=chrome_service, options=chrome_options)
browser.get("https://www.trthaber.com/haber/cevre/3.sayfa.html")

wait = WebDriverWait(browser, 20)
#### buradaki son linke göre düzenlenecek aşağıdakiler bi önceki arama butonuna göre
##kelime_arat= wait.until(EC.text_to_be_present_in_element((By.XPATH, )),"Çevre")
kelime_arat=browser.find_element(By.XPATH,"/html/body/div[3]/div/div/form/div[1]/input")
kelime_arat.send_keys("Çevre",Keys.ENTER)
#ara= wait.until(EC.element_to_be_clickable((By.XPATH, "/html/body/div[3]/div/div/form/div[1]/input")))
#browser.send_keys(Keys.ENTER)



sleep(10)
habere_tikla= wait.until(EC.element_to_be_clickable(
           (By.XPATH, "/html/body/div[3]/div/div/div[1]/div[2]/div[1]/div[2]/div[2]/a")))
habere_tikla.click()
print("aa")
haber_basligi=wait.until(EC.presence_of_all_elements_located(
           (By.XPATH, "/html/body/div[3]/div/div/div[1]/div[2]/div[1]/div[2]/div[2]/a")))
print([i.text for i in haber_basligi])

with webdriver.Firefox() as driver:
    # Open URL
    driver.get("https://seleniumhq.github.io")

    # Setup wait for later
    wait = WebDriverWait(driver, 10)

    # Store the ID of the original window
    original_window = driver.current_window_handle

    # Check we don't have other windows open already
    assert len(driver.window_handles) == 1

    # Click the link which opens in a new window
    driver.find_element(By.LINK_TEXT, "new window").click()

    # Wait for the new window or tab
    wait.until(EC.number_of_windows_to_be(2))

    # Loop through until we find a new window handle
    for window_handle in driver.window_handles:
        if window_handle != original_window:
            driver.switch_to.window(window_handle)
            break

    # Wait for the new tab to finish loading content
    wait.until(EC.title_is("SeleniumHQ Browser Automation"))

print("aa")
haber_basligia=wait.until(EC.presence_of_all_elements_located(
           (By.XPATH, "/html/body/div[3]/div/div[1]/div[1]/div[4]/p[1]")))
print([i.text for i in haber_basligia])

for i in range(1,5):
    haber_metni= wait.until(EC.presence_of_all_elements_located(
               (By.XPATH, "/html/body/div[3]/div/div[1]/div[1]/div[4]/p["+str(i)+"]")))
    print([i.text for i in haber_metni])


sleep(5)

