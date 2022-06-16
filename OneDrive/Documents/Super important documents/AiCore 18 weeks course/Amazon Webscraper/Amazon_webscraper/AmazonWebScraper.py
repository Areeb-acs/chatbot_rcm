# Import all packages

import time
import uuid
import json
import urllib
import os
import pickle
from tqdm import tqdm



from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver import ChromeOptions

import pandas as pd
from pydantic import validate_arguments
from typing import Union

import boto3
from sqlalchemy import create_engine





class AmazonUKScraper(): 

    """

    This class is used to scrape types of product from the Amazon UK store for 
    the best seller and most wished for product categories

    Attributes:

        options (str): The product category ("Best Seller" or "Most Wished For")
        items (str): The type of products e.g., "Computer & Accessories"

    """

    # Methods --> Accept Cookies, Scrolling down, Clicking on next page until run out of pages
    # Finding information about best sellers and most wished for products in a given section in the UK Amazon ecommerce site e.g., price etc

    @validate_arguments
    def __init__(self, options: str, items, url: (str), headless=False): 
        
        """
        See help(Amazon_UK_Scraper) for details
        """

        chrome_options = ChromeOptions()
        chrome_options.add_argument('--no-sandbox') 
        chrome_options.add_argument('--disable-dev-shm-usage') 
        chrome_options.add_argument("--window-size=1920, 1080")
        chrome_options.add_argument("--remote-debugging-port=9222") 
        s = Service(ChromeDriverManager().install())

        self.options = options.lower() # To keep text consistent
        self.items = items.lower() # To keep text consistent

        if headless:
            chrome_options.add_argument('--headless')
            self.driver = webdriver.Chrome(service=s, options=options)
        else:
            self.driver = webdriver.Chrome(service=s)

        self.driver.get(url)



    def accept_cookies(self):

        """
        This method locates and accepts cookies if any.

        """

        try: # if cookies present
            self.driver.find_element(by=By.XPATH, value='//span[@class="a-button a-button-primary"]').click()
            time.sleep(1)

        except NoSuchElementException:
            pass
        
    def change_region(self):
    
        """
        This method ensures the region is set to the UK when accessing this scraper in a different country as this scraper only works within the UK region
        """

        # We need to check whether region is set to the UK as that is important 
        # This scraper only works for products delivered to regions in the UK
        # First lets check whether scraper is used in the UK and if so avoid using the steps in this method
        region = input("Are you in the UK: ")
        time.sleep(1)
        if str(region).lower() == "no":
            time.sleep(1)
            self.driver.find_element(by=By.XPATH, value='//div[@id="nav-global-location-slot"]').click() # Locate the region button and click

            time.sleep(2)
            s = self.driver.find_element(by=By.XPATH, value='//input[@class="GLUX_Full_Width a-declarative"]') # Find the input text element
            s.click()
            time.sleep(1)
            s.send_keys('CV47AL')  # send an example UK postcode
            time.sleep(1) 
            s.send_keys(Keys.ENTER) # Press Enter with the example code 
            time.sleep(1)
            self.driver.find_element(by=By.XPATH, value='//input[@type="submit"]').submit() # Submit the code 
        else:
            pass

    def scroll_bottom(self):

        """
        This function  scrolls to the bottom of the webpage; useful for loading out all elements on a given webpage.
        """

        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")


    def find_container_elements(self):

        """
        This method locates all products and saves their XPATHS into a list. This function scrolls to the bottom of every webpage until all elements are loaded 
        and then locates XPATHS for every product inside a given container.

        Returns:
            list: A list of all the XPATHS concerning product data in the given container

        """


        try:
            time.sleep(1)
            next_button = self.driver.find_element(by=By.XPATH, value='//li[@class="a-last"]')
            next_button.location_once_scrolled_into_view   # Scrolls and waits until the next button appears in view to the scraper
            time.sleep(1)
            prop_container = self.driver.find_element(by=By.XPATH, value='//div[@class="p13n-gridRow _cDEzb_grid-row_3Cywl"]')
            prop_list = prop_container.find_elements(by=By.XPATH, value='./div[@id="gridItemRoot"]')

        except NoSuchElementException:
            time.sleep(1)
            previous_button = self.driver.find_element(by=By.XPATH, value='//li[@class="a-normal"]')
            time.sleep(1)
            # As the Next button is not visible/clickable anymore due it being the last page, we search for the previous button
            # Until that button is displayed, we keep scrolling to display all products on that given page
            previous_button.location_once_scrolled_into_view # Ensure all products are displayed
            time.sleep(1)
            prop_container = self.driver.find_element(by=By.XPATH, value='//div[@class="p13n-gridRow _cDEzb_grid-row_3Cywl"]')
            prop_list = prop_container.find_elements(by=By.XPATH, value='./div[@id="gridItemRoot"]')


        return prop_list

    @validate_arguments
    def get_links_per_page(self, container_elements: list):

        """
        With the XPATHS obtained from the "find elements" method, this method acquires all the links of the products on a given webpage.

        Args:
            container_elements (list): The list returned from the find_container_elements function

        Returns:
            list: A list of all the web links of every product on a given webpage

        """

        link_list = []

        for property_prod in container_elements:
            a_tag = property_prod.find_element(by=By.TAG_NAME, value='a') # Locate the <a> tag to retrieve the href link of the product
            link = a_tag.get_attribute('href')
            link_list.append(link)

        return link_list


    def get_all_links(self):

        """

        This function of the class uses the inputs from the class constructor and the set driver method to navigate to the correct URL. 
        Afterward, it uses the find_container_elements and get_links_per_page functions on a single webpage, clicks on the next button and 
        uses the previou methods again to append all the links of products listed.

        Returns:
            list: A list of all the links of products available for the given product category

        """

        if self.options == "best seller":
            if self.items == "computer & accessories":

                self.driver.get('https://www.amazon.co.uk/Best-Sellers-Computers-Accessories/zgbs/computers/ref=zg_bs_nav_0')


        elif self.options == "most wished for":   
            if self.items == "computer & accessories":

                self.driver.get('https://www.amazon.co.uk/gp/most-wished-for/computers/ref=zg_mw_nav_0')

        big_list = []

        for _ in range(2): # 2 pages of products for every category in the best sellers or most wished section

            prop_links = self.find_container_elements()
            l = self.get_links_per_page(prop_links)
            big_list.extend(l)
            try:
                time.sleep(2)
                button = self.driver.find_element(by=By.XPATH, value='//li[@class="a-last"]')
                element = WebDriverWait(self.driver, 2).until(EC.element_to_be_clickable(button))
                element.click()
                time.sleep(1)

            except NoSuchElementException:
                break


        return big_list


    @staticmethod
    def unique_id_gen(url):

        """
        This function takes in the URL of a product webpage and returns the unique product id. For example, the format 'pd_rd_i=XXXXXXXXX&psc=N'
        is given in every URL and thus, we extract the string XXXXXXXXX.

        Args:
            url (str): The url of a given product

        Returns:
            str: The string representation of the product ID given in the url argument.
        """

        product_id = url[url.find('pd_rd_i')+8: -6]     # The .find method locates the first index of the required 
                                                        # unique ID and the actual characters are found 8 characters after

        return str(product_id)



    @staticmethod
    def v4_uuid():

        """
        This function generates a unique uuid for every link using the uuid package

        Returns:

            str: The uuid value in string format

        """


        uuid_4 = uuid.uuid4()
        return str(uuid_4)  


    def retrieve_details_from_a_page(self):

        """
        This function inspects various properties of a product e.g., title, price, brand, reviews, description, image src link etc., using multiple try-except statements

        Returns:

            tuple: The tuple contains string format of product attributes

        """

        # There are some elements such as price or voucher which sometimes differ in location depending on the 
        # product and hence, we use multiple try and except statements to locate these if they exist. 

        # Title of the product
        try:
            title = self.driver.find_element(By.XPATH, '//span[@id="productTitle"]').text
        except NoSuchElementException:
            title = 'N/A'

        # Price of the product
        try:
            price = self.driver.find_element(By.XPATH, '//span[@class="a-price a-text-price header-price a-size-base a-text-normal"]').text
        except NoSuchElementException:
            try:
                price = self.driver.find_element(By.XPATH, '//span[@class="a-size-medium a-color-price priceBlockBuyingPriceString"]').text.replace('\n', '.')
            except:
                price = 'N/A'  # Different products have prices shown on different locations (normally it could be three places, hence we use the try except statement)


        if price == 'N/A':

            try:
                price = self.driver.find_element(By.XPATH, '//span[@class="a-price aok-align-center reinventPricePriceToPayMargin priceToPay"]').text.replace('\n', '.')
            except NoSuchElementException:

                try:
                    price = self.driver.find_element(By.XPATH, '//td[@class="a-span12"]').text
                except:
                    try:
                        price = self.driver.find_element(By.XPATH, '//span[@data-maple-math="cost"]').text
                    except NoSuchElementException:
                        price = 'N/A'

        # Similar to price, we find the same problems with Brand, Voucher, Promotion and hence we perform multiple try except statements

        # Brand
        try:
            brand = self.driver.find_element(By.XPATH, '//tr[@class="a-spacing-small po-brand"]').text.split(' ')[1]
        except:
            brand = 'N/A'
        # Voucher available
        try:
            voucher = self.driver.find_element(By.XPATH, '//div[@data-csa-c-slot-id="promo-cxcw-0-0"]').text
        except NoSuchElementException:
            voucher = 'N/A'

        # Percentage reduction in price
        # These reductions are found at different places depending on the link and hence we used several try and excepts here
        try:
            price_override = self.driver.find_element(By.XPATH, '//span[@class="a-size-large a-color-price savingPriceOverride aok-align-center reinventPriceSavingsPercentageMargin savingsPercentage"]').text

        except NoSuchElementException:
            price_override = 'N/A'

        if price_override == 'N/A':
            try:
                price_override = self.driver.find_element(By.XPATH, '//td[@class="a-span12 a-color-price a-size-base\
                                                  priceBlockSavingsString"]').text
            except NoSuchElementException:
                price_override = 'N/A'


        if price_override == 'N/A':
            try:
                price_override = self.driver.find_element(By.XPATH, '//td[@class="a-span12 a-color-price a-size-base"]').text
            except NoSuchElementException:
                price_override = 'N/A'

        # Below, we know for certain that these elements exist, hence we will not use try-except here

        
        try:
            # Review ratings
            review_ratings = self.driver.find_element(By.XPATH, '//span[@class="a-size-medium a-color-base"]').text
        except:
            review_ratings = 'No rating'
        
        try:  # Number of ratings
            global_ratings = self.driver.find_element(By.XPATH, '//div[@data-hook="total-review-count"]').text
        except:
            global_ratings = 'No global rating'

        try:
            # Review topics
            topics_review = self.driver.find_element(by=By.XPATH, value='//div[@class="cr-lighthouse-terms"]').text
        except:
            topics_review = 'No review topics'

        try:
            # Most helpful review
            first_review = self.driver.find_element(by=By.XPATH, value='//div[@id="cm-cr-dp-review-list"]').find_elements(by=By.XPATH, value='./div[@data-hook="review"]')[0]
            review_helpful = first_review.find_element(by=By.XPATH, value='//span[@data-hook="review-body"]').text

        except:
            review_helpful = 'No most helpful review'


        # Main Image Link
        src = self.driver.find_element(By.XPATH, '//div[@class="imgTagWrapper"]').find_element(By.TAG_NAME, 'img').get_attribute('src')


        return title, price, brand, voucher, price_override, review_ratings, global_ratings, topics_review, review_helpful, src


    def product_data(self, prod_dictionary=None):

        """
        This method initializes and returns an empty dictionary if no dictionary is already present and input into the function. If we have already 
        scraped the data, this function  returns the scraped product dictionary and prevents us from rescraping.

        Args:
            prod_dictionary (dict): Product dictionary (either None or full of scraped products from Amazon UK)

        Returns:
            dict: Either returns an empty product dictionary or with all product information 

        """

        if prod_dictionary == None:
                    product_dict = {

                    'UUID': [],
                    'Unique Product ID': [],

                    'Title': [],
                    'Price': [],
                    'Brand': [],
                    'Savings/Promotion': [],
                    'Voucher': [],

                    'Review Ratings': [],
                    'Global Ratings': [],
                    'Topics in Reviews': [],
                    'Most Helpful Review': [],
                    'Image link': [],
                    'Page Link': []
                    }
        else:
            product_dict = prod_dictionary
        
        return product_dict

    def engine_func(self):
        """
        This class method allows us to set up connection between us and the AWS RDS database and PostgresSQL/PgAdmin
        using psycopg2
        """

        PASSWORD = input("Please input PostgresSQL password: ")
        ENDPOINT = input("Please enter your AWS endpoint for RDS: ")  # Your AWS endpoint

        DATABASE_TYPE = 'postgresql'
        DBAPI = 'psycopg2'
        USER = input("Please enter your username for database: ") #'postgres'
        PORT = 5432
        DATABASE = 'postgres'
        engine = create_engine(f"{DATABASE_TYPE}+{DBAPI}://{USER}:{PASSWORD}@{ENDPOINT}:{PORT}/{DATABASE}")
        return engine

    @validate_arguments
    def prod_dict(self, data: Union[dict, None], links, n: Union[int, str]):

        """
        This function initializes a dictionary and by using the previously defined methods, retrieves different product information from every webpage,
        and appends the information to relevant list corresponding to the appropriate dictionary key.

        Args:
            data (dict): Product dictionary (either empty or full of scraped products from Amazon UK)
            links (list): List of links relating to all products 
            n (int): How many products to scrape and gather information of

        Returns:
            dict: All product information in the form of a dictionary 

        """
        prop_dict = self.product_data(data)
        if n == 'all':
            n = len(links)
        # We use tqdm to have a progress bar to ensure the scraper is working
        try:
            engine = self.engine_func()
            global conn
            conn = engine.connect()
            prod_id_most_wished_for = pd.DataFrame(conn.execute('''SELECT "Unique Product ID" FROM most_wished_for'''))
            prod_id_best_seller = pd.DataFrame(conn.execute('''SELECT "Unique Product ID" FROM best_seller'''))
        except:
            print("Can't connect / No data present in pgadmin")
        for link in tqdm(links[0:n]):  

            # We check whether record exists in the SQL database connected with AWS RDS
            if self.unique_id_gen(link) in prop_dict['Unique Product ID']: # This prevents rescraping if the product id is already scraped and added to the dict
                if self.options == 'most wished for':
                    s = prod_id_most_wished_for['Unique Product ID'].str.contains(self.unique_id_gen(link)).sum() # There should only be one unique link
                    if s == 1:
                        print('Already scraped this product')
                        continue
                    elif s == 0:
                        print('This record does not exist in the SQL data in AWS RDS & PgAdmin')
                        pass
                else:
                    s = prod_id_best_seller['Unique Product ID'].str.contains(self.unique_id_gen(link)).sum()
                    if s == 1: # There should only be one unique link
                        print('Already scraped this product')
                        continue

                    elif s == 0:
                        print('This record does not exist in the SQL data in AWS RDS & PgAdmin')
                        pass

            

            self.driver.get(link)
            time.sleep(1)
            self.scroll_bottom()
            time.sleep(2)
            

            prop_dict['Unique Product ID'].append(self.unique_id_gen(link))
            prop_dict['Page Link'].append(link)
            prop_dict['UUID'].append(self.v4_uuid())


            title, price, brand, voucher, price_override, review_ratings, global_ratings, \
            topics_review, review_helpful, src = self.retrieve_details_from_a_page()

            prop_dict['Title'].append(title)
            prop_dict['Price'].append(price)
            prop_dict['Brand'].append(brand)
            prop_dict['Voucher'].append(voucher)
            prop_dict['Savings/Promotion'].append(price_override)
            prop_dict['Review Ratings'].append(review_ratings)
            prop_dict['Global Ratings'].append(global_ratings)
            prop_dict['Topics in Reviews'].append(topics_review)
            prop_dict['Most Helpful Review'].append(review_helpful)
            prop_dict['Image link'].append(src)

        return prop_dict

    @validate_arguments 
    def dump_json_image_upload(self, prod_diction: dict):

        """
        This function creates a new json file, stores the product dictionary obtained from the prod_dict function in json format. 
        Furthermore, it creates another directory  within the raw_data directory or folder to store all the image data. 
        Lastly, using the urlib package, it retrieves the product image link from the product dictionary and stores each image as jpg. 

        Args:
            prod_diction (dict): Information about every product

        Returns:
            df_prod: All product information in a pandas dataframe to upload on the AWS RDS
        """
        # dump the generated dictionary into a json file

        with open('data.json', 'w') as f:
            df_prod = pd.DataFrame(prod_diction)  # First convert our dictionary of product information to a dataframe 
                                                  # and then to json
            json.dump(df_prod.to_json(), f)


        try:
            os.mkdir("images_"+self.options)
        except:
            print("Directory already exists")

        os.chdir('images_'+self.options)
        for i, img_link in enumerate(prod_diction['Image link']):
            # download the image
            if os.path.exists(f"{i}.jpg") == True:
                pass
            else:
                urllib.request.urlretrieve(img_link, f"{i}.jpg")

        return df_prod

    @staticmethod
    def create_raw_data_dir():

        """
        This method creates a directory called "raw_data" and changes the current directory

        """
        # Try except statement as directory will be created once the code is run and cannot be created twice
        try:
            os.mkdir("raw_data")
        except:
            print("Directory already exists")

        os.chdir('raw_data')

    def upload_to_cloud(self):

        """
        This class method uses boto3 to create a S3 bucket on AWS and upload the raw_data folder which includes all the image files 
        and the product information json file. 

        """
        empty = input("Do you want to overwrite the raw_data folder in the S3 bucket: ")
        if empty.lower() == 'yes':

            key_id = input('Enter your AWS key id: ')
            secret_key = input('Enter your AWS secret key: ')
            bucket_name = input('Enter your bucket name: ')
            region = input('Enter your regions: ')

            s3 = boto3.client("s3", 
                            region_name=region, 
                            aws_access_key_id=key_id, 
                            aws_secret_access_key=secret_key)

            s3.upload_file('raw_data/data.json', bucket_name, 'raw_data/data.json')


            for i in os.listdir('raw_data/images_'+self.options): # We list out all the image files and loop to upload the files to S3 one by one
                s3.upload_file('raw_data/images_'+self.options+'/'+i, bucket_name, 'raw_data/images_'+self.options+'/'+i)
        else:
            pass


    def upload_dataframe_rds(self, df):

        """
        This function takes the dataframe which was entered as an argument, converts it to SQL format and then saves it in the RDS.

        Args:
            df (DataFrame): Pandas dataframe containing all product information which was scraped

        """
        empty = input("Do you want to overwrite the previous SQL data in RDS: ")
        if empty.lower() == 'yes':
            if self.options == 'most wished for':
                df.to_sql("most_wished_for", conn, if_exists='replace', chunksize=60) # We have defined engine globally previously
            else:
                df.to_sql("best_seller", conn, if_exists='replace', chunksize=60)
        else:
            pass


    def move_to_parent_dir(self, n):

        """
        This class method allows us to move above to the parent directory a specified number of times using the os library

        Args:
            n (int): The number of times we want to move above to the parent directory

        """
        for _ in range(n):
            parent_directory = os.path.dirname(os.getcwd())
            os.chdir(parent_directory)

    def update_prod_file(self, product_dictionary):
    
        """
        This class method creates a pickle file named after the options input of the class and dumps the product dictionary there using pickle

        Args:
            product_dictionary (dict): The dictionary obtained after scraping all products (either best seller or most wished for)

        """
        empty = input("Do you want to overwrite the previous product data pickle file: ")
        if empty.lower() == 'yes':
            with open('product_data_'+self.options+'.pkl', 'wb') as dict_data:
                pickle.dump(product_dictionary, dict_data)
        else:
            pass

    def read_product_file(self):

        """
        This function asks the user if a product file already exists and reads that using the pandas read_pickle method

        Returns:
            data (dict or None): All product information in the form of a dictionary or returns None
        """

        file_exist = input("Does product file exist: ")
        if file_exist.lower() == 'no':
            return None
        else:
            data = pd.read_pickle('product_data_'+self.options+'.pkl')
            return data






if __name__ == '__main__':

    options = input("Please input your desired product category from (most wished for, best seller): ")
    scraper = AmazonUKScraper(options, "computer & accessories", "https://www.amazon.co.uk/", headless=True)
    scraper.accept_cookies()
    scraper.change_region()  # Use this if you are not in the UK as the scraper only works delivery regions in the UK

    prod_links = scraper.get_all_links()  # The get all links method get_all links function mainly justs appends the links to a 
    # main list and clicks on the next button to obtain the same data on the next page.
    # Either there is no file or there is a product data file with a dictionary containing product information scraped
    prod_data = scraper.read_product_file()
    # We can set the prod_data to None if we want to just scrape new products
    product_dictionary = scraper.prod_dict(prod_data, prod_links, 'all') # Get information about all products (We can specify numbers like 2, 3, 10 etc)
    scraper.update_prod_file(product_dictionary)
    scraper.create_raw_data_dir()
    dataframe = scraper.dump_json_image_upload(product_dictionary)

    # Go back two directories prior to be able to use other methods in the future

    scraper.move_to_parent_dir(2)
    scraper.upload_to_cloud()
    scraper.upload_dataframe_rds(dataframe)
    scraper.driver.quit()











