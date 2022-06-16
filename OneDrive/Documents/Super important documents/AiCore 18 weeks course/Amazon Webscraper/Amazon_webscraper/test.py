import unittest
import pandas as pd
from PIL import Image
import re
import json
from selenium.webdriver.common.by import By
import boto3
from sqlalchemy import create_engine
from AmazonWebScraper import AmazonUKScraper



class ScraperTest(unittest.TestCase):
    """
    This test class ensures all public methods of the AmazonUKScraper Class work as expected
    """
    @classmethod
    def setUpClass(cls):
        cls.options = input("Please input your desired product category from [most wished for, best seller]: ")
        cls.scrap_1 = AmazonUKScraper(cls.options, "computer & accessories", "https://www.amazon.co.uk/")


    def test_01browser_version(self):
        """
        The first test finds the chrome driver version and compares to our expected value. This test ensures everyone using this scraper class works with the
        same version.
        """
        expected_value = '101.0.4951.67' # Driver we worked with where one can change this if needed 
         # Assert statement to check expected and actual are the same values
        actual_value = self.scrap_1.__dict__['driver'].__dict__['caps']['browserVersion']
        self.assertEqual(expected_value, actual_value)

    def test_02cookies_accept(self):
        """
        This test attempts to find the XPATH of the cookies button element after using the accept_cookies methods
        and asserts that False if cookies are not found i.e., our accept_cookies method works.
        """
        self.scrap_1.accept_cookies()
        try:
            self.scrap_1.driver.find_element(by=By.XPATH, value='//span[@class="a-button a-button-primary"]') 
            val = True
        except:
            val = False
        self.assertFalse(val)

    def test_03scraper_options(self):
        """
        The third test checks that the category of products is the same as what we passed into the options input.
        """
        self.assertEqual(self.options, self.scrap_1.__dict__['options'])
        # Our last test for this method checks whether the options category is actually "Most Wished For" or not based on our setUp method
        # We can change this any other category of our choosing depending on how we initialize our Amazon_UK_Scraper class

    def test_04region(self):
        """
        This test checks that the change_region method worked correctly and the new region is Coventry.
        """
        self.scrap_1.change_region()
        self.scrap_1.driver.find_element(by=By.XPATH, value='//span[@class="nav-line-2 nav-progressive-content"]').text[:8] == 'Coventry'

    def test_05links_number(self):
        """
        This test checks whether the number of links scraped is greater than 30 which should be the expected result.
        """
        global links
        links = self.scrap_1.get_all_links()
        self.assertGreater(len(links), 30) # More than 30 for both best seller and most wished for

    def test_06price(self):
        """
        The sixth test checks if the Price column of the product dataframe does not contain any null values as every product has a price listed.
        """
        prod_data = self.scrap_1.read_product_file()
        global prop_dict
        prop_dict = self.scrap_1.prod_dict(prod_data, links, 10)

        dataframe = pd.DataFrame(prop_dict)
        for i, j  in enumerate(dataframe['Price']):
            dataframe['Price'][i] = re.sub("[^0-9.]", "", j) # Delete all non-numeric characters apart from '.'

        self.assertGreater(dataframe['Price'].str.replace('£', '').astype(float).sum(), 30) # The sum of price column should be at least greater than £30
        self.scrap_1.update_prod_file(prop_dict)
        

    def test_07any_duplicates(self):
        """
        This test ensures no duplicate rows are present in the dataframe as we have unique Product ID and UUIDs present so we essentially test these methods.
        """
        self.assertEqual(pd.DataFrame(prop_dict).duplicated().sum(), 0)
        self.assertEqual(pd.DataFrame(prop_dict)['UUID'].nunique(), pd.DataFrame(prop_dict).shape[0]) 

    def test_08save_json(self):
        """
        This test checks whether the data.json file has been saved in the right json format.
        """
        self.scrap_1.create_raw_data_dir()
        try:
            with open('data.json') as f:
                return json.load(f) 
        except ValueError as e:
            print('invalid json: %s' % e)
            return None

    def test_09check_images(self):
        """
        The ninth test checks whether the images in the directory are in JPEG format after using the dump_json_image_upload public method.
        """
        
        _ = self.scrap_1.dump_json_image_upload(prop_dict)
        with Image.open('2.jpg') as image:
            self.assertEqual(type(image).format, 'JPEG') # Test 

        self.scrap_1.move_to_parent_dir(2)
            
    def test_10data_S3(self):
        """
        This test ensures the S3 bucket has correct contents as it checks for the data.json file and an image file in the images folder after 
        we have run our main scraper file (AmazonWebScraper.py)
        """

        s3 = boto3.client('s3')
        file = s3.head_object(Bucket='aicorebucketareeb', Key='raw_data/data.json')
        self.assertEqual(file['ResponseMetadata']['HTTPStatusCode'], 200) # Success
        
        file2 = s3.head_object(Bucket='aicorebucketareeb', Key='raw_data/images_'+self.options+'/1.jpg')
        self.assertEqual(file2['ResponseMetadata']['HTTPStatusCode'], 200) # Success

    def test_11RDS_data(self):
        """
        The last test verifies that there is our product dataset present in the AWS RDS and pgadmin after we have run our main scraper file (AmazonWebScraper.py)
        """
        
        PASSWORD = input("Please input password: ")
        ENDPOINT = input("Please enter your AWS endpoint for RDS: ")  # Your AWS endpoint

        DATABASE_TYPE = 'postgresql'
        DBAPI = 'psycopg2'
        USER = 'postgres'
        PORT = 5432
        DATABASE = 'postgres'
        engine = create_engine(f"{DATABASE_TYPE}+{DBAPI}://{USER}:{PASSWORD}@{ENDPOINT}:{PORT}/{DATABASE}")
        engine.connect()

        self.assertGreater(len(engine.execute("select * from  most_wished_for").all()), 5)

            

    @classmethod
    def tearDownClass(cls) -> (None):
        return cls.scrap_1.driver.quit()



if __name__ == '__main__':
    unittest.main()

    