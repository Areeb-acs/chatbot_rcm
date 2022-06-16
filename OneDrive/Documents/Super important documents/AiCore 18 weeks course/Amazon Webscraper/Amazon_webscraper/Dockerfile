FROM python:3.8

# Set up the Chrome PPA
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -
RUN echo "deb http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list \
    && apt-get -y update\
    && apt-get install -y google-chrome-stable

# RUN wget -q -0 - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -\
#     && sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'\


COPY . .

RUN pip install -r requirements.txt

CMD ["python",  "AmazonWebScraper.py"]