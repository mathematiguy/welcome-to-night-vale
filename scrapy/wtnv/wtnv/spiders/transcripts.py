# -*- coding: utf-8 -*-
import os, re
from io import open
import scrapy
from scrapy import Request

class TranscriptsSpider(scrapy.Spider):
    name = 'transcripts'
    allowed_domains = ['cecilspeaks.tumblr.com']
    start_urls = ['http://cecilspeaks.tumblr.com/']

    def parse(self, response):
    	'''Extracts the links for all transcripts from the main page
    	   and sends a calls `parse_transcript` on each'''
        # get page links for all transcripts
        links = response.xpath('//*[@id="sidebar"]/div[2]/ul/li/a/@href')\
        				.extract()
        # add base domain to link
        links = map(lambda link: response.urljoin(link), links)

        for link in links:
        	# request transcript from link
        	yield Request(link, callback=self.parse_transcript, dont_filter=True)
    
    def parse_transcript(self, response):
	    '''Logs the page visited, then saves transcript text to a text file'''
	    # log page visited
	    # print("Visited %s" % response.url)

	    # get the page id - need this to form the right xpaths
	    page_id = response.xpath("//*[re:match(@id, 'post-[0-9]+')]/@id")\
	    				  .re('\d+')[0]

    	# get episode title
	    title = response.xpath(
    				'//*[@id="post-%s"]/div[1]/div[1]/h2/text()' % page_id
    				).extract()[0]

    	# get transcript text
	    transcript = response.xpath(
	    			 	 '//*[@id="post-%s"]/div[1]/div[1]/p' % page_id
	    			 	 ).extract()

	    # do some quick cleaning: remove tags and drop new-lines within lines
	    transcript = [re.sub("<.+>", "", line) for line in transcript]
	    transcript = [re.sub("\n", " ", line) for line in transcript]
	    
	    # set path to output files
	    OUTPUT_PATH = r"C:\Users\caleb\Documents\Data Science\welcome-to-night-vale\data\transcripts"
	    
	    # set file title as file name
	    transcript_file = os.path.join(OUTPUT_PATH, title + ".txt")
	    
	    # ensure file name is not rejected by os (windows)
	    transcript_file = transcript_file.replace(r'/:*?"<>|', '')
	    
	    with open(transcript_file, 'w', encoding='utf-8') as f:
	    	# save transcript to file
	    	# print('Writing transcript to file: %s' % title)
	    	f.write("\n\n".join(transcript))
