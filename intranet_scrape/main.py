#!/usr/bin/python 
#
#
#


#
#
# Just scrap intranet
#

#import 



mainSite = "intranet.hbtn.io"


dummyData = open("/home/rcwhite/Documents/scode/python/hol_intranet_break/dummy_data/Cohort TUL-0921 (C#16) | Holberton Tulsa, OK, USA Intranet.html").read() #read file into buffer
from bs4 import BeautifulSoup

soup = BeautifulSoup(dummyData, 'html.parser')

#print (dummyData)
#print(soup.prettify())


print(soup.find(id="students"))

#id="students" The div id of where student table is located on students page


#cohorts
#reviews
#graphs
#students 

#https://intranet.hbtn.io/users/3650 #NOTE that 3650 is the user id
