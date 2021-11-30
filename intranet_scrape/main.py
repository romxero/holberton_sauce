#!/usr/bin/env python3
"""

Attemp to scrape intranet page to get soem data out and place it insided of a time series database 
"""

globalUserName = "randy.white@holbertonschool.com"
globalPassWord = "UrSiG2Y!cY8^"

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.expected_conditions import presence_of_element_located

import time 
import sys 




with webdriver.Firefox() as driver:
    #wait = WebDriverWait(driver, 10) #I guess wait for 10 seconds right here 
    driver.get("https://intranet.hbtn.io/users/students")
    #driver.implicitly_wait(10)
    #
    #
    #other url to check out right here: https://intranet.hbtn.io/users/students
    #userLogin = driver.find_element(By.ID, "user_login").send_keys("randy.white" + Keys.RETURN) #Needs to be modified but this is where the user login informaiton is sent to
    #userPassword = driver.find_element(By.ID, "user_password").send_keys("randy.white" + Keys.RETURN) #
    
    userLogin = driver.find_element(By.ID, "user_login")
    userPassword = driver.find_element(By.ID, "user_password")
    loginButton = driver.find_element(By.NAME, "commit")

    #send the login and password 
    userLogin.send_keys(globalUserName)
    userPassword.send_keys(globalPassWord)

    #press the button to login
    loginButton.click() #...

    #hopefully we logged in fine..
    driver.implicitly_wait(30) #sleep

    #now go to the student page
    #studentsPage =  driver.get("https://intranet.hbtn.io/users/students?page=10") #now we grab the student page right here 

    #mainStudentTableBuffer = driver.find_element(By.TAG_NAME, "table")
    #class="table table-striped" #maybe we can grab it by the class name if its not successful grabbing element type

    #print(mainStudentTableBuffer.text) #debugging the text 
    

    #Now we are going to the next page 
    ## Get all the elements available with tag name 'a'

    #below doesn't work, just think about using the url to grab any available page with users on it.
    #anchorLinks = driver.find_elements(By.TAG_NAME, 'a')


    for epochs in range(1,5):
        time.sleep(10)
        #now go to the student page
        urlString = "https://intranet.hbtn.io/users/students?page=" +  str(epochs)  #make sure to add it to the string 
        studentsPage =  driver.get(urlString) #now we grab the student page right here 
        mainStudentTableBuffer = driver.find_element(By.TAG_NAME, "table")
        #class="table table-striped" #maybe we can grab it by the class name if its not successful grabbing element type
        print(mainStudentTableBuffer.text) #debugging the text 
        
        #print(epochs)

    

    # use this to debug the thing right here 
    #implicitly wait for 30 seconds 
    
    #driver.refresh()
    #exit 

#<input type="submit" name="" value="Log in" class="btn btn-primary" data-disable-with="Log in">
#<input autocomplete="off" class="form-control" type="password" name="user[password]" id="">
 

#     for anchors in anchorLinks:
#        if anchors.text == "Next":
#            anchors.click()
#            break
#  