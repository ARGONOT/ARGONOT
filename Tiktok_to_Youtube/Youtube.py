from selenium import webdriver 
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
import os
import glob
from info import username,password

class Bot:
    driverPath = "./chromedriver.exe"
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.browser = webdriver.Chrome(Bot.driverPath)

    def videoUpload(self):

        try:
            self.browser.get('https://accounts.google.com/v3/signin/identifier?dsh=S1862759884%3A1666626105027338&continue=https%3A%2F%2Fwww.youtube.com%2Fsignin%3Faction_handle_signin%3Dtrue%26app%3Ddesktop%26hl%3Dtr%26next%3Dhttps%253A%252F%252Fwww.youtube.com%252F%253FthemeRefresh%253D1&ec=65620&hl=tr&passive=true&service=youtube&uilel=3&flowName=GlifWebSignIn&flowEntry=ServiceLogin&ifkv=AQDHYWquoitoEWouUn07Alf5gqo9vSGfVq1uYBnYgjUpiCpNwC5VJMOTeAZuYvCr4pM1bwpB9ZfQ')
            time.sleep(10)
        except:
            self.browser.get('https://accounts.google.com/v3/signin/identifier?dsh=S1862759884%3A1666626105027338&continue=https%3A%2F%2Fwww.youtube.com%2Fsignin%3Faction_handle_signin%3Dtrue%26app%3Ddesktop%26hl%3Dtr%26next%3Dhttps%253A%252F%252Fwww.youtube.com%252F%253FthemeRefresh%253D1&ec=65620&hl=tr&passive=true&service=youtube&uilel=3&flowName=GlifWebSignIn&flowEntry=ServiceLogin&ifkv=AQDHYWquoitoEWouUn07Alf5gqo9vSGfVq1uYBnYgjUpiCpNwC5VJMOTeAZuYvCr4pM1bwpB9ZfQ')
            time.sleep(10)

        try:
            usernameInput = self.browser.find_element(By.NAME, "identifier")
            usernameInput.send_keys(username)
            usernameInput.send_keys(Keys.ENTER) 
            time.sleep(5)

        except:
            usernameInput = self.browser.find_element(By.NAME, "identifier")
            usernameInput.send_keys(username)
            usernameInput.send_keys(Keys.ENTER) 
            time.sleep(5)
        
        try:
            passwordInput = self.browser.find_element(By.NAME, "Passwd")
            passwordInput.send_keys(password)
            passwordInput.send_keys(Keys.ENTER) 
            time.sleep(5)

        except:
            passwordInput = self.browser.find_element(By.NAME, "Passwd")
            passwordInput.send_keys(password)
            passwordInput.send_keys(Keys.ENTER) 
            time.sleep(5)


        videoPath = []
        path = "./*.mp4"
        files = glob.glob(path)
        for video in files:
            videoPath.append(os.getcwd()+"\\"+"Videos\\"+video.split("\\")[1])

        for video in videoPath: 
            time.sleep(1) 
            try:
                self.browser.get("https://www.youtube.com/")
                time.sleep(6)
            except:
                self.browser.get("https://www.youtube.com/")
                time.sleep(6)
            
            try:
                self.browser.find_element(By.XPATH, "/html/body/ytd-app/div[1]/div/ytd-masthead/div[3]/div[3]/div[2]/ytd-topbar-menu-button-renderer[1]/div/a/yt-icon-button/button/yt-icon").click()
                time.sleep(2)   

            except:
                self.browser.find_element(By.XPATH, "/html/body/ytd-app/div[1]/div/ytd-masthead/div[3]/div[3]/div[2]/ytd-topbar-menu-button-renderer[1]/div/a/yt-icon-button/button/yt-icon").click()
                time.sleep(2)    

            try:
                self.browser.find_element(By.XPATH, "/html/body/ytd-app/ytd-popup-container/tp-yt-iron-dropdown/div/ytd-multi-page-menu-renderer/div[3]/div[1]/yt-multi-page-menu-section-renderer/div[2]/ytd-compact-link-renderer[1]/a/tp-yt-paper-item/div[2]/yt-formatted-string[1]").click()
                time.sleep(5)

            except:
                self.browser.find_element(By.XPATH, "/html/body/ytd-app/ytd-popup-container/tp-yt-iron-dropdown/div/ytd-multi-page-menu-renderer/div[3]/div[1]/yt-multi-page-menu-section-renderer/div[2]/ytd-compact-link-renderer[1]/a/tp-yt-paper-item/div[2]/yt-formatted-string[1]").click()
                time.sleep(5)
            
            try:
                uploadButton = self.browser.find_element(By.XPATH, "/html/body/ytcp-uploads-dialog/tp-yt-paper-dialog/div/ytcp-uploads-file-picker/div/input") ###
                uploadButton.send_keys(video)   
                time.sleep(5)   

            except:
                uploadButton = self.browser.find_element(By.XPATH, "/html/body/ytcp-uploads-dialog/tp-yt-paper-dialog/div/ytcp-uploads-file-picker/div/input") ###
                uploadButton.send_keys(video)   
                time.sleep(5)   

            try:
                text = self.browser.find_element(By.XPATH, "/html/body/ytcp-uploads-dialog/tp-yt-paper-dialog/div/ytcp-animatable[1]/ytcp-ve/ytcp-video-metadata-editor/div/ytcp-video-metadata-editor-basics/div[1]/ytcp-social-suggestions-textbox/ytcp-form-input-container/div[1]/div[2]/div/ytcp-social-suggestion-input/div")
                text.send_keys(Keys.BACK_SPACE)
                time.sleep(2)

            except:
                text = self.browser.find_element(By.XPATH, "/html/body/ytcp-uploads-dialog/tp-yt-paper-dialog/div/ytcp-animatable[1]/ytcp-ve/ytcp-video-metadata-editor/div/ytcp-video-metadata-editor-basics/div[1]/ytcp-social-suggestions-textbox/ytcp-form-input-container/div[1]/div[2]/div/ytcp-social-suggestion-input/div")
                text.send_keys(Keys.BACK_SPACE)
                time.sleep(2)

            tagname = open("hashtag.txt").read() 
            try:
                text.send_keys(tagname)
                time.sleep(3)

            except:
                text.send_keys(tagname)
                time.sleep(3)
            
            try:
                self.browser.find_element(By.XPATH, "/html/body/ytcp-uploads-dialog/tp-yt-paper-dialog/div/ytcp-animatable[1]/ytcp-ve/ytcp-video-metadata-editor/div/ytcp-video-metadata-editor-basics/div[5]/ytkc-made-for-kids-select/div[4]/tp-yt-paper-radio-group/tp-yt-paper-radio-button[2]/div[1]/div[1]").click()  
                self.browser.find_element(By.XPATH, "/html/body/ytcp-uploads-dialog/tp-yt-paper-dialog/div/ytcp-animatable[2]/div/div[2]/ytcp-button[2]/div").click()
                time.sleep(2)

            except:
                self.browser.find_element(By.XPATH, "/html/body/ytcp-uploads-dialog/tp-yt-paper-dialog/div/ytcp-animatable[1]/ytcp-ve/ytcp-video-metadata-editor/div/ytcp-video-metadata-editor-basics/div[5]/ytkc-made-for-kids-select/div[4]/tp-yt-paper-radio-group/tp-yt-paper-radio-button[2]/div[1]/div[1]").click()  
                self.browser.find_element(By.XPATH, "/html/body/ytcp-uploads-dialog/tp-yt-paper-dialog/div/ytcp-animatable[2]/div/div[2]/ytcp-button[2]/div").click()
                time.sleep(2)

            try:
                self.browser.find_element(By.XPATH, "/html/body/ytcp-uploads-dialog/tp-yt-paper-dialog/div/ytcp-animatable[2]/div/div[2]/ytcp-button[2]/div").click()
                time.sleep(2)

            except:
                self.browser.find_element(By.XPATH, "/html/body/ytcp-uploads-dialog/tp-yt-paper-dialog/div/ytcp-animatable[2]/div/div[2]/ytcp-button[2]/div").click()
                time.sleep(2)

            try:
                self.browser.find_element(By.XPATH, "/html/body/ytcp-uploads-dialog/tp-yt-paper-dialog/div/ytcp-animatable[2]/div/div[2]/ytcp-button[2]/div").click()
                time.sleep(2)
            except:
                self.browser.find_element(By.XPATH, "/html/body/ytcp-uploads-dialog/tp-yt-paper-dialog/div/ytcp-animatable[2]/div/div[2]/ytcp-button[2]/div").click()
                time.sleep(2)
            
            self.browser.find_element(By.XPATH, "/html/body/ytcp-uploads-dialog/tp-yt-paper-dialog/div/ytcp-animatable[1]/ytcp-uploads-review/div[2]/div[1]/ytcp-video-visibility-select/div[2]/tp-yt-paper-radio-group/tp-yt-paper-radio-button[3]/div[1]/div[1]").click()
            self.browser.find_element(By.XPATH, "/html/body/ytcp-uploads-dialog/tp-yt-paper-dialog/div/ytcp-animatable[2]/div/div[2]/ytcp-button[3]/div").click()
            time.sleep(60)
            


