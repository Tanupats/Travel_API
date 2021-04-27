
from fastapi import FastAPI, File, UploadFile
app = FastAPI()

import os
import pickle
import PIL
import botnoi as bn
from botnoi import scrape as sc
from botnoi import cv
from sklearn import svm
from sklearn.svm import LinearSVC
import requests
modFile = 'mymod.mod'
mod = pickle.load(open(modFile,'rb'))


#def predicting(urlp_image_url):  
 
  #a = cv.image(urlp_image_url)
  #feat = a.getresnet50()
  #res = mod.predict([feat])
  #return res

#a=predicting('https://www.paiduaykan.com/travel/wp-content/uploads/2020/01/SON08795-800x533.jpg')
#print('class',a)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/text/{text}")
def read_word(text):
    return {"word": text}

@app.get("/Predict/{imgurl}")
def predicting(imgurl:str):  

  modFile = 'mymod.mod'
  mod = pickle.load(open(modFile,'rb'))
  a = cv.image(imgurl)
  feat = a.getresnet50()
  res = mod.predict([feat])
  return { "class" : res}



if __name__ == '__main__':
   import uvicorn
   uvicorn.run(app,host="0.0.0.0",port=5000, debug=True) 
