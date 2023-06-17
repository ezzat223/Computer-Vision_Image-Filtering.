from flask import Flask , render_template,request, url_for,session
from flask_cors import CORS
from flask_session import Session
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
import os
import base64
import json
import Filters
import traceback
import Filters
import Frequency
from random import randint
import cv2
import histograms
from histograms import hybridFd

app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY
Session(app)

CORS(app)
def saveImg(img,path):
    with open(path, 'wb') as f:
        f.write(img)

def check_for_session_state():
    path = './static/img/output'
    old_images = os.listdir(path)
    for file in old_images:
        os.remove(path+'/'+file)     
    try:
        if session["uploaded_img"]:
            pass
    except:
        traceback.print_exc()
          
        session["uploaded_img"] = ""


def saveImg_unique(img,path, rgb=True, save_on_current = True):
    path_img = f"{path}{randint(1,100000)}.png"
    with open(path_img, 'wb') as f:
        
        f.write(img)
    if save_on_current:
        with open('./static/img/input/current.png', 'wb') as f:
            f.write(img)
        
    return path_img




@app.route("/noise_req",methods=["GET","POST"])
def noise_req():
    if request.method == "POST":
        size=request.form["size"]
        size=int(size)
        sigma=request.form["sigma"]
        sigma=float(sigma)
        D0=request.form["D0"]
        D0=int(D0)

        if request.form["noise_type"] == "uniform_noise":
            path = Filters.uniformNoise()
        elif request.form["noise_type"] == "gaussian_noise":
            path = Filters.GaussianNoise(sigma=sigma)
        elif request.form["noise_type"] == "salt_noise":
            path = Filters.salt_pepperNoise()
        elif request.form["noise_type"] == "average_filter":
            path = Filters.avr_filter(size=size)
        elif request.form["noise_type"] == "median_filter":
            path = Filters.med_filter(size=size)
        elif request.form["noise_type"] == "gaussian_filter":
            path = Filters.gass_filter(size=size,sigma=sigma)
        elif request.form["noise_type"] == "sobel":
            path = Filters.filter(kernal= "sobel")
        elif request.form["noise_type"] == "roberts":
            path = Filters.filter_robert()
        elif request.form["noise_type"] == "prewitt":
            path = Filters.filter(kernal = "prewitt")
        elif request.form["noise_type"] == "low_pass":
            path = histograms.Freq_filter(D0,'low')
        elif request.form["noise_type"] == "high_pass":
            path = histograms.Freq_filter(D0,'high')
        elif request.form["noise_type"] == "canny":
            path = Filters.canny(low_threshold = float(request.form["low_thresh"]),high_threshold = float(request.form["high_thresh"]),sigma=float(request.form["sigmaEd"]))
        print(path)
    return json.dumps({1:path})

@app.route("/freq_request",methods=["GET","POST"])
def freq_request():
    sizethr= request.form["sizethr"] 
    sizethr= int(sizethr)
    thr= request.form["thr"] 
    thr= int(thr)
    if request.method == "POST":
        if request.form["req_type"] == "histogram":
            path = Frequency.histogram()
        elif request.form["req_type"] == "equalize":
            path = Frequency.equalizeImage()
        elif request.form["req_type"] == "normalize":
            path = Frequency.normalizeImage()
        elif request.form["req_type"] == "local_thresholding":
            imgThr = cv2.imread('./static/img/input/current.png',0)
            path = Frequency.localThreshold(imgThr,size=sizethr)
        elif request.form["req_type"] == "global_thresholding":
            imgThr = cv2.imread('./static/img/input/current.png',0)
            path = Frequency.globalThreshold(imgThr,thr=thr)
        elif request.form["req_type"] == "rgb_histogram":
            path = Frequency.rgb_histogram()
        print(path)
    return json.dumps({1:path})

@app.route("/hybrid_req",methods=["GET","POST"])
def hybrid_req():
    D0Low= request.form["D0Low"] 
    D0Low= int(D0Low)
    D0High= request.form["D0High"] 
    D0High= int(D0High)
    if request.method == "POST":
        img1_path = saveImg_unique(base64.b64decode( request.form["img_upload1"].split(',')[1]), "./static/img/output/", rgb=True, save_on_current = False)
        img2_path = saveImg_unique(base64.b64decode( request.form["img_upload2"].split(',')[1]), "./static/img/output/", rgb=True, save_on_current = False)
        path1 = hybridFd(img1_path=img1_path,img2_path=img2_path,low=D0Low,high=D0High)
        path2 = hybridFd(img1_path=img1_path,img2_path=img2_path,t=2,low=D0Low,high=D0High)
    return json.dumps({1: path1,2:path2})
#start render tempates
@app.route("/edgedetection",methods=["GET","POST"])
def edgedetection():
    return render_template("edgedetection.html", img = session["uploaded_img"])

@app.route("/equalize",methods=["GET","POST"])
def equalize():
    return render_template("equalize.html", img = session["uploaded_img"])

@app.route("/hybrid",methods=["GET","POST"])
def hybrid():
    return render_template("hybrid.html", img = session["uploaded_img"])

@app.route("/",methods=["GET","POST"])
def main():
    #initializing the session states
    check_for_session_state()
    return render_template("index.html", img = session["uploaded_img"])

@app.route("/noise",methods=["GET","POST"])
def noise():
    return render_template("noise.html", img = session["uploaded_img"])





#end render tempates
@app.route("/reset_current_img",methods=["GET","POST"])
def reset_current_img():
    if request.method=="POST":
        if session["uploaded_img"]  != "":
            saveImg(base64.b64decode(session["uploaded_img"] .split(',')[1]),'./static/img/input/current.png')
    return json.dumps({1: session["uploaded_img"]})



@app.route('/saveImg',methods =['POST',"GET"])
def save_Img():
    if request.method == "POST":
        #save original images
       session["uploaded_img"] = request.form["img_upload"]
       
       saveImg(base64.b64decode( request.form["img_upload"].split(',')[1]),'./static/img/input/uploaded.png')
       path = saveImg_unique(base64.b64decode( request.form["img_upload"].split(',')[1]), "./static/img/output/", rgb=True, save_on_current = True)

    #    img= Filters.read_rgb('Images/inputs/input.png')
    #    saltnosie= Filters.salt_pepperNoise(img)
    #    print(saltnosie)
    #    saveImg(base64.b64decode(saltnosie.split(',')[1]),'./Images/output/out.png')
        
        
    return json.dumps({1: path})

if __name__ == "__main__":
    app.run(debug=True)