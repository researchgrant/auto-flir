
import flirimageextractor
from matplotlib import cm
import numpy as np



import PIL
import numpy as np
import pandas as pd
import cv2
from tkinter.filedialog import askdirectory
import os
from matplotlib import pyplot as plt
import pdb
from scipy import stats
import matplotlib as mpl
# mpl.use("TkAgg")

folderName = askdirectory()
fileList = [file for file in os.listdir(folderName) if file.endswith('.jpg')]

flir = flirimageextractor.FlirImageExtractor(palettes=[cm.jet, cm.bwr, cm.gist_ncar])

#trackbar callback fucntion does nothing but required for trackbar
def thresh(x):
	pass

#create a seperate window named 'controls' for trackbar
cv2.namedWindow('controls')
#create trackbar in 'controls' window with name 'r''
cv2.createTrackbar('Threshold','controls',70,100,thresh)

def tempToGrey(x):
    return 255*(x/33)

def getTemp(file):

    image = PIL.Image.open(folderName+'/'+file)
    im_ar = np.array(image)
    img = cv2.cvtColor(im_ar, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (160,120), interpolation = cv2.INTER_AREA)
#    img = img[25:,40:]
    flir.process_image(folderName+'/'+file)
    thermal_image = flir.get_thermal_np()
    # load the image and convert it to grayscale
#    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # apply a Gaussian blur to the image then find the brightest
    # region
    #create a while loop to allow user to threshold
    temp=cv2.resize(img, (160*4, 120*4))
    cv2.putText(temp,"Select ROI", (55*4,10*4),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255))
    x,y,w,h=(np.array(list(cv2.selectROI("controls",temp)))/4).astype(int)
    if w!=0:
        img=img[y:y+h, x:x+w]
        thermal_image=thermal_image[y:y+h, x:x+w]
    gray= np.array(tempToGrey(thermal_image),  dtype = np.uint8) 
    while(1):
        temp=np.copy(img)
        #create a black image 
        #returns current position/value of trackbar 
        user_thresh= (np.log(np.linspace(1,2.71,101)))[int(cv2.getTrackbarPos('Threshold','controls'))]*255
        ret,thresh1 = cv2.threshold(gray,user_thresh,255,cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        temp[thresh1==255] = (255, 255, 255)
        
        temp=cv2.resize(temp, (160*4, 120*4))
        cv2.putText(temp,"Adjust Threshold", (50*4,10*4),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255))
        cv2.imshow('controls',temp)
    	
        # waitfor the user to press escape and break the while loop 
        k = cv2.waitKey(1)
        if k != -1:
            break
    # loop over the contours
    fig,axs=plt.subplots(1,4)
    axs[0].imshow(thresh1)
    axs[1].imshow(thermal_image)
    try:
        c = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        img=img[y:y+h, x:x+w]    
        thresh1=thresh1[y:y+h, x:x+w]
        thermal_image=thermal_image[y:y+h, x:x+w]
    except:pass
    axs[2].imshow(thresh1)

#    axs[1].imshow(thermal_image)
    mask=thresh1==255
    temp_array=[]
    #pdb.set_trace()
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if mask[x,y]:temp_array.append(thermal_image[x,y])
#    pdb.set_trace()
    
    axs[3].hist(temp_array,bins=20,range=(20,30))
    return temp_array

# gets pixel temps for each image
results = {name:getTemp(name) for name in fileList}
# extracts the groups from the file names
groups = {name:name[name.find("_")+1:name.find(".")] for name in fileList}
group_list = list(set(groups.values()))
#groups the data by treatment
all_data =pd.DataFrame.from_dict(results,orient='index')
grouped_data = {group:all_data[np.array(list(groups.values())) == group] for group in group_list}
fig2,axs=plt.subplots(1)
all_flattened={}
#plots a cumulative distribution for each group
for i,group in enumerate(grouped_data):
    all_events=grouped_data[group].to_numpy().flatten()
    all_events=all_events[~np.isnan(all_events)]
    all_flattened[group]=all_events
    n,bins,patches=axs.hist(all_events, 200, density=True, histtype='step',cumulative=True, label=group)
    patches[0].set_xy(patches[0].get_xy()[:-1])
    patches[0].set_linewidth(2)
#performs t-stats, comparing all groups
all_comparisons={}
for i,group in enumerate(all_flattened):
    others=list(all_flattened.keys())[i+1:]
    for other_group in others:
        key=str(group+" vs "+other_group)
        all_comparisons[key]=stats.ttest_ind(all_flattened[group], all_flattened[other_group])
axs.legend(loc='upper left')
fig2.tight_layout()
#use seabron to plot
table={"ID":[],"Stress":[],"Sex":[],"Chow":[],"Treatment":[],"Mean_Temp":[]}
for key,value in results.items():
    table["ID"].append(key.split("_")[0])
    table["Stress"].append(groups[key][0])
    table["Sex"].append(groups[key][1])
    table["Chow"].append(groups[key][2])
    table["Treatment"].append(groups[key][3])
    table["Mean_Temp"].append(np.mean(value))
table=pd.DataFrame(table)
import seaborn as sns
sns.catplot(data=table, kind='bar', y='Mean_Temp',x="Treatment",hue='Chow',col='Sex',row='Stress')

#exports a csv doc with mean temps
output = {item:np.average(results[item]) for item in results}
df=pd.DataFrame.from_dict(output,orient='index')
df.to_csv('temps.csv')