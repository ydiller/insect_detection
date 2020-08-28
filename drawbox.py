import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('../bounding_boxes.csv')
#names=df['File path'].unique().tolist()
names = df['File path'].unique()
DataFrameDict = {elem: pd.DataFrame for elem in names}
#df = df[1:84]
for key in DataFrameDict.keys():
    DataFrameDict[key] = df[:][df['File path'] == key]
    path = DataFrameDict[key].iloc[0][0]
    img=cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    for index, row in DataFrameDict[key].iterrows():
        x=  row['X']
        y = row['Y']
        w = row['W']
        h = row['H']
        img_with_rects = cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 3)
    plt.imsave('../results/'+path[-15:-4]+'-high_quality.jpg',img_with_rects,cmap='gray')
    #plt.imsave('../old_results/'+path[-15:-4]+'-original.jpg', cv.imread(path))