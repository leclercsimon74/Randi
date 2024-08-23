# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:13:17 2024

@author: sleclerc

Fusion of the classifier randomizer and of database maker, with option!


TODO:
    - open recent file log              DONE
        - reorder them as well          DONE
    - about window
    - more help part                    partially DONE
    - Different images shape scenario   DONE
    - Different input image format
    - Different output format
    
    - Add a training and prediction capacity - CPU
        - Classical network
        - prebuilt network
        
    - Make a database from a single folder (containing at least one image)

BUG:
    - If no database created/open if selected classifier->new
        self.df = self.df.sample(frac=1).reset_index(drop=True) #shuffle
        AttributeError: 'NoneType' object has no attribute 'sample')
    
    - if closing the database creation window
        self.database = pd.read_csv(self.database_log['Name']+os.sep+'database.csv')
        TypeError: 'NoneType' object is not subscriptable
    

"""

#classifier import
import pandas as pd #database generation
import os #file and folder managment
import sys #window creation
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QMainWindow, QLabel, QDialog, QLineEdit, QPushButton,
                             QMessageBox, QComboBox, QCheckBox, QProgressBar,
                             QDialogButtonBox, QGroupBox, QVBoxLayout, QHBoxLayout,
                             QFormLayout, QSpinBox, QWidget, QAction, QStatusBar,
                             QApplication, QTableWidget, QTableWidgetItem)
from PyQt5.QtGui import QPixmap, QFont, QIntValidator, QImage
import numpy as np #image manipulation
import random #randomize image orientation and display
from functools import partial
from shutil import rmtree

#database maker import
import tifffile #image opening and saving
from skimage import filters, measure, transform #image quick manipulation such as segmentation and rotation
from scipy import ndimage as nd #image binary manipulation
import json

#optionnal
import matplotlib.pyplot as plt #testing purpose, may remove

DEBUG = True


randi_json = {}
randi_json_path = os.getcwd()+os.sep+"Randi"+os.sep+'randi.json'

def read_jsn():
    global randi_json, randi_json_path
    if os.path.isfile(randi_json_path): #check that we have the file
        with open(randi_json_path, 'r', encoding='utf-8') as f:
            randi_json = json.load(f)
    else: #need to put the minimum amount of information in the dictionnary, even if blank
        randi_json['open recent'] = []
    return randi_json


def save_jsn(randi_json):
    global randi_json_path
    with open(randi_json_path, 'w', encoding='utf-8') as f:
        json.dump(randi_json, f, ensure_ascii=False, indent=4)

def add_recent_path_to_json(new_path):
    randi_json = read_jsn()
    open_recent_list = randi_json['open recent']
    if new_path in open_recent_list: #remove the old reference, so the new one can be added to the top!
        del open_recent_list[open_recent_list.index(new_path)]
    open_recent_list.insert(0, new_path) #add on top of the list
    #remove last item if too long
    if len(open_recent_list) >= 5:
        open_recent_list = open_recent_list[:5]
    randi_json['open recent'] = open_recent_list
    save_jsn(randi_json)


class MainWindow(QMainWindow):
    def __init__(self, parent = None):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Randi")
        self.resize(600, 400)
        self.app_path = os.getcwd()
        self.randi_json = read_jsn()
        self.database = None
        self.isdatabase = False
        self.database_log = None
        
        self._createUI()
        self._createActions()
        self._createMenuBar()
        self._connectActions()
        self._createStatusBar()
        

    def _createMenuBar(self):
        menuBar = self.menuBar()
        # Database menu
        fileMenu = menuBar.addMenu("&Database")
        fileMenu.addAction(self.new_databaseAction)
        # self.openRecentMenu = fileMenu.addAction(self.open_r_databaseAction)
        self.openRecentMenu = fileMenu.addMenu("Open Recent")
        fileMenu.addSeparator()
        fileMenu.addAction(self.exitAction)
        # Classifier menu
        editMenu = menuBar.addMenu("&Classifier")
        editMenu.addAction(self.new_classiAction)
        #help menu
        helpMenu = menuBar.addMenu("&Help")
        helpMenu.addAction(self.helpContentAction)
        helpMenu.addAction(self.aboutAction)
        
    
    def _createUI(self):
        central = QWidget()
        self.databaselogWidget = QLabel(self)
        self.databaselogWidget.setAlignment(Qt.AlignTop)
        self.databaselogWidget.setWordWrap(True)
        self.databaselogWidget.setText('No database loaded')

        self.vbox = QVBoxLayout(central)
        self.vbox.addWidget(self.databaselogWidget)
        self.configureTable()
        self.vbox.addWidget(self.table)
        

        
        self.setCentralWidget(central)
        
    
    def configureTable(self):
        if isinstance(self.database, pd.DataFrame):
            #remove the previous widget
            self.vbox.removeWidget(self.table)
            self.table.deleteLater()
            #add the Table
            self.table = QTableWidget()
            self.table.setRowCount(len(self.database))
            self.table.setColumnCount(len(list(self.database)))
            self.table.setHorizontalHeaderLabels(list(self.database))
            
            for idx, row in self.database.iterrows(): #by row
                for ydx, col in enumerate(list(self.database)): #and col
                    item = QTableWidgetItem(str(row[col]))
                    item.setFlags(Qt.ItemIsEnabled)
                    self.table.setItem(idx, ydx, item)
            
            self.vbox.addWidget(self.table)

        else:
            self.table = QLabel(self)
            self.table.setText('No database loaded')
            self.table.setAlignment(Qt.AlignTop)
        
        
    def _createActions(self):
        self.new_databaseAction = QAction("&New/Open", self)
        self.new_databaseAction.setStatusTip("Create a new database or load an existing database")

        self.open_r_databaseAction = QAction("&Open Recent", self)
        self.open_r_databaseAction.setStatusTip("Open a list of the 5 most recent databases")

        self.new_classiAction = QAction("&New", self)
        self.new_classiAction.setStatusTip("Start a new human driven classification")
        
        self.exitAction = QAction("&Exit", self)
        self.exitAction.setStatusTip("Exit the program. With confirmation")
        self.helpContentAction = QAction("&Help Content", self)
        self.helpContentAction.setStatusTip("Need to be build")
        self.aboutAction = QAction("&About", self)
        self.aboutAction.setStatusTip("Need to be build")
        
    def _createStatusBar(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)


    def _connectActions(self):
        # Connect Database actions
        self.new_databaseAction.triggered.connect(self.Database)
        self.open_r_databaseAction.triggered.connect(self.openRecentFile)
        self.exitAction.triggered.connect(self.close)
        # Connect Classifier actions
        self.new_classiAction.triggered.connect(self.Classifier)
        # Connect Help actions
        self.helpContentAction.triggered.connect(self.helpContent)
        self.aboutAction.triggered.connect(self.about)
        self.openRecentMenu.aboutToShow.connect(self.populateOpenRecent)


    def Database(self):
        #create a classical 'Select Folder' window
        folderpath = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder')
        if folderpath != '': #detect if cancel
            self._openDatabase(folderpath)
    
    def _openDatabase(self, folderpath):
        os.chdir(folderpath) #change the working directory
            
        #analyse the folderpath to determine classification and/or randomization
        dirs = os.path.basename(os.getcwd())+' - database'
        if os.path.exists(dirs) and os.path.isfile(dirs+os.sep+'database.csv'): #database already exist, assuming classifier only
            self.display_database_log()
            self.database = pd.read_csv(self.database_log['Name']+os.sep+'database.csv')
        
        elif os.path.isfile('database.csv'):#it is the database itself!!
            self.isdatabase = True
            self.databaselogWidget.setText("Database loaded without original images")
            self.database = pd.read_csv('database.csv')
        
        else: #does NOT exist, need to create the database first!
            self.data_window = DatabaseWindow(dirs)
            self.data_window.exec_()
            self.display_database_log()
            self.database = pd.read_csv(self.database_log['Name']+os.sep+'database.csv')

        
        self.configureTable()  
    
    def display_database_log(self):
        if os.path.isfile('log.json'): #check that we have the log
            with open('log.json') as f: #open it!
                self.database_log = json.load(f)
            
            self.databaselogWidget.setText("\n".join(["Current database"]+str(self.database_log).replace('\'','').replace('{','').replace('}','').split(',')))

    def populateOpenRecent(self):
        # Remove the old options from the menu
        self.openRecentMenu.clear()
        # Dynamically create the actions
        self.randi_json = read_jsn()
        actions = []
        filenames = self.randi_json['open recent']
        for filename in filenames:
            action = QAction(filename, self)
            action.triggered.connect(partial(self.openRecentFile, filename))
            actions.append(action)
        # Add the actions to the menu
        self.openRecentMenu.addActions(actions)

    def openRecentFile(self, folderpath, arg2):
        #check if the dir exist
        if os.path.isdir(folderpath):
            if DEBUG: print('Quick Load path ', folderpath)
            if os.path.isdir(folderpath): #open if it exists!
                add_recent_path_to_json(folderpath)
                self._openDatabase(folderpath)
            else:
                print('WARNING, path does not exist', folderpath)
                del randi_json['open recent'][randi_json['open recent'].index(folderpath)]
            
    
    def close(self):
        reply = QMessageBox.question(self, 'Closing', 'Are you sure you want to quit?', QMessageBox.Ok | QMessageBox.Cancel)
        if reply == QMessageBox.Ok:
            QApplication.quit()
            
            
    def Classifier(self):
        #classification itself
        ClassifierWindow(self.database, self.database_log, self.isdatabase)
        if self.isdatabase:
            self.database = pd.read_csv('database.csv')
        else:
            self.database = pd.read_csv(self.database_log['Name']+os.sep+'database.csv')
        self.configureTable()
    
       
    
    def helpContent(self): #TODO
        pass
    
    
    def about(self): #TODO
        pass
        
        

class DatabaseWindow(QDialog):
    def uniqueid(self):
        seed = random.getrandbits(32)
        while True:
           yield seed
           seed += 1
           
    def __init__(self, dirs):
        super(DatabaseWindow, self).__init__()
        self.dirs = dirs
        #image format
        self.zcxy_a = False
        self.czxy = False
        self.zxy = False
        self.cxy = False
        self.xy = False
        self.c = False
        self.z = False
        self.all_c = False #if the channel of the image is in the same image
        self.C_n = None #number of channel
        
        #grab all tif image and path - recursive, but assuming 1 depth
        result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(os.getcwd()) for f in filenames if f.endswith('.tif')]
        result = [f.replace(os.getcwd()+os.sep, '') for f in result] #simplify the path
        
        #quicker than opening the data!
        with tifffile.TiffFile(result[0]) as tif:
            self.axes = tif.series[0].axes #grab the axes! YX or ZYX
            self.shape = tif.series[0].shape #grab the axes! YX or ZYX
        
        if DEBUG: print("Image axis and shape", self.axes, self. shape)
        
        #determine the image format
        if os.path.basename(result[0]).startswith("C1-"): #start with C, image that got splitted!
            self.c = True
            img_name = [os.path.basename(f).replace("C1-", '') for f in result if os.path.basename(f).startswith("C1-")] #generic image name!
            img_folder = [f.split(os.sep)[0] for f in result if os.path.basename(f).startswith("C1-")] #grab the folder
            if self.axes == 'ZYX' or self.axes == 'ZXY': #stack
                self.czxy = True
                self.z = True
        
            elif self.axes == 'YX' or self.axes == 'XY': #single plane image
                self.cxy = True
        
        elif "C" in self.axes: #image that are not splitted
            self.c = True
            if "Z" in self.axes:
                self.z = True
                self.zcxy_a = True
            self.all_c = True
            self.C_n = self.shape[self.axes.find('C')]
            img_name = [os.path.basename(f) for f in result] #generic image name!
            img_folder = [f.split(os.sep)[0] for f in result] #grab the folder
        
        else: #no color, grayscale image
            img_name = [os.path.basename(f) for f in result] #generic image name!
            img_folder = [f.split(os.sep)[0] for f in result] #grab the folder
            if self.axes == 'ZYX' or self.axes == 'ZXY': #stack
                self.zxy = True
                self.z = True
        
            elif self.axes == 'YX' or self.axes == 'XY': #single plane image
                self.xy = True
        
        #initialize the database
        self.df = pd.DataFrame()
        self.df.loc[:,"Folder"] = img_folder
        self.df.loc[:,"Name"] = img_name
        
        if DEBUG:
            print('self.zcxy_a', self.zcxy_a)
            print('self.czxy', self.czxy)
            print('self.zxy', self.zxy)
            print('self.cxy', self.cxy)
            print('self.xy', self.xy)
            print('self.c', self.c)
            print('self.z', self.z)
            print('self.all_c', self.all_c)
            print('self.C_n', self.C_n)           
        
        #create the UI
        self.initUI()
    
    def initUI(self):
        # setting window title
        self.setWindowTitle("Database Maker")
        # self.setGeometry(100, 100, 200, 200)
        self.resize(200, 200)
        self.formGroupBox = QGroupBox("Settings")
        # Best plane selector
        if self.z:
            self.plane_CB = QCheckBox("Best plane selector", self)
            self.plane_CB.setChecked(True) #True by default
        # crop to selection
        self.crop_CB = QCheckBox("Crop to selection", self)
        self.crop_CB.setChecked(True) #True by default
        if self.c: #if there are nore than 1 channel
            self.channel_plane_label = QLabel("Channel for the selection")
            self.planeC_selector = QSpinBox()
            self.planeC_selector.setValue(1)
        #Z projection selector
        if self.z:
            self.zproj_CB = QCheckBox("Z projection", self)
            self.zproj_CB.setChecked(False) #false by default
            self.projtype_label = QLabel("Projection type")
            self.zproj = QComboBox(self)
            self.zproj.addItems(["Max", "Mean", "Sum"])
            self.zproj.currentTextChanged.connect(self.on_combobox_func)
        #indicate the channel of interest
        if self.c: #if there are nore than 1 channel
            self.channel_label = QLabel("Channel for the database")
            self.C_selector = QSpinBox()
            self.C_selector.setValue(1)
        #image size 
        self.imgsize_label = QLabel("Image size (pxl)")
        self.imgsize_selector = QSpinBox()
        self.imgsize_selector.setMaximum(1000)
        self.imgsize_selector.setValue(100)
        #progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        
        #group the best plane and z proj button to make them exlusive
        if self.z:
            button_group = QtWidgets.QButtonGroup(self)
            button_group.addButton(self.plane_CB)
            button_group.addButton(self.zproj_CB)
        
        #create the form
        self.createForm()
 
        # creating a dialog button for ok
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok)
        # adding action when form is accepted
        self.buttonBox.accepted.connect(self.process_database)

        # creating a vertical layout
        mainLayout = QVBoxLayout()
        # adding form group box to the layout
        mainLayout.addWidget(self.formGroupBox)
        # adding button box to the layout
        mainLayout.addWidget(self.buttonBox)
 
        # setting lay out
        self.setLayout(mainLayout)
        self.show()

    def on_combobox_func(self, text): #update the text in the combobox
        self.current_text = text
    
    def get_state(self):
        if self.z:
            self.zproj_text = self.zproj.currentText() #type of projection to do
            
            if self.zproj_CB.checkState() == 2:
                self.zproj_state = True
            else:
                self.zproj_state = False
            
            if self.plane_CB.checkState() == 2:
                self.plane_state = True
            else:
                self.plane_state = False
        else:
            self.zproj_state = None
            self.plane_state = None
            self.zproj_text = None
        
        if self.c:
            self.plane_C = int(self.planeC_selector.text())
            self.selec_C = int(self.C_selector.text())

        else:
            self.plane_C = None
            self.selec_C = None        

        if self.crop_CB.checkState() == 2:
            self.crop_state = True
        else:
            self.crop_state = False

        self.size = int(self.imgsize_selector.text())
        
        #write the data to a file
        self.log = {'Path':os.getcwd(), #?? FOLDER?
                    'Name':self.dirs,
                    'Best plane':self.plane_state,
                    'Crop state':self.crop_state,
                    'Z projection':self.zproj_state,
                    'Z projection type':self.zproj_text,
                    'Channel for selection':self.plane_C,
                    'Channel for database':self.selec_C,
                    'Image output size':self.size}
        
        if DEBUG: print(self.log)
        
        with open('log.json', 'w', encoding='utf-8') as f:
            json.dump(self.log, f, ensure_ascii=False, indent=4)
        
        add_recent_path_to_json(self.log['Path'])

        
    # process the data according to the setup when form is accepted
    def process_database(self):
        self.buttonBox.hide()
        self.get_state() #this is to save and fix these value!
        valid = False
        
        #Condition checking
        row = self.df.iloc[0]
        if self.czxy or self.cxy: #split channel
            if os.path.isfile(os.path.join(row.Folder, "C"+str(self.plane_C)+"-"+row['Name'])):
                if os.path.isfile(os.path.join(row.Folder, "C"+str(self.selec_C)+"-"+row['Name'])):
                    valid = True
                    
        if self.all_c and self.C_n != None: #fused channel
            if self.plane_C <= self.C_n and self.selec_C <= self.C_n:
                valid = True
                
        if self.xy or self.zxy: #mono color
            valid = True
        
        if valid:
            #create database folder
            dirs = os.path.basename(os.getcwd())+' - database'
            if os.path.isdir(dirs): #delete if already present! (error during previous creation)
                rmtree(dirs)
            os.makedirs(dirs) #make the directory
            self.progress_bar.show()
            
            
            uids = []
            for idx, row in self.df.iterrows():
                self.progress_bar.setValue(int((idx+1)/len(self.df)*100))
                if self.progress_bar.value() == 100:
                    self.progress_bar.hide()
                
                #open the image
                if self.all_c: #fused img, open the full image
                    ori_img = tifffile.imread(os.path.join(row.Folder, row['Name']))
                    #select the correct channel
                    img = ori_img.take(indices=self.selec_C-1, axis=self.axes.find('C'))
                elif self.c: #split channel based
                    img = tifffile.imread(os.path.join(row.Folder, "C"+str(self.selec_C)+"-"+row['Name'])) #channel of interest
                elif self.xy:
                    img = tifffile.imread(os.path.join(row.Folder, row['Name']))
                
                #3D gestion
                if self.z:
                    if self.zproj_state: #Z proj ONLY
                        img = self.ZProjection(img)
                        
                    if self.plane_state or self.crop_state: #best plane selector ON! Default
                        if self.all_c: #select the correct channel
                            bestplane, bbox = self.bestPlaneSelector(ori_img.take(indices=self.selec_C-1, axis=self.axes.find('C')))
                        elif self.c: #split channel based
                            bestplane, bbox = self.bestPlaneSelector(tifffile.imread(os.path.join(row.Folder, "C"+str(self.plane_C)+"-"+row['Name'])))
                        elif self.zxy:
                            img = tifffile.imread(os.path.join(row.Folder, row['Name']))
                            bestplane, bbox = self.bestPlaneSelector(img)
                            
                            
                        if self.zproj_state: #Zproj AND crop
                            img = img[bbox[0]:bbox[1], bbox[2]:bbox[3]]
                        if self.plane_state and self.crop_state: #bestplane AND crop
                            img = img[bestplane, bbox[0]:bbox[1], bbox[2]:bbox[3]]
                        elif self.plane_state: #bestplane ONLY
                            img = img[bestplane]
                
                elif self.crop_CB.checkState(): 
                    if self.crop_state: #2D, crop the image to the bounding box
                        img = self.bbox_finder_2D(img)
                
                #resize the image
                img_small = transform.resize(img, (self.size, self.size), preserve_range=True) #resize
                img_small = img_small/np.max(img_small) * 255 #normalize
                img_small = img_small.astype('uint8')
                
                #assign an unique id to each image. Be sure to NOT deupli
                unique_sequence = self.uniqueid()
                uid = next(unique_sequence)
                while True:
                    if uid not in uids:
                        uids.append(uid)
                        break
                    uid = next(unique_sequence)
                    
                #save the image in the database
                tifffile.imwrite(dirs+os.sep+str(uid)+'.tif', img_small, compression='zlib', metadata={'axes': 'YX'})
            
            self.df.loc[:,'uid'] = uids
            self.df = self.df.loc[:, ~self.df.columns.str.contains('^Unnamed')]
            self.df.to_csv(dirs+os.sep+'database.csv')
            # Finish the database creation process
            reply = QMessageBox.question(self, 'Database!', 'Database has been created', QMessageBox.Ok, QMessageBox.Ok)
            if reply == QMessageBox.Ok:
                self.close()
                
        else:
            QMessageBox.critical(None, "ERROR", "At least one field is invalid!")

    # create form method
    def createForm(self):
        # creating a form layout
        layout = QFormLayout()
        # for best plane selector
        if self.z: layout.addRow(self.plane_CB)
        #crop to selection
        layout.addRow(self.crop_CB)
        if self.c and self.z: layout.addRow(self.channel_plane_label, self.planeC_selector)
        # for projection type
        if self.z:
            layout.addRow(self.zproj_CB)
            layout.addRow(self.projtype_label, self.zproj)
        #channel of interest
        if self.c: layout.addRow(self.channel_label, self.C_selector)
        #for image size
        layout.addRow(self.imgsize_label, self.imgsize_selector)
        #progressbar
        layout.addRow(self.progress_bar)
        # setting layout
        self.formGroupBox.setLayout(layout)
    
    def ZProjection(self, img):
        if self.zproj_text == "Max": img = np.max(img, axis=0)
        elif self.zproj_text == "Mean": img = np.mean(img, axis=0)
        elif self.zproj_text == "Sum": img = np.sum(img, axis=0)
        return img

    def bbox_finder_2D(self, img): #YX
        # detect the largest object and remove everything else!
        nucleus_b = img > filters.threshold_otsu(img)
        nucleus_b = nd.binary_erosion(nucleus_b)
        nucleus_b = nd.binary_dilation(nucleus_b)
        nucleus_b = nd.binary_fill_holes(nucleus_b)
        nucleus_l = measure.label(nucleus_b)
        nucleus_df = measure.regionprops_table(nucleus_l, properties=('label', 'area', 'bbox'))
        nucleus_df = pd.DataFrame(nucleus_df)
        nucleus_df = nucleus_df.sort_values('area', ascending=False)
        nucleus_df.reset_index(inplace=True)
        nucleus = nucleus_df.iloc[nucleus_df['area'].idxmax()]
        nucleus_b = np.where(nucleus_l == nucleus.label, True, False)
        bbox = (int(nucleus['bbox-0']), int(nucleus['bbox-2']), int(nucleus['bbox-1']), int(nucleus['bbox-3']))
        return img[bbox[0]:bbox[1], bbox[2]:bbox[3]]
        
    def bestPlaneSelector(self, img): #ZYX
        #I estimate that most of the time, the best plane is on the largest plane of the nucleus! Time consuming though!
        #segment the nucleus
        nucleus_b = img[:,:,:] > filters.threshold_otsu(img[:,:,:])
        #then remove noise
        nucleus_b = nd.binary_erosion(nucleus_b)
        nucleus_b = nd.binary_closing(nucleus_b, iterations=2)
        nucleus_b = nd.binary_opening(nucleus_b, iterations=2)
        nucleus_b = nd.binary_dilation(nucleus_b)
        nucleus_b = nd.binary_fill_holes(nucleus_b)
        nucleus_l = measure.label(nucleus_b)
        #measure it
        nucleus_df = measure.regionprops_table(nucleus_l, properties=('label', 'area', 'bbox'))
        nucleus_df = pd.DataFrame(nucleus_df)
        nucleus_df = nucleus_df.sort_values('area', ascending=False)
        nucleus_df.reset_index(inplace=True)
        nucleus = nucleus_df.iloc[nucleus_df['area'].idxmax()]
        nucleus_b = np.where(nucleus_l == nucleus.label, True, False)
        bbox = (int(nucleus['bbox-1']), int(nucleus['bbox-4']), int(nucleus['bbox-2']), int(nucleus['bbox-5']))
        #found the best plane, aka the largest zplane of the nucleus
        return (np.argmax(np.sum(nucleus_b, axis=(1,2))), bbox)

class ClassifierWindow(QDialog):
# class ClassifierWindow(QWidget):
    def __init__(self, df, log, isdatabase):
        # super(ClassifierWindow, self).__init__()
        self.img_size = 550
        
        self.isdatabase = isdatabase
        self.df = df
        self.df = self.df.sample(frac=1).reset_index(drop=True) #shuffle
        self.log = log
        
        self.idx_df = 0
        self.result = []
        self.user_name = ''
        self.cat_n = 0
        
        self.ini_window = Window()
        self.ini_window.exec_()
        self.user_name = self.ini_window.nameLineEdit.text()
        self.cat_n = int(self.ini_window.catSpinBar.text())
        self.initUI()
        
        #security in case the user close the dialog window with invalid values
        if self.user_name == '' and self.cat_n <= 1:         
            reply = QMessageBox.question(self, 'Not good!',
                                         'Abort! Name or number of category not valid!!!!',
                                         QMessageBox.Ok, QMessageBox.Ok)
            if reply == QMessageBox.Ok:
                self.close()


    def initUI(self):
        # QDialog.__init__(self, parent)
        super().__init__()
        
        self.setWindowTitle('Classifier')
        # self.setMinimumSize(self.img_size+50,self.img_size+50)
        self.resize(self.img_size+50, self.img_size+50)
        
        self.lbl1 = QLabel("Classify image to "+str(self.cat_n)+" categorie(s)", self)
        self.lbl1.setFont(QFont('Arial', fontsize))
        self.lbl1.resize(self.img_size, fontsize+5)
        
        #load image
        if self.isdatabase:
            path = str(self.df.iloc[0].uid)+'.tif'
        else:
            path = os.path.join(self.log['Path'],
                                self.log['Name'],
                                str(self.df.iloc[0].uid)+'.tif')
        img = tifffile.imread(path)
        #some random transformation (rot90 and fliplr)
        img = np.rot90(img, k=random.choice([1,2,3,4]))
        if random.random() > 0.5: img = np.fliplr(img)
        #load the image in the 
        pixmap = QImage(img.copy().data, 100, 100, 100, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(pixmap)
        self.lbl2 = QLabel(self)
        self.lbl2.setPixmap(pixmap.scaledToHeight(self.img_size))
    
        #text value
        self.textbox1 = QLineEdit()
        self.textbox1.setFont(QFont('Arial', fontsize))
        self.textbox1.resize(60,22)
        self.textbox1.setValidator(QIntValidator(0, self.cat_n))
        
    
        # Create a button in the window
        self.button = QPushButton('Next image', self)
        self.button.setFont(QFont('Arial', fontsize))  
        self.button.resize(120, fontsize*2)
        self.button.clicked.connect(self.on_click)
       
        #top part, having a short description, the user input and the next button
        self.topLayout = QHBoxLayout()
        self.topLayout.addWidget(self.lbl1)
        self.topLayout.addWidget(self.textbox1)
        self.topLayout.addWidget(self.button)
        
        #main part, display the top part and the image
        layout = QVBoxLayout()
        topLayout = QHBoxLayout()
        topLayout.addWidget(self.lbl1)
        topLayout.addWidget(self.textbox1)
        topLayout.addWidget(self.button)
        layout.addLayout(topLayout)
        layout.addWidget(self.lbl2)
        self.setLayout(layout)
        self.exec()

        
    def next_image(self):
        if self.isdatabase:
            path = str(self.df.iloc[self.idx_df].uid)+'.tif'
        else:
            path = os.path.join(self.log['Path'],
                                self.log['Name'],
                                str(self.df.iloc[self.idx_df].uid)+'.tif')
        pixmap = QPixmap(path)
        self.lbl2.setPixmap(pixmap.scaledToHeight(self.img_size))
        
        
    def on_click(self):           
        #read the text in the textbox
        textbox1Value = self.textbox1.text()
        #reset the text in the textbox
        self.textbox1.setText("")
        
        if textbox1Value == '':
            textbox1Value = 0
        else:
            textbox1Value = int(textbox1Value)
            
        if self.idx_df < len(self.df)-1:
            self.result.append(textbox1Value)
            self.idx_df += 1
            self.next_image()
        else: #end!
            self.result.append(int(textbox1Value))
            
            #save the result in the dataframe
            if len(self.result) == len(self.df): #good size
                self.df[self.user_name+'_'+str(self.cat_n)+'_category_h'] = self.result
            else: #error message
                QMessageBox.critical(None, "CRITICAL ERROR", "Some trouble to save the data!")

            #save the dataframe in both csv and pckl
            self.df.sort_values('Folder', inplace=True, ignore_index=True)
            self.df.reset_index(drop=True)
            self.df = self.df.loc[:, ~self.df.columns.str.contains('^Unnamed')]
            if self.isdatabase:
                self.df.to_csv('database.csv')
            else:
                self.df.to_csv(self.log['Name']+os.sep+'database.csv')
            
            reply = QMessageBox.question(self, 'Program ends!', 'Congratulation!', QMessageBox.Ok, QMessageBox.Ok)
            if reply == QMessageBox.Ok:
                self.close()
                



class Window(QDialog):
    def __init__(self):
        super(Window, self).__init__()

        # setting window title
        self.setWindowTitle("Starting Randomizer")
        self.resize(100, 100)
        self.formGroupBox = QGroupBox("User")
        # creating spin box to select categorie number
        self.catSpinBar = QSpinBox()
        self.catSpinBar.setFont(QFont('Arial', fontsize))
        self.label2 = QLabel("Number of category")
        self.label2.setFont(QFont('Arial', fontsize))
        # creating a line edit
        self.nameLineEdit = QLineEdit()
        self.nameLineEdit.setFont(QFont('Arial', fontsize))
        self.label1 = QLabel("Name")
        self.label1.setFont(QFont('Arial', fontsize))
        # calling the method that create the form
        self.createForm()
 
        # creating a dialog button for ok
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok)
        # adding action when form is accepted
        self.buttonBox.accepted.connect(self.getInfo)

        # creating a vertical layout
        mainLayout = QVBoxLayout()
        # adding form group box to the layout
        mainLayout.addWidget(self.formGroupBox)
        # adding button box to the layout
        mainLayout.addWidget(self.buttonBox)
 
        # setting lay out
        self.setLayout(mainLayout)
        self.show()


    # get info method called when form is accepted
    def getInfo(self):
        if self.nameLineEdit.text() != '' and int(self.catSpinBar.text()) > 1:
            # closing the window
            self.close()
        else:
            QMessageBox.critical(None, "ERROR", "At least one field is invalid!")

    # create form method
    def createForm(self):
        # creating a form layout
        layout = QFormLayout()
        # for name and adding input text
        layout.addRow(self.label1, self.nameLineEdit)
        # for age and adding spin box
        layout.addRow(self.label2, self.catSpinBar)
        # setting layout
        self.formGroupBox.setLayout(layout)

if __name__ == "__main__":
    fontsize = 10
    
    def run_app():
        app = QtWidgets.QApplication(sys.argv)
        mainwin = MainWindow()
        mainwin.show()
        sys.exit(app.exec_())
    
    run_app()