# -*- coding: utf-8 -*-
"""
Tue Apr 24 07:59:01 2018: Dhiraj
"""

#Read list of files in directory
from os import listdir
from os.path import isfile, join
mypath= "E:/pyWork/pyData/iitg/input/"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
onlyfiles

from glob import glob
glob("E:/pyWork/pyData/iitg/input/*")


import os
os.listdir("E:/pyWork/pyData/iitg/input/")


#full path names
import os
mypath = "E:/pyWork/pyData/iitg/input/"
files_path = [os.path.abspath(x) for x in os.listdir(mypath)]
files_path

#%%
# current dir
import os
arr = os.listdir('.')
arr


#one dir up
# method 1
x = os.listdir('..')
x
# method 2 up dir  on level
x= os.listdir('/')
x

#particular folder
import os
arr = os.listdir('E:\\pyWork')
arr

#%%
#particular sub dir
import os

x = os.listdir("./class")
x
#%%
#current dir
import os
arr = next(os.walk('.'))[2]
arr

#all files & folders
import glob
print(glob.glob("*"))

#%%
#next(os.walk('.')) and os.path.join('dir','file')
#NOT WORKING
import os
arr = []
for d,r,f in next(os.walk("E:\pyWork")):
    for file in f:
        arr.append(os.path.join(r,file))
for f in arr:
    print(files)
arr
#----
#%%
#all files in dir
x = glob.glob("*")
x
#https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory

#os.walk
import os
mypath = "E:/pyWork/pyData/iitg" 
for root, dirs, files in os.walk(mypath, topdown=False):
   for name in files:
      print(os.path.join(root, name))
   for name in dirs:
      print(os.path.join(root, name))
#%%      
#----! this is the code 
mypath= "E:/pyWork/pyData/iitg/input/"
def read_directory(mypath):
    current_list_of_files = []
    while True:
        for (dirname, dirnames, filenames) in os.walk(mypath):
            current_list_of_files = filenames
        logging.info("Reading the directory for the list of file names")
        return current_list_of_files
read_directory(mypath)
#

#%%


     
fileList = []
count = 0
filenames=[]
directorypath = "E:/pyWork/pyData/iitg/input/"

for dirname, dirnames, filesnames in os.walk(directorypath):
    for filename in filenames:
       if filename.endswith(".txt"):
          fileList.append(filename)
          count = count + 1
count
filenames
#%%

import os
os.path.abspath("textfiles/100.txt")

from path import path
path('textfiles/100.txt').abspath()

#pip install pathlib
from pathlib import Path
p = Path("git.exe").resolve()
p


#%% List all files in current folder
import glob
for x in glob.glob('*.txt'):
    print(x)
    
# different folder
import glob
for x in glob.glob('E:/pyWork/pyProjects/dupython18/textfiles/*'):
    print(x)