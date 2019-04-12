# -*- coding: utf-8 -*-
"""
Tue Apr 24 08:51:10 2018: Dhiraj
"""

# Read & Write from / into File 

# pip -install path
from path import path

filename = "E:\pyWork\pyData\iitg\input\x00.txt"
with file(filename) as f:
    s = f.read()

filename
with open('x.py') as f: s = f.read()
with open(filename) as f: s = f.read()

contents = open(filename)
contents
contents = [line for line in open(filename)]


with open (filename, "r") as myfile:
    data=myfile.readlines()
data

file = open(filename, 'r')
text = file.read().strip()
file.close()

#%%
# file-output.py
f = open('helloworld.txt','w')
f.write('hello world')
f.close()

# file-input.py
f = open('helloworld.txt','r')
message = f.read()
print(message)
f.close()

#Append mode
# file-append.py
f = open('helloworld.txt','a')
f.write('\n' + 'hello world')
f.close()

# file-input.py
f = open('helloworld.txt','r')
print(f.readlines)
message = f.read()
#print(f.read(5))
print(message)

for line in f: 
    print(line)

for line in f.readlines: 
    print(line)



f.close()



#%%

#Read contents from different files 



a=str(input("Enter the name of the file with .txt extension:"))

file2=open(a,'r')
filename
file2=open(filename,'r')
line=file2.readline()
while(line!=""):
    print(line)
    line=file2.readline()
file2.close()

#%%  - working
with open("helloworld.txt") as f: 
    data = f.readlines() 
data    

#%%
with open("E:\pyWork\pyData\iitg\input\x00.txt") as f: 
    data = f.readlines() 
data   #error

#%%
x = os.listdir("./textfiles")
x
file1 =  "E:\pyWork\pyProjects\\dupthon18\\textfiles\\100.txt"
file1
with open(file1, 'r') as f:
    data = f.readlines()
 
for line in data:
words = line.split()
print words