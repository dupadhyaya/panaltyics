#hello is function written in 17b-hello1.py
#use it here by importing the function

import hello
hello.sayhello("Dhiraj", "Upadhyaya",'HOD')

# Script to check reload. Do changes in hello module first and run this
import imp
imp.reload(hello)

hello.sayhello("Dhiraj", "Upadhyaya", "Dean DS")


