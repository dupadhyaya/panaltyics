# -*- coding: utf-8 -*-
"""
Tue Apr 24 07:56:52 2018: Dhiraj
"""
import logging
import os


# Reads and returns the list of files from a directory
mypath= "E:/pyWork/pyData/iitg/input/"
def read_directory(mypath):
    current_list_of_files = []
    while True:
        for (_, _, filenames) in os.walk(mypath):
            current_list_of_files = filenames
        logging.info("Reading the directory for the list of file names")
        return current_list_of_files


# Function you will be working with
def creating_subclusters(list_of_terms, name_of_file):
    # Your code that converts the cluster into subclusters and saves the output in the output folder with the same name as input file
    # Note the writing to file has to be handled by you.

    pass


# Main function
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

    # Folder where the input files are present
    mypath = "input"
    list_of_input_files = read_directory(mypath)
    for each_file in list_of_input_files:
        with open(os.path.join(mypath, each_file), "r") as f:
            file_contents = f.read()
        list_of_term_in_cluster = file_contents.split()

        # Sending the terms to be converted to subclusters in your code
        creating_subclusters(list_of_term_in_cluster, each_file)


        # End of code
