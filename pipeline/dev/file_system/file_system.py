import os
import mysql.connector

class file_system:
    # create a file system if not created
    # detect if a folder exist
    # return the path of a file according to parameters passed
    # provide services that managing file system 
    def __init__(self,host_pika = 'localhost',host_sql = 'localhost'):
        self.cnx = mysql.connector.connect(user='root', password='root',
                              host=host_sql,
                              database='tianyudev',port = 8889)
    
    def init_file_system(self,par):
        pass