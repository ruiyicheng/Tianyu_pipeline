import mysql.connector
import pandas as pd
class sql_interface:
    def __init__(self,user='tianyu', password='tianyu',
                            host='192.168.1.107',
                            database='tianyudev'):
        self.user = user
        self.password = password
        self.host = host
        self.database = database

        self.cnx = mysql.connector.connect(user='tianyu', password='tianyu',
                            host='192.168.1.107',
                            database='tianyudev')
    # @property
    # def cnx(self):
    #     return mysql.connector.connect(user='tianyu', password='tianyu',host='192.168.1.107',database='tianyudev')
    @property
    def observation_type_id(self):
        if not hasattr(self,"_observation_type_id"):
            self._observation_type_id = self.get_table_dict("observation_type")
            return self._observation_type_id
    
    @property
    def target_type_id(self):
        if not hasattr(self,"_target_type_id"):
            self._target_type_id = self.get_table_dict("target_type")
            return self._target_type_id
    @property
    def image_type_id(self):
        if not hasattr(self,"_image_type_id"):
            self._image_type_id = self.get_table_dict("image_type")
            return self._image_type_id
    @property
    def instrument_id(self):
        if not hasattr(self,"_instrument_id"):
            self._instrument_id = self.get_table_dict("instrument")
            return self._instrument_id
    @property
    def obs_site_id(self):
        if not hasattr(self,"_obs_site_id"):
            self._obs_site_id = self.get_table_dict("obs_site")
        return self._obs_site_id
    @property
    def observer_id(self):
        if not hasattr(self,"_observer_id"):
            self._observer_id = self.get_table_dict("observer")
            return self._observer_id  
             
    def get_table_dict(self,table,index_key=1,index_value=0,db = 'tianyudev'):
        with mysql.connector.connect(user=self.user, password=self.password,
                                    host=self.host,
                                    database = db) as cnx:
                    with cnx.cursor() as mycursor:
                        mycursor.execute("SELECT * from "+table+";")
                        myresult = mycursor.fetchall()
            # print(myresult)
        res_dict = {}
        for row in myresult:
            res_dict[row[index_key]] = row[index_value]
        return res_dict
    def execute(self, sql, args, return_last_id=False,db = 'tianyudev'):
        with mysql.connector.connect(user=self.user, password=self.password,
                            host=self.host,
                            database = db) as cnx:
            with cnx.cursor() as mycursor:
                mycursor.execute(sql, args)
                cnx.commit()
                if return_last_id:
                    mycursor.execute("SELECT LAST_INSERT_ID()")
                    last_id = mycursor.fetchone()[0]
                    return last_id
                
    def executemany(self, sql, args,db = 'tianyudev'):
        with mysql.connector.connect(user=self.user, password=self.password,
                            host=self.host,
                            database = db) as cnx:
            with cnx.cursor() as mycursor:
                mycursor.executemany(sql, args)
                cnx.commit()
    def querytemp(self,sql_create,sql_insert,arg_insert,sql_query,arg_query,return_df = True,db = 'tianyudev'):
        if "TEMPORARY" not in sql_create.upper():
            print('CREATE TEMPORARY TABLE should be used!')
            return -1
        with mysql.connector.connect(user=self.user, password=self.password,
                            host=self.host,
                            database = db) as cnx:
            with cnx.cursor() as mycursor:
                mycursor.execute(sql_create)
                mycursor.executemany(sql_insert,arg_insert)
                cnx.commit()
                mycursor.execute(sql_query,arg_query)
                myresult = mycursor.fetchall()
                headers = [i[0] for i in mycursor.description]
                if return_df:
                    df = pd.DataFrame(myresult)
                    if len(df)>0:
                        df.columns = headers
                    return df
                return myresult,headers
    def query(self,sql,args,return_df = True,db = 'tianyudev'):
        with mysql.connector.connect(user=self.user, password=self.password,
                            host=self.host,
                            database = db) as cnx:
            with cnx.cursor() as mycursor:
                # self.cnx.commit()
                # mycursor = self.cnx.cursor()
                mycursor.execute(sql,args)
                myresult = mycursor.fetchall()
                headers = [i[0] for i in mycursor.description]
        #print(myresult)
        if return_df:
            df = pd.DataFrame(myresult)
            if len(df)>0:
                df.columns = headers
            return df
        return myresult,headers
    
    def get_process_dependence(self,PID,pid_type = "master"):
        args = (PID,)
        if pid_type=="master":
            sql = "SELECT * from process_dependence WHERE master_process_id = %s;"
        else:
            sql = "SELECT * from process_dependence WHERE dependence_process_id = %s;"
        result = self.query(sql,args).to_dict('list')
        if pid_type=="master":
            return result['dependence_process_id']
        else:
            return result['master_process_id']

class sql_caller:
    # This is a class with the attribute of a sql_caller object
    # The process consumer, publisher and manager would inherit this class
    def __init__(self,user='tianyu', password='tianyu',
                            host='192.168.1.107',
                            database='tianyudev'):
        self.sql_interface = sql_interface(user=user,password=password,host=host,database=database)
