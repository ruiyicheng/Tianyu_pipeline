#import mysql.connector
import time
import numpy as np
from Tianyu_pipeline.pipeline.utils import sql_interface
class process_publisher:
    def __init__(self,host_pika = '192.168.1.107',host_sql = '192.168.1.107',site_id=1,group_id = 1):
        print('connecting to db')
        self.sql_interface = sql_interface.sql_interface()
        #self.cnx = mysql.connector.connect(user='tianyu', password='tianyu',host=host_sql,database='tianyudev')
        print('done')
        self._default_site_id = site_id
        self._default_group_id = group_id
    @property
    def default_site_id(self):
        if hasattr(self,"_default_site_id"):
            return self._default_site_id

    
    @property
    def default_group_id(self):
        if hasattr(self,"_default_group_id"):
            return self._default_group_id
    def generate_PID(self):
        return (time.time_ns()+np.random.randint(0,1000))*100000+np.random.randint(0,100000)
    

    # def reduction_nighty_obs(self,obs_id,PID_sub,PID_div,sky_id):

    #     pid_cal_list = self.calibrate_observation(obs_id,PID_sub,PID_div)
    #     self.generate_template(pid_cal_list,sky_id)
    def process_TOO_obs(self,PID_bias,PID_flat,PID_raw,PID_sky): # Generate new sky template with the stacking results
        # calibrate the raw image
        sql = 'SELECT birth_process_id FROM img JOIN observation as obs on obs.obs_id = img.obs_id WHERE obs.process_id = %s and n_stack=1 and (img.image_type_id=1 or img.image_type_id=5 or img.image_type_id = 9 or img.image_type_id = 7);'
        args = (PID_bias,)
        PID_biases = self.sql_interface.query(sql,args)
        args = (PID_flat,)
        PID_flats = self.sql_interface.query(sql,args)
        args = (PID_raw,)
        sql = "SELECT obs_id from observation where observation.process_id = %s;"
        obs_id_raw = self.sql_interface.query(sql,args)
        PID_super_bias = self.stacking(list(PID_biases['birth_process_id']))
        PID_flat_debiased_list = []
        for pf in list(PID_flats['birth_process_id']):
            flat_debiased_pid = self.calibrate({'PID_cal':int(pf),'PID_sub':PID_super_bias,'subtract_bkg':0})
            PID_flat_debiased_list.append(flat_debiased_pid)
        PID_super_flat = self.stacking(PID_flat_debiased_list,method = 'flat_stacking',num_image_limit=50)
        PID_calibrated_img = self.calibrate_observation(int(obs_id_raw.loc[0,'obs_id']),PID_super_bias,PID_super_flat)
        
        # Generate new sky template with the stacking results
        stacked_PID = self.align_stack_img(PID_calibrated_img)
        # resolve targets
        sql = 'select sky_id from sky where process_id = %s;'
        args = (PID_sky,)
        sky_id = int(self.sql_interface.query(sql,args).loc[0,'sky_id'])
        resolve_star_pid = self.detect_source(stacked_PID,sky_id)
        PID_crossmatch = self.crossmatch(sky_id,dep_PIDs=[resolve_star_pid])
        PID_flux_batch = []
        for img in PID_calibrated_img:
            PID_flux_batch.append(self.extract_flux(int(img),int(resolve_star_pid)))

        PID_reference_star = self.select_reference_star(resolve_star_pid,PID_crossmatch,PID_flux_extraction_list = PID_flux_batch)
        relative_photometry_list = []
        for PID_flux in PID_flux_batch:
            relative_photometry_list.append(self.relative_photometry(PID_reference_star,PID_flux))

        return relative_photometry_list
    def relative_photometry_batch(self,obs_id):
        sql = "SELECT * FROM reference_star where obs_id = %s LIMIT 1;"
        args = (obs_id,)
        result = self.sql_interface.query(sql,args)
        PID_reference_star = int(result.loc[0,'process_id'])
        sql = "SELECT sim.birth_process_id as process_id FROM star_pixel_img as sim INNER JOIN img on img.image_id = sim.image_id where img.obs_id = %s;"
        result = self.sql_interface.query(sql,args)
        PID_extract_flux_list = list(set([int(i) for i in result['process_id']]))
        for PID_extract_flux in PID_extract_flux_list:
            self.relative_photometry(PID_reference_star,PID_extract_flux)
    def relative_photometry(self,PID_reference_star,PID_extract_flux):
        param_dict = {
            'PID_reference_star': PID_reference_star,
            'PID_extract_flux':PID_extract_flux
        }
        return self.publish_CMD(self.default_site_id, self.default_group_id, f'relative_photometry|{param_dict}', [PID_reference_star,PID_extract_flux])
    def select_reference_star(self,PID_template_generating,PID_crossmatch,PID_flux_extraction_list=[]):
        param_dict = {
            'PID_template_generating': PID_template_generating, 'PID_crossmatch': PID_crossmatch
        }
        if PID_crossmatch!=-1:
            PID_crossmatch_list =  [PID_crossmatch]
        else:
            PID_crossmatch_list = []
        return self.publish_CMD(self.default_site_id, self.default_group_id, f'select_reference_star|{param_dict}', [PID_template_generating]+PID_crossmatch_list+PID_flux_extraction_list)
    
    def extract_flux_batch(self,obs_id, resolve_id,nstack = 1):
        if nstack==1:
            image_type_id = 2
        else:
            image_type_id = 3

        sql = "SELECT * from img where n_stack= %s AND image_type_id = %s AND obs_id = %s;"
        args = (nstack,image_type_id,obs_id)
        results = self.sql_interface.query(sql,args)
        PID_extract_list = []
        if len(results)>0:
            for _,r in results.iterrows():
                PID_extract_list.append(self.extract_flux(int(r['birth_process_id']),int(resolve_id)))
        return PID_extract_list
    def align_stack_img(self,calibrated_birth_PID_list,template_birth_PID = -1):
        if template_birth_PID==-1:
            template_birth_PID = min(calibrated_birth_PID_list)
        align_pid_list = []
        for calibrated_birth_PID in calibrated_birth_PID_list:
            align_pid_list.append(self.align(template_birth_PID,calibrated_birth_PID))

        # Operation for the already-generated template 
        pid_select = self.select_good_img(align_pid_list)


        # align the template to origin template of sky
        # select the img to stack 
        # stacking new template
        stack_template_pid = self.stacking_chosen(pid_select)
        return stack_template_pid
        # syncronize the position of star       
        # crossmatch the new resolved star and origin star
        #resolve_star_pid = self.detect_source(stack_template_pid,sky_pid,as_new_template = True)

        # add new source into sky
        # self.crossmatch_new_star(resolve_star_pid)
    def select_good_img(self,align_pid_list,consume_site_id=-1,consume_group_id=-1):
        if consume_site_id==-1:
            consume_site_id=self.default_site_id
        if consume_group_id==-1:
            consume_group_id=self.default_group_id
        PID_this = self.publish_CMD(consume_site_id,consume_group_id,f'select_good_img',align_pid_list)
        return PID_this


    def detect_source(self, new_template_PID, sky_id, as_new_template = 1):
        """
        交叉匹配新解析的恒星和原始恒星
        """
        param_dict = {
            'new_template_PID': new_template_PID,
            'sky_id': sky_id,
            'as_new_template': as_new_template
        }
        return self.publish_CMD(self.default_site_id, self.default_group_id, f'detect_source|{param_dict}', [new_template_PID])
    def crossmatch(self,sky_id,dep_PIDs = []):
        param_dict = {'sky_id': sky_id}
        return self.publish_CMD(self.default_site_id, self.default_group_id, f'crossmatch|{param_dict}', dep_PIDs)
    
    def extract_flux(self,PID_img,PID_detect_source):
        PID_img = int(PID_img)
        PID_detect_source = int(PID_detect_source)
        param_dict = {
            'PID_img': PID_img,
            'PID_detect_source': PID_detect_source
        }
        return self.publish_CMD(self.default_site_id, self.default_group_id, f'extract_flux|{param_dict}', [PID_img,PID_detect_source])
    def calibrate_observation(self,obs_id,PID_sub,PID_div):
        sql = 'SELECT birth_process_id FROM img WHERE obs_id = %s and n_stack=1 and image_type_id=1;'
        args = (obs_id,)
        result = self.sql_interface.query(sql,args)
        pid_frame_list = list(result['birth_process_id'])
        pid_cal_list = []
        for PID_frame in pid_frame_list:
            pid_cal_list.append(self.calibrate({'PID_cal':int(PID_frame),'PID_sub':PID_sub,'PID_div':PID_div}))
        return pid_cal_list


    def prepare_calibration_img(self,dark_obs_id,flat_obs_id,dark_flat_obs_id):
        sql = 'SELECT birth_process_id FROM img WHERE obs_id = %s and n_stack=1;'
        args = (dark_flat_obs_id,)
        result = self.sql_interface.query(sql,args)
        pid_dark_flat = list(result['birth_process_id'])
        dark_flat_stack_pid = self.stacking(pid_dark_flat)

        sql = 'SELECT birth_process_id FROM img WHERE obs_id = %s and n_stack=1;'
        args = (flat_obs_id,)
        result = self.sql_interface.query(sql,args)
        pid_flat = list(result['birth_process_id'])
        flat_stack_pid = self.stacking(pid_flat)

        flat_debiased_pid = self.calibrate({'PID_cal':flat_stack_pid,'PID_sub':dark_flat_stack_pid,'subtract_bkg':0})

        sql = 'SELECT birth_process_id FROM img WHERE obs_id = %s and n_stack=1;'
        args = (dark_obs_id,)
        result = self.sql_interface.query(sql,args)
        pid_dark = list(result['birth_process_id'])
        dark_stack_pid = self.stacking(pid_dark)      
        
        return flat_debiased_pid,dark_stack_pid
    def calibrate(self,param_dict,consume_site_id=-1,consume_group_id=-1):
        if consume_site_id==-1:
            consume_site_id=self.default_site_id
        if consume_group_id==-1:
            consume_group_id=self.default_group_id
        PID_dep = [param_dict['PID_cal']]
        if "PID_sub" in param_dict:
            PID_dep.append(param_dict['PID_sub'])
        if "PID_div" in param_dict:
            PID_dep.append(param_dict['PID_div'])
        PID_this = self.publish_CMD(consume_site_id,consume_group_id,f'calibrate|{param_dict}',PID_dep)
        return PID_this

    def align(self,template_birth_PID,cal_birth_PID,consume_site_id=-1,consume_group_id=-1):
        if consume_site_id==-1:
            consume_site_id=self.default_site_id
        if consume_group_id==-1:
            consume_group_id=self.default_group_id 
        param_dict={'template_birth_PID':template_birth_PID,'cal_birth_PID':cal_birth_PID}
        if type(cal_birth_PID)==list:
            PID_dep_list = cal_birth_PID+[template_birth_PID]
        else:
            PID_dep_list = [cal_birth_PID]+[template_birth_PID]
        PID_align = self.publish_CMD(consume_site_id,consume_group_id,f'align|{param_dict}',PID_dep_list)
        return PID_align
    

    def transfer_img(self,param_dict,consume_site_id=-1,consume_group_id=-1):
        if consume_site_id==-1:
            consume_site_id=self.default_site_id
        if consume_group_id==-1:
            consume_group_id=self.default_group_id 
        PID_this = self.publish_CMD(consume_site_id,consume_group_id,f'transfer_img|{param_dict}',[])
        return PID_this
    def create_dir(self,param_dict,consume_site_id=-1,consume_group_id=-1):
        '''
        example: register_info({"cmd":"INSERT INTO tabname (.....) VALUES (%s,%s.....);","args":'[...]'})
        '''
        if consume_site_id==-1:
            consume_site_id=self.default_site_id
        if consume_group_id==-1:
            consume_group_id=self.default_group_id 
        PID_this = self.publish_CMD(consume_site_id,consume_group_id,f'create_dir|{param_dict}',[])
        return PID_this

    def load_UTC(self,PIDs,consume_site_id=-1,consume_group_id=-1):
        if consume_site_id==-1:
            consume_site_id=self.default_site_id
        if consume_group_id==-1:
            consume_group_id=self.default_group_id         
        PID_this = self.publish_CMD(consume_site_id,consume_group_id,f'load_UTC',PIDs)
        return PID_this

    def register_info(self,param_dict,consume_site_id=-1,consume_group_id=-1):
        '''
        example: register_info({"cmd":"INSERT INTO tabname (.....) VALUES (%s,%s.....);","args":'[...]'})
        '''
        if consume_site_id==-1:
            consume_site_id=self.default_site_id
        if consume_group_id==-1:
            consume_group_id=self.default_group_id         
        PID_this = self.publish_CMD(consume_site_id,consume_group_id,f'register|{param_dict}',[])
        return PID_this

    def stacking_chosen(self,PID_choose,consume_site_id=-1,consume_group_id=-1):
        if consume_site_id==-1:
            consume_site_id=self.default_site_id
        if consume_group_id==-1:
            consume_group_id=self.default_group_id  
        PID_align_list = self.sql_interface.get_process_dependence(PID_choose)
        PID_this = self.stacking(PID_align_list, PID_type='align',consume_site_id=consume_site_id,consume_group_id=consume_group_id,consider_goodness=1,additional_dependence_list = [PID_choose] )

        return PID_this


    def stacking(self,PIDs,PID_type='birth',method = 'mean',num_image_limit = 5,consume_site_id=-1,consume_group_id=-1,consider_goodness=0,additional_dependence_list = []):
        PIDs = [int(p) for p in PIDs]
        additional_dependence_list = [int(p) for p in additional_dependence_list]
        if consume_site_id==-1:
            consume_site_id=self.default_site_id
        if consume_group_id==-1:
            consume_group_id=self.default_group_id  
        Next_hierarchy_PID_list = []
        for i in range((len(PIDs)-1)//num_image_limit+1):
            stack_this = PIDs[i*num_image_limit:(i+1)*num_image_limit]
            if len(stack_this)!=1:
                PID_this = self.publish_CMD(consume_site_id,consume_group_id,'stack|{"PID_type":"'+PID_type+'","consider_goodness":'+str(consider_goodness)+',"method":"'+method+'"}',stack_this+additional_dependence_list)
                Next_hierarchy_PID_list.append(PID_this)
            else:
                Next_hierarchy_PID_list.append(stack_this[0])

        if len(Next_hierarchy_PID_list)>1:
            PID_ret = self.stacking(Next_hierarchy_PID_list,method = method,num_image_limit=num_image_limit,consume_site_id=consume_site_id,consume_group_id=consume_group_id,consider_goodness=consider_goodness)
        else:
            return PID_this
        return PID_ret


    def publish_CMD(self,process_site,process_group,CMD,dep_PID_list):
        PID_this = self.generate_PID()
        
        mycursor = self.sql_interface.cnx.cursor()
        sql = "INSERT INTO process_list (process_id,process_cmd,process_status_id,process_site_id,process_group_id) VALUES (%s, %s,1,%s,%s)"
        argsql = (PID_this,CMD,process_site,process_group)
        mycursor.execute(sql, argsql)
        # self.sql_interface.cnx.commit()
        for PID_dep in dep_PID_list:

            sql = "INSERT INTO process_dependence (master_process_id, dependence_process_id) VALUES (%s, %s)"
            argsql = (PID_this,PID_dep)
            mycursor.execute(sql, argsql)
        self.sql_interface.cnx.commit()
        return PID_this
    

    def test(self,item = 'stacking'):
        if item =="stacking":
            print(self.stacking(1,[self.publish_CMD(1,1,1,"capture",[]) for i in range(35)]))
if __name__=="__main__":
    pp = process_publisher()
    pp.test()
