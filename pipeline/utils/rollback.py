from Tianyu_pipeline.pipeline.utils import delete_undocumented_img
from Tianyu_pipeline.pipeline.utils import sql_interface
import pandas as pd
import tqdm
def rollback(obs_id,bias_id = 7,flat_id = 6,Flatbias = True):
    #obs_id used to remove reference_star
    si = sql_interface.sql_interface()
    #obtain sky_id
    sql = "select * from sky left join target_n on target_n.target_id = sky.target_id left join observation on observation.target_id = target_n.target_id where observation.obs_id = %s"
    args = (obs_id,)
    sky_id = int(si.query(sql,args).loc[0,'sky_id']) # Used to remove sky_image_map, tianyu_source

    #obtain image_id to remove
    sql = "select * from img where obs_id = %s and image_type_id != 1 order by n_stack,image_id;"
    args = (obs_id,)
    image_to_remove = si.query(sql,args).loc[:,'image_id'] #Used to remove image and image depencences
    image_to_remove = [[int(i)] for i in list(image_to_remove)]


    sql = "select * from img where obs_id = %s order by n_stack,image_id;"
    args = (obs_id,)
    image_all = si.query(sql,args).loc[:,'image_id'] #Used to remove image and image depencences
    image_all = [[int(i)] for i in list(image_all)]
    if Flatbias:
        sql = "select * from img where obs_id = %s and image_type_id != 5 order by n_stack,image_id;"
        args = (flat_id,)
        query_result = si.query(sql,args)
        query_result = query_result[query_result['n_stack']<max(query_result['n_stack'])]
        image_to_remove = image_to_remove + [[int(i)] for i in list(query_result.loc[:,'image_id'])]
        #print('flat to remove:',[[int(i)] for i in list(si.query(sql,args).loc[:,'image_id'])][:-1])
        sql = "select * from img where obs_id = %s and n_stack > 1 order by n_stack,image_id;"
        args = (bias_id,)
        query_result = si.query(sql,args)
        query_result = query_result[query_result['n_stack']<max(query_result['n_stack'])]
        image_to_remove = image_to_remove + [[int(i)] for i in list(query_result.loc[:,'image_id'])]
        #print('bias to remove:',[[int(i)] for i in list(si.query(sql,args).loc[:,'image_id'])][:-1])
    #obtain template_image_id
    sql = "select * from sky_image_map where sky_id = %s"
    args = (sky_id,)
    query_result = si.query(sql,args)
    if query_result.empty:
        print('No template image to remove')
        template_image_id = -1
    else:
        template_image_id = int(query_result.loc[0,'image_id'])
    #template_image_id = int(si.query(sql,args).loc[0,'image_id']) # Used to remove tianyu_source_position
    
    #obtain source_id
    sql = "select * from tianyu_source where sky_id = %s"
    args = (sky_id,)
    query_result = si.query(sql,args)
    if query_result.empty:
        print('No source to remove')
        source_id = []
    else:
        source_id = query_result.loc[:,'source_id']
    #source_id = si.query(sql,args).loc[:,'source_id'] # Used to remove source_crossmatch,star_pixel_img

    #Echo the extracted quantities
    print("obs_id", obs_id)
    print('sky_id',sky_id)
    print("image_to_remove")
    print(image_to_remove)
    print("source_id")
    print(source_id)

    #Remove rows according to the extracted quantities
    # First remove the entries that would not cause any foreign key constraint violation
    # Then remove the entries that would influence the above steps for the sake of debugging
    #Remove star_pixel_img
    si.execute('SET foreign_key_checks = 0;',[])
    sql = "delete from star_pixel_img where source_id = %s"
    source_id_list = [int(i) for i in list(source_id)]
    # args = [[i] for i in source_id_list]
    print('removing star_pixel_img')
    # si.executemany(sql,args)
    for i in tqdm.tqdm(source_id_list):
        args = (i,)
        si.execute(sql,args)
    si.execute('SET foreign_key_checks = 1;',[])
    #Remove source_crossmatch
    print('removing crossmatch')
    si.execute('SET foreign_key_checks = 0;',[])
    sql = "delete from source_crossmatch where source_id = %s"
    for i in tqdm.tqdm(source_id_list):
        args = (i,)
        si.execute(sql,args)
    # args = [[i] ]
    # si.executemany(sql,args)
    si.execute('SET foreign_key_checks = 1;',[])
    #Remove source_position
    if template_image_id>0:
        print('removing tianyu_source_position')
        sql = "delete from tianyu_source_position where template_img_id = %s"
        args = (template_image_id,)
        si.execute(sql,args)

    #Remove reference_star 
    print('removing reference_star')
    sql = "delete from reference_star where obs_id = %s;"
    args = (obs_id,)
    si.execute(sql,args)
    
    #Remove tianyu_source
    print('removing tianyu_source')
    sql = "delete from tianyu_source where sky_id = %s"
    args = (sky_id,)
    si.execute(sql,args)

    #Remove sky_image_map
    print('removing sky_image_map')
    sql = "delete from sky_image_map where sky_id = %s"
    args = (sky_id,)
    si.execute(sql,args)

    #Remove image dependence
    print('removing image dependence')
    sql = "delete from img_stacking where image_id = %s or stacked_id = %s;"
    args = [[i[0],i[0]] for i in image_to_remove]
    si.executemany(sql,args)

    #Remove img self-dependence
    print('removing img self-dependence')
    for args in [image_to_remove,image_all]:
        
        sql = """
            UPDATE img
            SET align_target_image_id = NULL,flat_image_id = NULL,dark_image_id = NULL
            WHERE image_id = %s
        """
        #args = image_to_remove
        si.executemany(sql,args)
    #cursor.execute(update_dependencies_query, (image_id_to_delete,))

    #Remove image
    print('removing image')
    sql = "delete from img where image_id = %s"
    args = image_to_remove
    si.executemany(sql,args)

    delete_undocumented_img.deleter().files()











