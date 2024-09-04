

#DROP TABLE source_crossmatch;
#--site flag: 1:is_sql_server; 2:is pika server; 4: store the image; 8:telescope control; 16 data process center; 32 mission publisher; 64 visialization center
#--group flag: 1: gpu; 2: large memory;






INSERT INTO data_process_site (process_site_name,process_site_ip,process_site_flag,file_path,user_name) VALUES ('macbook_rui','192.168.1.102',96,'/Users/ruiyicheng/Documents/code/projects/TianYu/debug_Tianyu_file_system','ruiyicheng');
INSERT INTO data_process_site (process_site_name,process_site_ip,process_site_flag,file_path,user_name) VALUES ('control_desktop_wsl','172.20.119.167',40,'/mnt/d/Tianyu_data','root');
INSERT INTO data_process_site (process_site_name,process_site_ip,process_site_flag,file_path,user_name) VALUES ('data_desktop','192.168.1.107',55,'/media/test/nf/mgo_data','test');
#INSERT INTO data_process_site (process_site_name,process_site_ip,process_site_flag,file_path,user_name) VALUES ('control_desktop_windows','172.20.119.167',40,'/mnt/d/Tianyu_data','root');

#UPDATE data_process_site SET process_site_ip="192.168.1.100" where process_site_name='macbook_rui';
#UPDATE data_process_site SET process_site_ip="172.20.119.167" where process_site_name='control_desktop';

#select * from data_process_site;

INSERT INTO filters (filter_name) values ("L");
INSERT INTO filters (filter_name) values ("R");
INSERT INTO filters (filter_name) values ("G");
INSERT INTO filters (filter_name) values ("B");
INSERT INTO filters (filter_name) values ("Halpha");
INSERT INTO filters (filter_name) values ("OIII");
INSERT INTO filters (filter_name) values ("SII");
INSERT INTO filters (filter_name) values ("Solar");
INSERT INTO filters (filter_name) values ("Dark");
INSERT INTO filters (filter_name) values ("None");

INSERT INTO image_type (image_type) values ("raw");
INSERT INTO image_type (image_type) values ("calibrated_single");
INSERT INTO image_type (image_type) values ("calibrated_stacked");
INSERT INTO image_type (image_type) values ("calibrated_difference");
INSERT INTO image_type (image_type) values ("flat_raw");
INSERT INTO image_type (image_type) values ("flat_debiased");
INSERT INTO image_type (image_type) values ("dark");
INSERT INTO image_type (image_type) values ("dark_flat");
INSERT INTO image_type (image_type) values ("bias");
INSERT INTO image_type (image_type) values ("mask");
INSERT INTO image_type (image_type) values ("background_rms");
INSERT INTO image_type (image_type) values ("psf");



#SELECT * from image_type;
INSERT INTO instrument (instrument_name,filter_id) VALUES ("L350+QHY600m",1);
#select * from instrument;

INSERT INTO obs_site (obs_site_name,obs_site_lon,obs_site_lat,obs_site_height,process_site_id) VALUES ("TDLI_MGO",121.60805556,31.164722222,1,2);
#UPDATE obs_site SET process_site_id = 2;
#select * from observation_type;
INSERT INTO observation_type (observation_type_name) VALUES ("science");
INSERT INTO observation_type (observation_type_name) VALUES ("outreach");

INSERT INTO observer_type (observer_type) VALUES ("researcher");
INSERT INTO observer_type (observer_type) VALUES ("visitor");
INSERT INTO observer_type (observer_type) VALUES ("robot");

INSERT INTO observer (observer_name,observer_type_id) VALUES ("robot",3);
INSERT INTO observer (observer_name,observer_type_id) VALUES ("Yicheng Rui",1);

INSERT INTO target_type (target_type) VALUES ("calibration");
INSERT INTO target_type (target_type) VALUES ("star_field");
INSERT INTO target_type (target_type) VALUES ("moon");
INSERT INTO target_type (target_type) VALUES ("planet");
INSERT INTO target_type (target_type) VALUES ("sun");
INSERT INTO target_type (target_type) VALUES ("cluster");
INSERT INTO target_type (target_type) VALUES ("nebula");
INSERT INTO target_type (target_type) VALUES ("galaxy");

select target_type_id as tgid from target_type where target_type="calibration";


INSERT INTO target_n (target_name, target_type_id) VALUES ("sun",(SELECT target_type_id FROM target_type where target_type.target_type = 'sun' LIMIT 1));
INSERT INTO target_n (target_name, target_type_id) VALUES ("moon",(SELECT target_type_id FROM target_type where target_type.target_type = 'moon' LIMIT 1));
INSERT INTO target_n (target_name, target_type_id) VALUES ("jupiter",(SELECT target_type_id FROM target_type where target_type.target_type = 'planet' LIMIT 1));
INSERT INTO target_n (target_name, target_type_id) VALUES ("saturn",(SELECT target_type_id FROM target_type where target_type.target_type = 'planet' LIMIT 1));
INSERT INTO target_n (target_name, target_type_id) VALUES ("flat",(SELECT target_type_id FROM target_type where target_type.target_type = 'calibration' LIMIT 1));
INSERT INTO target_n (target_name, target_type_id) VALUES ("dark",(SELECT target_type_id FROM target_type where target_type.target_type = 'calibration' LIMIT 1));
INSERT INTO target_n (target_name, target_type_id) VALUES ("bias",(SELECT target_type_id FROM target_type where target_type.target_type = 'calibration' LIMIT 1));
#INSERT INTO target_n (target_name, target_type_id) VALUES ("HAT-P-20",(SELECT target_type_id FROM target_type where target_type.target_type = 'star_field' LIMIT 1));

INSERT INTO target_n (target_name, target_type_id) VALUES ("TrES5",(SELECT target_type_id FROM target_type where target_type.target_type = 'star_field' LIMIT 1));
INSERT INTO target_n (target_name, target_type_id) VALUES ("HAT-P-7",(SELECT target_type_id FROM target_type where target_type.target_type = 'star_field' LIMIT 1));

#select * from target_n;
#select * from obs_site;
#select * from observer;
INSERT INTO data_process_group (process_site_id) VALUES (1);
INSERT INTO data_process_group (process_site_id) VALUES (2);
INSERT INTO data_process_group (process_site_id) VALUES (3);
#SELECT * FROM data_process_site;
#SELECT * FROM data_process_group;
#UPDATE data_process_site SET process_site_ip="127.0.1.1" where process_site_id=2;

