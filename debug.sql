### manually register
#pp_this.register_info({"cmd":"INSERT INTO observation (observation_type_id,target_id,batch_size,instrument_id,obs_site_id,observer_id,bin_size) values (%s,%s,%s,%s,%s,%s,%s)","args":"[1,5,50,1,1,2,1]"})
#pp_this.register_info({"cmd":"INSERT INTO observation (observation_type_id,target_id,batch_size,instrument_id,obs_site_id,observer_id,bin_size) values (%s,%s,%s,%s,%s,%s,%s)","args":"[1,6,50,1,1,2,1]"})
#pp_this.register_info({"cmd":"INSERT INTO observation (observation_type_id,target_id,batch_size,instrument_id,obs_site_id,observer_id,bin_size) values (%s,%s,%s,%s,%s,%s,%s)","args":"[1,6,50,1,1,2,1]"})
#pp_this.register_info({"cmd":"INSERT INTO observation (observation_type_id,target_id,batch_size,instrument_id,obs_site_id,observer_id,bin_size) values (%s,%s,%s,%s,%s,%s,%s)","args":"[1,8,5,1,1,2,1]"})
#pp_this.register_info({"cmd":"INSERT INTO observation (observation_type_id,target_id,batch_size,instrument_id,obs_site_id,observer_id,bin_size) values (%s,%s,%s,%s,%s,%s,%s)","args":"[1,9,5,1,1,2,1]"})
#pp_this.register_info({"cmd":"INSERT INTO sky (target_id,ra,`dec`,fov_x,fov_y) values (%s,%s,%s,%s,%s)","args":"[8,305.2218674394,59.44877091036,0.76,0.5]"})
USE tianyudev;
SELECT * FROM target_n;
use tianyudev;
SELECT * FROM obs_site;
SELECT * FROM observer;
select * from target_n;
select * from observation LEFT JOIN target_n on observation.target_id=target_n.target_id;
#select * from obs_site;
#select * from observer;
DELETE FROM process_list where not process_status_id =5;
#Error Code: 1451. Cannot delete or update a parent row: a foreign key constraint fails (`tianyudev`.`process_dependence`, CONSTRAINT `process_dependence_ibfk_1` FOREIGN KEY (`master_process_id`) REFERENCES `process_list` (`process_id`))

SELECT * FROM data_process_site;
SELECT * FROM data_process_group;


SELECT * FROM observation;

select * from img where n_stack=1 and image_type_id=2;
DELETE FROM img where image_id <= 106;
SHOW COLUMNS FROM img;
SELECT * FROM image_type;
#DELETE FROM observation where instrument_id=1;
SELECT * FROM process_list where not process_status_id=5 ORDER BY process_id;
SELECT * FROM process_dependence;
DELETE FROM process_dependence where master_process_id>=172567522748834927793656 or dependence_process_id>=172567522748834927793656;
DELETE FROM process_list where process_id>=172567522748834927793656;
DELETE FROM process_list where process_id=172567522748834927793656;

UPDATE process_list SET process_status_id = 1 where process_id=172536517560333557525176;
SELECT * from img where image_type_id=2;
DELETE FROM img where image_type_id=2;
#SHOW TABLES;
#DROP TABLE process;
SELECT * FROM data_process_site;
SELECT * from observation;
SELECT * from img where dark_image_id=dark_image_id;
DELETE FROM img where image_id=1451;
SELECT * FROM process_type;
SELECT * FROM img where n_star_resolved=n_star_resolved;
SELECT tsp.x_template as x_template, tsp.y_template as y_template FROM img INNER JOIN tianyu_source_position as tsp on img.image_id = tsp.template_image_id WHERE img.birth_process_id = 172360570755623226984087;