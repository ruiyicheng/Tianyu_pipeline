DROP DATABASE tianyudev;
CREATE database tianyudev;
use tianyudev;

CREATE TABLE data_process_site(
	process_site_id INT UNIQUE NOT NULL PRIMARY KEY AUTO_INCREMENT,
    file_path TEXT,
    process_site_name TEXT,
    process_site_ip TEXT,
    process_site_flag INT,
    user_name TEXT
);
#alter table data_process_site ADD column user_name TEXT;
#alter table data_process_site  drop process_site_root_path;
#show columns from data_process_site;
#DROP TABLE data_process_group;
CREATE TABLE data_process_group(
	process_group_id INT UNIQUE NOT NULL PRIMARY KEY AUTO_INCREMENT,
    process_site_id INT NOT NULL,
    property_flag INT DEFAULT 0,
    FOREIGN KEY (process_site_id) REFERENCES data_process_site(process_site_id)
);

CREATE TABLE process_type(
    process_status_id INT UNIQUE NOT NULL PRIMARY KEY AUTO_INCREMENT,
    process_status TEXT
);






CREATE TABLE process_list(
    process_id DECIMAL(25) UNIQUE NOT NULL PRIMARY KEY,
    process_cmd TEXT,
    process_status_id INT,
    process_site_id INT,
    process_group_id INT,
    FOREIGN KEY (process_status_id) REFERENCES process_type(process_status_id),
    FOREIGN KEY (process_site_id) REFERENCES data_process_site(process_site_id),
    FOREIGN KEY (process_group_id) REFERENCES data_process_group(process_group_id)
);
#select * from process_list;
#DELETE FROM process_list where process_status_id = 2;
#DROP TABLE process_dependence;
CREATE TABLE process_dependence(
    master_process_id DECIMAL(25),
    dependence_process_id DECIMAL(25),
    FOREIGN KEY (master_process_id) REFERENCES process_list(process_id),
    FOREIGN KEY (dependence_process_id) REFERENCES process_list(process_id)
);




CREATE TABLE observation_type(
    observation_type_id INT UNIQUE NOT NULL PRIMARY KEY AUTO_INCREMENT,
    observation_type_name TEXT
);


CREATE TABLE filters(
    filter_id INT UNIQUE NOT NULL PRIMARY KEY AUTO_INCREMENT,
    filter_name TEXT
);

CREATE TABLE instrument(
    instrument_id INT UNIQUE NOT NULL PRIMARY KEY AUTO_INCREMENT,
    instrument_name TEXT,
    x_resolution INT,
    y_resolution INT,
    filter_id INT,
    local_folder_path TEXT,
    FOREIGN KEY (filter_id) REFERENCES filters(filter_id)
);
#DROP TABLE obs_site;
CREATE TABLE obs_site(
	obs_site_id INT UNIQUE NOT NULL PRIMARY KEY AUTO_INCREMENT,
    process_site_id INT,
    obs_site_name TEXT,
    obs_site_lon DOUBLE DEFAULT NULL,
    obs_site_lat DOUBLE DEFAULT NULL,
    obs_site_height DOUBLE DEFAULT NULL,
    FOREIGN KEY (process_site_id) REFERENCES data_process_site(process_site_id) 
);



 #INSERT INTO data_process_site (process_site_name,process_site_ip,mysql_user_name,mysql_user_psw) values ('macbook','127.0.0.1','root','root');


#INSERT INTO observation_type (observation_type_name) VALUES ("science");
#INSERT INTO observation_type (observation_type_name) VALUES ("outreach");

#INSERT INTO data_process_site (file_path,process_site,process_site_ip,mysql_user_name,mysql_user_psw,is_main_site) values ();

CREATE TABLE observer_type(
    observer_type_id INT UNIQUE NOT NULL PRIMARY KEY AUTO_INCREMENT,
    observer_type TEXT
);

CREATE TABLE observer(
    observer_id INT UNIQUE NOT NULL PRIMARY KEY AUTO_INCREMENT,
    observer_name TEXT,
    observer_type_id INT DEFAULT NULL,
    FOREIGN KEY (observer_type_id) REFERENCES observer_type(observer_type_id) 
);

CREATE TABLE target_type(
    target_type_id INT UNIQUE NOT NULL PRIMARY KEY AUTO_INCREMENT,
    target_type TEXT
);

CREATE TABLE target_n(
    target_id INT UNIQUE NOT NULL PRIMARY KEY AUTO_INCREMENT,
    target_name TEXT,
    target_type_id INT,
    FOREIGN KEY (target_type_id) REFERENCES target_type(target_type_id)
);

CREATE TABLE observation(
    obs_id INT UNIQUE NOT NULL PRIMARY KEY AUTO_INCREMENT,
    observation_type_id INT,
    target_id INT DEFAULT NULL,
    n_pic INT,
    batch_size INT,
    instrument_id INT DEFAULT NULL,
    obs_site_id INT DEFAULT NULL,
    observer_id INT  DEFAULT NULL,
    process_id DECIMAL(25),
    bin_size INT DEFAULT 1,
    FOREIGN KEY (process_id) REFERENCES process_list(process_id),
    FOREIGN KEY (observation_type_id) REFERENCES observation_type(observation_type_id),
    FOREIGN KEY (target_id) REFERENCES target_n(target_id),
    FOREIGN KEY (instrument_id) REFERENCES instrument(instrument_id) ,
    FOREIGN KEY (obs_site_id) REFERENCES obs_site(obs_site_id) ,
    FOREIGN KEY (observer_id) REFERENCES observer(observer_id)
);

#--INSERT INTO observation (observation_type_id,target_id,batch_size,instrument_id,obs_site_id,observer_id,bin_size) VALUES (%s,%s,%s,%s,%s,%s,%s);
#select * from target_n;
CREATE TABLE image_type(
    image_type_id INT UNIQUE NOT NULL PRIMARY KEY AUTO_INCREMENT,
    image_type TEXT
);


CREATE TABLE img(
    image_id BIGINT UNIQUE NOT NULL PRIMARY KEY AUTO_INCREMENT,
    store_site_id INT,
    jd_utc_start DOUBLE DEFAULT NULL,
    jd_utc_mid DOUBLE DEFAULT NULL,
    jd_utc_end DOUBLE DEFAULT NULL,
    bjd_tdb_start_approximation DOUBLE DEFAULT NULL,
    bjd_tdb_mid_approximation DOUBLE DEFAULT NULL,
    bjd_tdb_end_approximation DOUBLE DEFAULT NULL,
    n_stack INT DEFAULT 1,
    n_star_resolved INT,
    batch INT DEFAULT 1,
    processed BOOLEAN DEFAULT 0, 
    used_as_template BOOLEAN DEFAULT 0, 
    image_type_id INT,
    flat_image_id BIGINT DEFAULT NULL,
    dark_image_id BIGINT DEFAULT NULL,
    mask_image_id BIGINT DEFAULT NULL,
    align_target_image_id BIGINT DEFAULT NULL,
    x_to_template INT DEFAULT NULL,
    y_to_template INT DEFAULT NULL,
    obs_id INT DEFAULT NULL,
    img_name TEXT DEFAULT NULL,
    birth_process_id DECIMAL(25),
    align_process_id DECIMAL(25),
    deleted BOOLEAN DEFAULT 0,
    is_mask BOOLEAN DEFAULT 0,
    FOREIGN KEY (store_site_id) REFERENCES data_process_site(process_site_id),
    FOREIGN KEY (birth_process_id) REFERENCES process_list(process_id),
    FOREIGN KEY (align_process_id) REFERENCES process_list(process_id),
    FOREIGN KEY (image_type_id) REFERENCES image_type(image_type_id),
    FOREIGN KEY (align_target_image_id) REFERENCES img(image_id),
    FOREIGN KEY (flat_image_id) REFERENCES img(image_id),
    FOREIGN KEY (dark_image_id) REFERENCES img(image_id),
    FOREIGN KEY (obs_id) REFERENCES observation(obs_id), 
    INDEX(n_stack),
    INDEX(jd_utc_mid)
);

#SHOW COLUMNS from img;
#ALTER TABLE img DROP column img_path;
#ALTER TABLE img ADD column img_name TEXT;
#ALTER TABLE img ADD column align_target_image_id BIGINT;
#ALTER TABLE img ADD FOREIGN KEY (align_target_image_id) REFERENCES img(image_id);
#ALTER TABLE img ADD column used_as_template BOOLEAN DEFAULT 0;
CREATE TABLE img_stacking(
    image_id BIGINT  NOT NULL,
    stacked_id BIGINT NOT NULL,
    FOREIGN KEY (image_id) REFERENCES img(image_id),
    FOREIGN KEY (stacked_id) REFERENCES img(image_id)
);

#--deg
#--clockwise x is ra+ when 
#-- NOT NULL DEFAULT ST_SRID(POINT(0,0),4326),SPATIAL INDEX(fov_pos),




CREATE TABLE sky(
    sky_id BIGINT UNIQUE NOT NULL PRIMARY KEY AUTO_INCREMENT,
    target_id INT DEFAULT NULL,
    ra DOUBLE DEFAULT NULL,
    `dec` DOUBLE DEFAULT NULL,
    fov_x DOUBLE DEFAULT NULL,
    fov_y DOUBLE DEFAULT NULL,
    scan_angle DOUBLE DEFAULT 0,
    fov_pos GEOMETRY SRID 4326,
    process_id DECIMAL(25),
    FOREIGN KEY (process_id) REFERENCES process_list(process_id),
    FOREIGN KEY (target_id) REFERENCES target_n(target_id)
);


CREATE TABLE sky_image_map(
sky_id BIGINT,
image_id BIGINT,
template_in_use BOOLEAN DEFAULT 0,
absolute_deviation_x INT DEFAULT 0,
absolute_deviation_y INT DEFAULT 0,
FOREIGN KEY (sky_id) REFERENCES sky(sky_id),
FOREIGN KEY (image_id) REFERENCES img(image_id)
);

#DROP TABLE sky_image_map;
#ALTER TABLE sky_image_map ADD COLUMN absolute_deviation_x INT DEFAULT 0;
#ALTER TABLE sky_image_map ADD COLUMN absolute_deviation_y INT DEFAULT 0;
CREATE TABLE source_type(
    source_type_id INT UNIQUE NOT NULL PRIMARY KEY AUTO_INCREMENT,
    source_type TEXT
);




CREATE TABLE tianyu_source(
    source_id BIGINT UNIQUE NOT NULL PRIMARY KEY AUTO_INCREMENT,
    source_type_id INT NOT NULL,
    sky_id BIGINT,
    FOREIGN KEY (source_type_id) REFERENCES source_type(source_type_id),
    FOREIGN KEY (sky_id) REFERENCES sky(sky_id)
);


CREATE TABLE tianyu_source_position(
    source_id BIGINT NOT NULL,
    template_img_id BIGINT NOT NULL,
    x_template DOUBLE,
    y_template DOUBLE,
    flux_template DOUBLE,
    e_flux_template FLOAT,
    INDEX(flux_template) USING BTREE,
    FOREIGN KEY (source_id) REFERENCES tianyu_source(source_id),
    FOREIGN KEY (template_img_id) REFERENCES img(image_id)
);

CREATE TABLE star_pixel_img(
    star_pixel_img_id BIGINT UNIQUE NOT NULL PRIMARY KEY AUTO_INCREMENT,
    source_id BIGINT,
    image_id BIGINT,
    flux_raw DOUBLE,
    flux_relative DOUBLE,
    flux_normalized DOUBLE,
    flux_raw_error DOUBLE,
    flux_relative_error DOUBLE,
    flux_normalized_error DOUBLE,
    mag_calibrated_instrument DOUBLE,
    e_mag_calibrated_instrument DOUBLE,
    mag_calibrated_absolute DOUBLE,
    e_mag_calibrated_absolute DOUBLE,    
    bjd_tdb_start DOUBLE DEFAULT NULL,
    bjd_tdb_mid DOUBLE DEFAULT NULL,
    bjd_tdb_end DOUBLE DEFAULT NULL,
    is_reference BOOLEAN DEFAULT 0,
    birth_process_id DECIMAL(25),
    relative_process_id DECIMAL(25),
    normalization_process_id DECIMAL(25),
    mag_calibration_process_id DECIMAL(25),
    timing_process_id DECIMAL(25),
    FOREIGN KEY (birth_process_id) REFERENCES process_list(process_id),
    FOREIGN KEY (relative_process_id) REFERENCES process_list(process_id),
    FOREIGN KEY (normalization_process_id) REFERENCES process_list(process_id),
    FOREIGN KEY (mag_calibration_process_id) REFERENCES process_list(process_id),
    FOREIGN KEY (timing_process_id) REFERENCES process_list(process_id),
    INDEX(bjd_tdb_start),
    INDEX(bjd_tdb_mid),
    INDEX(bjd_tdb_end),
    INDEX(flux_raw),
    INDEX(flux_relative),
    INDEX(mag_calibrated_absolute),
    FOREIGN KEY (source_id) REFERENCES tianyu_source(source_id),
    FOREIGN KEY (image_id) REFERENCES img(image_id)
);


CREATE TABLE reference_star(
    obs_id INT,
    source_id BIGINT,
   FOREIGN KEY (obs_id) REFERENCES observation(obs_id),
    FOREIGN KEY (source_id) REFERENCES tianyu_source(source_id)
);



CREATE TABLE source_crossmatch(
    gdr3_id BIGINT,
    panstarr_id BIGINT,
    source_id BIGINT,
    FOREIGN KEY (source_id) REFERENCES tianyu_source(source_id)
);
#DROP TABLE source_crossmatch;
#--site flag: 1:is_sql_server; 2:is pika server; 4: store the image; 8:telescope control; 16 data process center; 32 mission publisher; 64 visialization center
#--group flag: 1: gpu; 2: large memory;

