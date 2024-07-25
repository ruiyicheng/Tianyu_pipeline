use tianyudev;
CREATE TABLE process(
    process_id DECIMAL(25) UNIQUE NOT NULL PRIMARY KEY,
    process_cmd TEXT,
    process_status_id INT,
    process_site_id INT,
    process_group_id INT,
    FOREIGN KEY (process_status_id) REFERENCES process_type(process_status_id),
    FOREIGN KEY (process_site_id) REFERENCES data_process_site(process_site_id),
    FOREIGN KEY (process_group_id) REFERENCES data_process_group(process_group_id)
);


DROP TABLE process;
CREATE TABLE process_dependence(
    master_process_id DECIMAL(25),
    dependence_process_id DECIMAL(25),
    FOREIGN KEY (master_process_id) REFERENCES process(process_id),
    FOREIGN KEY (dependence_process_id) REFERENCES process(process_id)
);

DROP TABLE process_dependence;
INSERT INTO process_type (process_status) VALUES ("WAITING");
INSERT INTO process_type (process_status) VALUES ("IN QUEUE");
INSERT INTO process_type (process_status) VALUES ("RUNNING");
INSERT INTO process_type (process_status) VALUES ("FAIL");
INSERT INTO process_type (process_status) VALUES ("FINISH");

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
    filter_id INT,
    local_folder_path TEXT,
    FOREIGN KEY (filter_id) REFERENCES filters(filter_id) 
);

CREATE TABLE obs_site(
    process_site_id INT,
    obs_site_name TEXT,
    obs_site_lon DOUBLE DEFAULT NULL,
    obs_site_lat DOUBLE DEFAULT NULL,
    obs_site_height DOUBLE DEFAULT NULL,
    FOREIGN KEY (process_site_id) REFERENCES data_process_site(process_site_id) 
);


CREATE TABLE data_process_group(
	process_group_id INT UNIQUE NOT NULL PRIMARY KEY AUTO_INCREMENT,
    process_site_id INT NOT NULL,
    is_GPU BOOLEAN DEFAULT 0
);


CREATE TABLE data_process_site(
	process_site_id INT UNIQUE NOT NULL PRIMARY KEY AUTO_INCREMENT,
    file_path TEXT,
    process_site_name TEXT,
    process_site_ip TEXT,
    mysql_user_name TEXT,
    mysql_user_psw TEXT
);
 INSERT INTO data_process_site (process_site_name,process_site_ip,mysql_user_name,mysql_user_psw) values ('macbook','127.0.0.1','root','root');

CREATE TABLE data_process_site_info(
	process_site_id INT NOT NULL,
    site_status_id INT NOT NULL
);

CREATE TABLE data_process_site_status(
	site_status_id INT UNIQUE NOT NULL PRIMARY KEY AUTO_INCREMENT,
    site_status TEXT
);

INSERT INTO data_process_site_status (site_status) VALUES ("is_img_store_site");
INSERT INTO data_process_site_status (site_status) VALUES ("is_sql_site");
INSERT INTO data_process_site_status (site_status) VALUES ("is_pika_site");
INSERT INTO data_process_site_status (site_status) VALUES ("is_obs_site");
INSERT INTO data_process_site_status (site_status) VALUES ("is_visualization_site");
INSERT INTO data_process_site_status (site_status) VALUES ("is_data_processing_site");

INSERT INTO observation_type (observation_type_name) VALUES ("science");
INSERT INTO observation_type (observation_type_name) VALUES ("outreach");

INSERT INTO data_process_site (file_path,process_site,process_site_ip,mysql_user_name,mysql_user_psw,is_main_site) values ();

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
    process_id DECIMAL(25),
    FOREIGN KEY (target_type_id) REFERENCES target_type(target_type_id),
    FOREIGN KEY (process_id) REFERENCES process(process_id)
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
    FOREIGN KEY (process_id) REFERENCES process(process_id),
    FOREIGN KEY (observation_type_id) REFERENCES observation_type(observation_type_id),
    FOREIGN KEY (target_id) REFERENCES target_n(target_id),
    FOREIGN KEY (instrument_id) REFERENCES instrument(instrument_id) ,
    FOREIGN KEY (obs_site_id) REFERENCES obs_site(obs_site_id) ,
    FOREIGN KEY (observer_id) REFERENCES observer(observer_id)
);

CREATE TABLE image_type(
    image_type_id INT UNIQUE NOT NULL PRIMARY KEY AUTO_INCREMENT,
    image_type TEXT
);

--whether extracted information

CREATE TABLE img(
    image_id BIGINT UNIQUE NOT NULL PRIMARY KEY AUTO_INCREMENT,
    jd_utc_start DOUBLE DEFAULT NULL,
    jd_utc_mid DOUBLE DEFAULT NULL,
    jd_utc_end DOUBLE DEFAULT NULL,
    bjd_tdb_start_approximation DOUBLE DEFAULT NULL,
    bjd_tdb_mid_approximation DOUBLE DEFAULT NULL,
    bjd_tdb_end_approximation DOUBLE DEFAULT NULL,
    n_stack INT DEFAULT 1,
    batch INT DEFAULT 1,
    processed BOOLEAN DEFAULT 0, 
    image_type_id INT,
    flat_image_id BIGINT DEFAULT NULL,
    dark_image_id BIGINT DEFAULT NULL,
    mask_image_id BIGINT DEFAULT NULL,
    x_to_template INT DEFAULT NULL,
    y_to_template INT DEFAULT NULL,
    obs_id INT DEFAULT NULL,
    img_path TEXT DEFAULT NULL,
    birth_process_id DECIMAL(25),
    align_process_id DECIMAL(25),
    deleted BOOLEAN DEFAULT 0,
    FOREIGN KEY (birth_process_id) REFERENCES process(process_id),
    FOREIGN KEY (align_process_id) REFERENCES process(process_id),
    FOREIGN KEY (image_type_id) REFERENCES image_type(image_type_id),
    FOREIGN KEY (flat_image_id) REFERENCES img(image_id),
    FOREIGN KEY (dark_image_id) REFERENCES img(image_id),
    FOREIGN KEY (obs_id) REFERENCES observation(obs_id), 
    INDEX(n_stack),
    INDEX(jd_utc_mid)
);

CREATE TABLE img_stacking(
    image_id BIGINT  NOT NULL,
    stacked_id BIGINT NOT NULL,
    FOREIGN KEY (image_id) REFERENCES img(image_id),
    FOREIGN KEY (stacked_id) REFERENCES img(image_id)
);

--deg
--clockwise x is ra+ when 
-- NOT NULL DEFAULT ST_SRID(POINT(0,0),4326),SPATIAL INDEX(fov_pos),

CREATE TABLE sky(
    sky_id BIGINT UNIQUE NOT NULL PRIMARY KEY AUTO_INCREMENT,
    ra DOUBLE DEFAULT NULL,
    `dec` DOUBLE DEFAULT NULL,
    fov_x DOUBLE DEFAULT NULL,
    fov_y DOUBLE DEFAULT NULL,
    scan_angle DOUBLE DEFAULT 0,
    template_image_id BIGINT,
    fov_pos GEOMETRY SRID 4326,
    process_id DECIMAL(25),
    FOREIGN KEY (process_id) REFERENCES process(process_id),
    FOREIGN KEY (template_image_id) REFERENCES img(image_id)
);

CREATE TABLE source_type(
    source_type_id INT UNIQUE NOT NULL PRIMARY KEY AUTO_INCREMENT,
    source_type TEXT
);


CREATE TABLE source(
    source_id BIGINT UNIQUE NOT NULL PRIMARY KEY AUTO_INCREMENT,
    source_type_id INT NOT NULL,
    sky_id BIGINT,
    x_template DOUBLE,
    y_template DOUBLE,
    flux_template DOUBLE,
    e_flux_template FLOAT,
    INDEX(flux_template) USING BTREE,
    FOREIGN KEY (source_type_id) REFERENCES source_type(source_type_id),
    FOREIGN KEY (sky_id) REFERENCES sky(sky_id)
);

CREATE TABLE star_pixel_img(
    star_pixel_img_id BIGINT UNIQUE NOT NULL PRIMARY KEY AUTO_INCREMENT,
    star_id BIGINT,
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
    FOREIGN KEY (birth_process_id) REFERENCES process(process_id),
    FOREIGN KEY (relative_process_id) REFERENCES process(process_id),
    FOREIGN KEY (normalization_process_id) REFERENCES process(process_id),
    FOREIGN KEY (mag_calibration_process_id) REFERENCES process(process_id),
    FOREIGN KEY (timing_process_id) REFERENCES process(process_id),
    INDEX(bjd_tdb_start),
    INDEX(bjd_tdb_mid),
    INDEX(bjd_tdb_end),
    INDEX(flux_raw),
    INDEX(flux_calibrated),
    INDEX(flux_normalized),
    FOREIGN KEY (star_id) REFERENCES star(star_id),
    FOREIGN KEY (image_id) REFERENCES img(image_id)
);

CREATE TABLE reference_star(
    obs_id INT,
    star_id BIGINT,
   FOREIGN KEY (obs_id) REFERENCES observation(obs_id),
    FOREIGN KEY (star_id) REFERENCES star(star_id)
);


INSERT INTO filters (filter_name) values ("l");
INSERT INTO filters (filter_name) values ("r");
INSERT INTO filters (filter_name) values ("g");
INSERT INTO filters (filter_name) values ("b");
INSERT INTO filters (filter_name) values ("Halpha");
INSERT INTO filters (filter_name) values ("OIII");
INSERT INTO filters (filter_name) values ("SII");
INSERT INTO filters (filter_name) values ("Solar");
INSERT INTO filters (filter_name) values ("none");

INSERT INTO image_type (image_type) values ("science_raw");
INSERT INTO image_type (image_type) values ("science_processed");
INSERT INTO image_type (image_type) values ("flat_raw");
INSERT INTO image_type (image_type) values ("flat_debiased");
INSERT INTO image_type (image_type) values ("dark");
INSERT INTO image_type (image_type) values ("bias");
INSERT INTO image_type (image_type) values ("deep_raw");
INSERT INTO image_type (image_type) values ("deep_processed");
INSERT INTO image_type (image_type) values ("mask");

INSERT INTO instrument (instrument_name,filter_id) VALUES ("L350+QHY600m",1);

INSERT INTO obs_site (obs_site_name,obs_site_lon,obs_site_lat,obs_site_height) VALUES ("TDLI_MGO",121.60805556,31.164722222,1);

INSERT INTO observation_type (observation_type_name) VALUES ("science");
INSERT INTO observation_type (observation_type_name) VALUES ("outreach");

INSERT INTO observer_type (observer_type) VALUES ("researcher");
INSERT INTO observer_type (observer_type) VALUES ("visitor");
INSERT INTO observer_type (observer_type) VALUES ("robot");

INSERT INTO observer (observer_name,observer_type_id) VALUES ("robot",3);
INSERT INTO observer (observer_name,observer_type_id) VALUES ("Yicheng Rui",1);

INSERT INTO target_type (target_type) VALUES ("calibration");
INSERT INTO target_type (target_type) VALUES ("star_field");
INSERT INTO target_type (target_type) VALUES ("single_star");
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
INSERT INTO target_n (target_name, target_type_id) VALUES ("HAT-P-20",(SELECT target_type_id FROM target_type where target_type.target_type = 'single_star' LIMIT 1));




SHOW TABLES;
DROP TABLE process;
