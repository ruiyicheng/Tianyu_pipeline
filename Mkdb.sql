use tianyudev;
CREATE TABLE process(
	process_group_id INT NOT NULL,
    process_id DECIMAL(25) UNIQUE NOT NULL PRIMARY KEY,
    process_cmd TEXT,
    process_status_id INT,
    FOREIGN KEY (process_status_id) REFERENCES process_type(process_status_id),
    FOREIGN KEY (process_site_id) REFERENCES data_process_site(process_site_id)
);
CREATE TABLE process_dependence(
    master_process_id DECIMAL(25),
    dependence_process_id DECIMAL(25),
    FOREIGN KEY (master_process_id) REFERENCES process(process_id),
    FOREIGN KEY (dependence_process_id) REFERENCES process(process_id)
);

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



SHOW TABLES;
DROP TABLE process;
