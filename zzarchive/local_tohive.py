def local_tohive(folder, tablename):
    try:
        hiveql = "/tmp/hivequery.hql"
        csvtable = tablename + "_csv"
        foldr = folder if folder[-1] == "/" else folder + "/"
        with open(hiveql, mode='w') as f: 
            f.write("DROP TABLE IF EXISTS {};\n".format(csvtable))
            f.write("CREATE TABLE {} LIKE {} ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde' WITH SERDEPROPERTIES ('separatorChar' = ',') STORED AS TEXTFILE TBLPROPERTIES('skip.header.line.count' = '1');\n".format(csvtable, tablename))
            f.write("LOAD DATA LOCAL INPATH '{}*.csv' OVERWRITE INTO TABLE {};\n".format(foldr, csvtable))
            f.write("INSERT OVERWRITE TABLE {} SELECT * FROM {};\n".format(tablename, csvtable))
            f.write("DROP TABLE {};\n".format(csvtable))
        return execute('hive -f ' + hiveql, capture_stderr = True)
    except Exception as e:
        print(e)