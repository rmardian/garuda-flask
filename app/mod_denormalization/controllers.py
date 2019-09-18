import pymysql

def get_keys_info(SOURCE):

    mysql_conn = pymysql.connect(host=SOURCE['host'],
                                user=SOURCE['username'],
                                password=SOURCE['password'],
                                db=SOURCE['database'],
                                )

    cursor = mysql_conn.cursor()
    query = 'SELECT TABLE_NAME, COLUMN_NAME, CONSTRAINT_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE WHERE TABLE_NAME IN ("language");'

    cursor.execute(query)

    keys = []
    for result in cursor.fetchall():
        #keys.append(result[0])
        print(result)

    mysql_conn.close()

    return keys