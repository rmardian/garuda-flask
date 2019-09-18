from pymongo import MongoClient
import decimal

def generate(schema, data):

    payloads = []

    for row in data:
        new_dict = {}
        for i in range(len(schema)):
            if (type(row[i])==decimal.Decimal):
                new_dict[schema[i]] = float(row[i])
            else:
                new_dict[schema[i]] = row[i]
        payloads.append(new_dict)

    return payloads

def load(CONFIG, collection, payloads):

    client = MongoClient('mongodb://' + CONFIG['host'] + ':' + CONFIG['port'])
    db = client[CONFIG['database']]
    collection = db[collection]

    collection.insert_many(payloads)

'''
def _connect_mongo(host, port, username, password, db):
    """ A util for making a connection to mongo """

    if username and password:
        mongo_uri = 'mongodb://%s:%s@%s:%s/%s' % (username, password, host, port, db)
        conn = MongoClient(mongo_uri)
    else:
        conn = MongoClient(host, port)

    return conn[db]


def read_mongo(db, collection, query={}, host='localhost', port=27017, username=None, password=None, no_id=True):
    """ Read from Mongo and Store into DataFrame """

    # Connect to MongoDB
    db = _connect_mongo(host=host, port=port, username=username, password=password, db=db)

    # Make a query to the specific DB and Collection
    cursor = db[collection].find(query)

    # Expand the cursor and construct the DataFrame
    df =  pd.DataFrame(list(cursor))

    # Delete the _id
    if no_id:
        del df['_id']

    return df
'''