"""
Queries VMware Data Warehouse for pertinent sales data to regress against macroeconomic factors and uploads
to data warehouse as its own table.

Primary export:
    sse_ccmi.dc_model_vmware_sales_2010_onward: Table in database containing VMware AMER sales grouped by day.

    Original author: Daniel Cohen, Summer 2019 Competitive Intelligence Analyst Intern
"""


import time  # standard libraries

from sqlalchemy.exc import DBAPIError, SQLAlchemyError  # third-party libraries

from Utils import change_dir, get_credentials, get_sqlalchemy_engine  # package libraries


LOCAL = True


def create(q, tbl_name):
    """
    Runs query and creates table tbl_name. Subject to fewer security restrictions because merely downloading
    query String from source.
    :param q: String to be executed; must be complete
    :param tbl_name: name of table to be uploaded into data warehouse
    """

    try:
        engine = get_sqlalchemy_engine()
        print("Connected to PostgreSQL database.")

        print("Attempting to create table", tbl_name)
        start_time = time.time()
        conn = engine.connect()
        trans = conn.begin()

        drop_tbl_query = "DROP TABLE IF EXISTS {};".format(tbl_name)
        create_tbl_query = "CREATE TABLE {} AS {}".format(tbl_name, q)
        conn.execute(drop_tbl_query)
        conn.execute(create_tbl_query)

        trans.commit()
        conn.close()
        end_time = time.time()
        print("Successfully created table", tbl_name, "in %s seconds." % (end_time - start_time))

        # cursor.close()
        # connection.close()
        print("PostgreSQL connection is closed.")
    except SQLAlchemyError or DBAPIError as e:
        print("Error while connecting to database", e)


def get_query_from_file(file_extension):
    """
    Gets query from file_extension in ~/OneDrive - VMware, Inc/Economic Modelling/SQL Queries;
    Removes all tabs and spaces because they fuck everything up.
    :param file_extension: File extension of query
    :return: String contained in specified file
    """

    try:
        change_dir("SQL Queries")
    except OSError:
        print("Cannot change directory; please verify directory.")
        raise SystemExit
    try:
        with open(file_extension, "r") as file:
            file_string = file.read()
            file_string = file_string.replace('\n', ' ').replace('\t', ' ')
        return file_string
    except OSError:
        print("File does not exist or cannot be read.")
        raise SystemExit
