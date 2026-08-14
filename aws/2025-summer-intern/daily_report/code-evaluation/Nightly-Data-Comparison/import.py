import sys
import json
import math
import os
from multiprocessing.dummy import current_process

import pandas as pd
from pyathena import connect
from typing import Dict, Any
import datetime
import sqlite3

# Set AWS credentials from the Isengard console.
# Do NOT hardcode credentials here. Export them in your shell before running:
#   export AWS_ACCESS_KEY_ID=...
#   export AWS_SECRET_ACCESS_KEY=...
#   export AWS_SESSION_TOKEN=...
# boto3 / pyathena pick these up automatically from the environment.


class DB(object):
    def __init__(self, conn):
        self.con = conn

    run_field_mapping = {
        'efa installer version': 'efa_installer',
        'instance type': 'instance',
        'ami': 'ami',
        'os': 'os',
        'operation': 'operation',
        'branch': 'branch',
        'account id': 'account',
        'aws region': 'aws_region',
        'cma status': 'cma_status',
        'libfabric provider': 'libfabric_provider',
        'nccl version': 'nccl_version',
        'aws ofi nccl plugin version': 'plugin_version',
        'aws ofi nccl build type': 'aws_ofi_nccl_build_type',
        'nccl build type': 'nccl_build_type',
        'aws ofi nccl env vars': 'env_vars',
        'aws ofi nccl use perf env vars': 'use_perf_env_vars',
        'aws ofi nccl commit': 'aws_ofi_nccl_commit',
        'timestamp': 'timestamp',
        'mpi type': 'mpi_type',
        'mpi version': 'mpi_version',
        'test name': 'test_name',
        'number of nodes': 'number_of_nodes',
        'processes per node': 'processes_per_node',
        'cuda version': 'cuda_version',
        'additional nccl test args': 'additional_nccl_test_args',
        'cluster name': 'cluster_name',
        'test metrics tag': 'test_metrics_tag',
        'aws ofi nccl configure command': 'aws_ofi_nccl_configure_command',
        'nccl_tests_split_mask': 'split_mask',
        'cluster size': 'cluster_size',
        'test type': 'test_type',
        'year': 'year',
        'month': 'month',
        'day': 'day'
    }
    perf_field_mapping = {
        'timestamp': 'timestamp',
        'message size (byte)': 'size',
        'time per iteration (us)': 'latency',
        'bus bw (gb/s)': 'busbw',
        'time per iteration - in place (us)': 'latency_inplace',
        'bus bw - in place (gb/s)': 'busbw_inplace',
        'time per iteration - out of place (us)': 'latency_out_of_place',
        'bus bw - out of place (gb/s)': 'busbw_out_of_place'
    }

    def create(self, what='runs,perf,completed'):
        cur = self.con.cursor()
        if 'runs' in what:
            try:
                cur.execute('DROP TABLE runs;')
                print('dropped runs table')
            except Exception as e:
                print(e)
            cur.execute("""CREATE TABLE runs (
                timestamp TEXT PRIMARY KEY NOT NULL,
                number_of_nodes INTEGER NOT NULL,
                operation TEXT NOT NULL,
                split_mask TEXT NOT NULL,
                branch TEXT NOT NULL,
                instance TEXT NOT NULL,
                os TEXT NOT NULL,
                efa_installer TEXT NOT NULL,
                nccl_version TEXT NOT NULL,
                plugin_version TEXT NOT NULL,
                mpi_version TEXT NOT NULL,
                cuda_version TEXT,
                env_vars TEXT NOT NULL,
                use_perf_env_vars INTEGER NOT NULL,
                libfabric_provider TEXT NOT NULL,
                additional_nccl_test_args TEXT NOT NULL,
                aws_ofi_nccl_configure_command TEXT NOT NULL,
                ami TEXT NOT NULL,
                account TEXT NOT NULL,
                processes_per_node INTEGER NOT NULL,
                aws_region TEXT,
                cma_status INTEGER,

                aws_ofi_nccl_build_type TEXT,
                nccl_build_type TEXT,
                test_type TEXT,
                test_name TEXT,
                aws_ofi_nccl_commit TEXT  NOT NULL,
                mpi_type TEXT,
                cluster_name TEXT,
                cluster_size INTEGER,
                test_metrics_tag TEXT,
                year TEXT,
                month TEXT,
                day TEXT
            );
            """)
            print('created runs table')
        if 'perf' in what:
            try:
                cur.execute('DROP TABLE perf;')
                print('dropped perf table')
            except Exception as e:
                print(e)
            cur.execute("""CREATE TABLE perf (
                timestamp TEXT NOT NULL,
                size INTEGER,
                latency REAL,
                busbw REA NOT NULL,
                latency_inplace REAL,
                busbw_inplace REAL,
                latency_out_of_place REAL,
                busbw_out_of_place REAL
            );
            """)
            print('created perf table')

            try:
                cur.execute('DROP INDEX perf_timestamp')
                print('dropped perf index')
            except Exception as e:
                print(e)
            cur.execute("""CREATE INDEX perf_timestamp ON perf(timestamp);""")
            print('created perf index')
            self.con.commit()
        if 'completed' in what:
            try:
                cur.execute('DROP TABLE completed;')
                print('dropped completed table')
            except Exception as e:
                print(e)
            cur.execute("""CREATE TABLE completed (
                table_name TEXT PRIMARY KEY NOT NULL,
                timestamp TEXT NOT NULL
            );
            """)
            print('created completed table')

    def add_run(self, data: Dict[str, Any]) -> bool:
        cur = self.con.cursor()

        # Check for existing timestamp
        cur.execute("SELECT COUNT(*) FROM runs WHERE timestamp = ?",
                    (data['timestamp'],))
        if cur.fetchone()[0] > 0:
            return False

        # Prepare data for insertion
        mapped_data = {}
        for orig_key, db_key in DB.run_field_mapping.items():
            value = data.get(orig_key)

            # Handle special cases
            if db_key in ['cma_status', 'use_perf_env_vars']:
                mapped_data[db_key] = 1 if value else 0
            elif db_key == 'env_vars' and value:
                mapped_data[db_key] = json.dumps(value)
            else:
                mapped_data[db_key] = value

        # Create the SQL insert statement
        columns = ', '.join(mapped_data.keys())
        placeholders = ', '.join(['?' for _ in mapped_data])
        sql = f"INSERT INTO runs ({columns}) VALUES ({placeholders})"

        # Execute the insert
        cur.execute(sql, list(mapped_data.values()))

        # delete old perf data
        cur.execute('DELETE FROM perf WHERE timestamp=?', [data['timestamp']])
        return True

    def add_perf(self, data: Dict[str, Any]) -> bool:
        cur = self.con.cursor()

        # Prepare data for insertion
        mapped_data = {}
        for orig_key, db_key in DB.perf_field_mapping.items():
            value = data.get(orig_key)
            mapped_data[db_key] = value
        # Create the SQL insert statement
        columns = ', '.join(mapped_data.keys())
        placeholders = ', '.join(['?' for _ in mapped_data])
        sql = f"INSERT INTO perf ({columns}) VALUES ({placeholders})"

        # Execute the insert
        cur.execute(sql, list(mapped_data.values()))
        return True

    def update_completed(self, table, timestamp):
        print(f'updated completed on {table} to {timestamp}')
        cur = self.con.cursor()
        cur.execute(
            f"INSERT INTO completed(table_name, timestamp) VALUES('{table}', '{timestamp}') ON CONFLICT(table_name) DO UPDATE SET timestamp='{timestamp}';")
        return True

    def get_completed(self, table):
        cur = self.con.cursor()
        df = pd.read_sql('SELECT * FROM completed;', self.con)
        df = df.set_index('table_name')
        ret = df.to_dict('index')
        return ret

    def commit(self):
        self.con.commit()

if __name__ == '__main__':
    # pya_conn = connect(s3_staging_dir='s3://athena-query-results-471112928337-us-west-2', region_name='us-west-2',
    #                    schema_name='subspace-performance-database-partitioned-data')
    # sql_con = sqlite3.connect("data.db")
    # db = DB(sql_con)
    # db.create(what='')

    # Modify the connect call to include the AWS credentials
    pya_conn = connect(
        s3_staging_dir='s3://athena-query-results-471112928337-us-west-2',
        region_name='us-west-2',
        schema_name='subspace-performance-database-partitioned-data',
        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
        aws_session_token=os.environ['AWS_SESSION_TOKEN']
    )

    # Rest of your code remains the same
    sql_con = sqlite3.connect("data.db")
    db = DB(sql_con)
    db.create()


    def save_table(table, operation, branch):
        c = pya_conn.cursor()
        completed = db.get_completed(table)
        if table in completed: current_progress = completed[table]['timestamp']
        else: current_progress = None
        print(f'current progress for {table} is {current_progress}')
        q = c.execute(f'SELECT * FROM {table} ORDER BY timestamp DESC;')
        n = 0
        first = None
        for row in c:
            row = dict(zip([z[0] for z in c.description], row))
            if not first:
                first = row['timestamp']
                print(f"first for {table} is {row['timestamp']}")
            if row['timestamp'] == current_progress:
                print(f'caught up imported {n}')
                break
            row['operation'] = operation
            row['branch'] = branch
            result = db.add_run(row)
            result = db.add_perf(row)
            db.commit()
            n += 1
            if n % 10000 == 0:
                print(n, row['timestamp'])
            if '2025-07-01' in row['timestamp']: break
            if '2024' in row['timestamp']: break
        result = db.update_completed(table, first)
        db.commit()

    # save_table('nccl_allreduce_aws_ofi_nccl_master', 'All Reduce', 'master')
    # save_table('nccl_allgather_aws_ofi_nccl_master', 'All Gather', 'master')
    # save_table('nccl_reduce_scatter_aws_ofi_nccl_master', 'Reduce Scatter', 'master')
    # save_table('nccl_alltoall_aws_ofi_nccl_master', 'All to All', 'master')
    save_table('nccl_allreduce_aws_ofi_nccl_release_branch', 'All Reduce', 'release_branch')
    save_table('nccl_allgather_aws_ofi_nccl_release_branch', 'All Gather', 'release_branch')
    save_table('nccl_reduce_scatter_aws_ofi_nccl_release_branch', 'Reduce Scatter', 'release_branch')
    # save_table('nccl_alltoall_aws_ofi_nccl_release_branch', 'All to All', 'release_branch')