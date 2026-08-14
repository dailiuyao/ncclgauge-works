import sys
import json
import math
from nbclient.client import timestamp
from pandas.core.interchange.dataframe_protocol import DataFrame
from pyathena import connect
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pyathena import connect
import pandas as pd
import datetime
import sqlite3
import warnings
import numpy as np
import model
import os
import glob
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import math
import json
from dataclasses import dataclass
from typing import Optional, Callable, Any, List, Dict, Tuple

class DBRunsInfo(object):
    def __init__(self, timestamps, runs, operation, branch):
        self.operation = operation
        self.branch = branch
        self.runs = runs
        self.timestamps = timestamps

class RunInfo(object):
    def __init__(self, operation, mask, timestamp=None, instance=None, os=None, plugin_version=None, nccl_version=None, efa_installer=None, aws_ofi_nccl_commit=None, nodes=None):
        self.instance = instance
        self.operation = operation
        self.mask = mask
        self.os = os
        self.nodes = nodes
        self.plugin_version = plugin_version
        self.nccl_version = nccl_version
        self.efa_installer = efa_installer
        self.timestamp = timestamp
        self.aws_ofi_nccl_commit = aws_ofi_nccl_commit

class FileRunsInfo(RunInfo):
    def __init__(self, operation, mask, files, instance=None, os=None, plugin_version=None, nccl_version=None, efa_installer=None, timestamp=None):
        RunInfo.__init__(self, operation, mask, instance=instance, os=os, nccl_version=nccl_version, efa_installer=efa_installer, timestamp=timestamp)
        self.files = files



@dataclass
class PlotOptions:
    """Configuration options for plotting NCCL performance data"""
    bw: bool = True  # True for bandwidth, False for latency
    title: str = None
    annotation: str = ''
    instance_type: Optional[str] = None
    time_dim: Optional[str] = None
    operation: str = 'All Reduce'
    branch: str = 'release_branch'
    upper: Optional[float] = 0.9
    lower: Optional[float] = 0.1
    bw_value_field = 'busbw_out_of_place'
    bw_units = 'gb/sec'
    latency_value_field = 'latency_out_of_place'
    latency_units = 'usec'
    filter_func: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None

@dataclass
class TimeseriesOptions:
    """Configuration options for plotting NCCL performance data"""
    bw: bool = True  # True for bandwidth, False for latency
    title: str = None
    annotation: str = ''
    instance_type: Optional[str] = None
    time_dim: Optional[str] = None
    message_sizes: List[int] = (8192,)
    operation: str = 'All Reduce'
    branch: str = 'release_branch'
    bw_value_field = 'busbw_out_of_place'
    bw_units = 'gb/sec'
    latency_value_field = 'latency_out_of_place'
    latency_units = 'usec'
    filter_func: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None
    env_vars: Optional[dict] = None
    meta_data: Optional[dict] = None

class DataSizeFilter:
    """Predefined filters for data size ranges"""

    @staticmethod
    def last_n(df: pd.DataFrame) -> pd.DataFrame:
        """Filter for large message sizes (≥ 8MB)"""
        return df[df['size'] >= 8388608]

    @staticmethod
    def first_n(df: pd.DataFrame) -> pd.DataFrame:
        """Filter for small message sizes (256B - 16MB)"""
        return df[df['size'].between(256, 16777216)]

    @staticmethod
    def all(df: pd.DataFrame) -> pd.DataFrame:
        """Filter for all message sizes (256B - 8GB)"""
        return df[df['size'] >= 256]


class NCCLPlotter:
    """Handles plotting of NCCL performance data"""

    def __init__(self, db_connection):
        self.fig = None
        self.db_connection = db_connection
        self.n_plots = 0

    def convert_size(self, size_bytes: int) -> str:
        """Convert bytes to human readable format"""
        if size_bytes == 0:
            return "0B"
        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = int(round(size_bytes / p, 2))
        return f"{s} {size_name[i]}"

    def make_plot(self, options: PlotOptions):
        """Initialize the plotting figure"""
        self.fig = go.Figure()
        self.n_plots = 0
        # Get title from options
        plot_title = getattr(options, 'title', 'Performance Analysis')  # Default title if not specified

        y_title = "Bus BW (GiB/sec)" if options.bw else "Time (usec)"
        self.fig.update_layout(
            title=dict(
                text=plot_title,  # Main title text
                x=0.5,  # Center the title
                y=1,  # Position from top
                xanchor='center',  # Anchor point for x position
                yanchor='top',  # Anchor point for y position
                font=dict(
                    size=20,  # Font size
                    color='black',  # Font color
                    family='Arial'  # Font family
                )
            ),
            yaxis_title=y_title,
            xaxis_title="Data Size (bytes)",
            width=900,
            height=600,
            legend=dict(x=0, y=1.02, xanchor='left', yanchor='bottom', grouptitlefont=dict(weight='bold'))
        )
        self.fig.update_xaxes(minor=dict(ticklen=6, tickcolor="black", tickmode='auto', nticks=10, showgrid=True))
        self.fig.update_yaxes(minor_ticks="inside")

    def make_timeseries_plot(self, options: TimeseriesOptions):
        """Initialize the plotting figure"""
        self.fig = go.Figure()

        plot_title = getattr(options, 'title', 'Timeseries')  # Default title if not specified

        y_title = "Bus BW (GiB/sec)" if options.bw else "Time (usec)"
        self.fig.update_layout(
            title = dict(
                text=plot_title,  # Main title text
                x=0.5,  # Center the title
                y=1,  # Position from top
                xanchor='center',  # Anchor point for x position
                yanchor='top',  # Anchor point for y position
                font=dict(
                    size=20,  # Font size
                    color='black',  # Font color
                    family='Arial'  # Font family
                )
            ),
            yaxis_title = y_title,
            xaxis_title = "Data Size (bytes)",
            width = 1000,
            height = 800,
            legend=dict(orientation='h', x=0, y=1.02, xanchor='left', yanchor='bottom', grouptitlefont=dict(weight='bold'))
        )
        self.fig.update_yaxes(type="log")


    def get_plot_data(self, df: pd.DataFrame, options: PlotOptions) -> tuple[pd.DataFrame, str]:
        """Process test data and generate plot name"""
        if options.bw:
            value_field = options.bw_value_field
            units = options.bw_units
        else:
            value_field = 'latency_out_of_place'
            value_field = options.latency_value_field
            units = options.latency_units

        if value_field + '_mean' in df.columns:
            value_field = value_field+'_mean'

        if len(df) == 0:
            raise Exception("No data")

        first = df.iloc[0]
        env = first.get("env_vars")
        if env:
            env = env.replace('\\', '').strip('"')
            env_dict = json.loads(env)
            for k, v in env_dict.items():
                if '/' in v:
                    v = v.split('/')[-1]
                    env_dict[k] = f'.../{v}'
            env = json.dumps(env_dict, indent=2).replace('\n', '<br>')

        #df = df[df['size'].between(1024, 4294967296)]
        df['message size'] = df['size'].apply(self.convert_size)

        size_of_interest = ['64 KB', '128 KB', '256 KB', '512 KB', '1 MB', '2 MB', '16 MB', '64 MB', '256 MB',
                            '1 GB', '4 GB']
        # name = f'Type: {first["instance"]} Timestamp: {first["timestamp"]} NCCL: {first["nccl_version"]} Plugin: {first["plugin_version"]} EFA Installer: {first["efa_installer"]} OS: {first["os"]} Git: {first["aws_ofi_nccl_commit"]} <br>ENV: {env}<br>'
        name = f'Type: {first["instance"]} Timestamp: {first["timestamp"]} NCCL: {first["nccl_version"]} Plugin: {first["plugin_version"]}<br>EFA Installer: {first["efa_installer"]} OS: {first["os"]} Git: {first["aws_ofi_nccl_commit"]}<br>ENV: {env}<br>'
        for i in size_of_interest:
            x = df.loc[df['message size'] == i]
            if x.empty or x[value_field].values[0] is None:
                continue
            s = int(round(x[value_field].values[0]))
            name += f'{i}: {s} {units}; '

        return df, name

    def add_plot(self, df: pd.DataFrame, name: str, key: str, options: PlotOptions):
        """Add a trace to the plot"""
        extended = False
        if options.bw:
            value_field = options.bw_value_field
            units = options.bw_units
        else:
            value_field = 'latency_out_of_place'
            value_field = options.latency_value_field
            units = options.latency_units

        if value_field + '_mean' in df.columns:
            value_field_upper = value_field +'_percentile_upper'
            value_field_lower = value_field + '_percentile_lower'
            value_field = value_field + '_mean'
            extended = True

        self.n_plots += 1
        colors = ['rgb(99, 110, 250)', 'rgb(239, 85, 59)', 'rgb(0, 204, 150)', 'rgb(171, 99, 250)', 'rgb(255, 161, 90)', 'rgb(25, 211, 243)', 'rgb(255, 102, 146)', 'rgb(182, 232, 128)', 'rgb(255, 151, 255)', 'rgb(254, 203, 82)']
        color = colors[self.n_plots % len(colors)]
        area_color = color.replace(')', ',0.3)').replace('rgb','rgba')

        if extended:
            trace = go.Scatter(
                name=name+' upper',
                x=df['message size'],
                y=df[value_field_upper],
                mode='lines',
                fillcolor=area_color,
                line=dict(color=area_color, width=0.5),
                showlegend=False
            )
            self.fig.add_trace(trace)
            trace = go.Scatter(
                name=name+' lower',
                x=df['message size'],
                y=df[value_field_lower],
                line=dict(color=area_color, width=0.5),
                mode='lines',
                fillcolor=area_color,
                fill='tonexty',
                showlegend=False
            )
            self.fig.add_trace(trace)
        trace = go.Scatter(
            x=df['message size'],
            y=df[value_field],
            mode='lines',
            name=name,
            #fillcolor=color,
            line=dict(color=color, width=2),
            legendgroup=options.annotation,
            legendgrouptitle_text=options.annotation
        )
        self.fig.add_trace(trace)

    def get_timeseries_data(self, df: pd.DataFrame, options: TimeseriesOptions) -> tuple[pd.DataFrame, str]:
        if options.bw:
            value_field = options.bw_value_field
            units = options.bw_units
        else:
            value_field = 'latency_out_of_place'
            value_field = options.latency_value_field
            units = options.latency_units

        if len(df) == 0:
            raise Exception("No data")

        first = df.iloc[0]
        env = first.get("env_vars")
        if env:
            env = env.replace('\\', '').strip('"')
            env_dict = json.loads(env)
            for k, v in env_dict.items():
                if '/' in v:
                    v = v.split('/')[-1]
                    env_dict[k] = f'.../{v}'
            env = json.dumps(env_dict, indent=2).replace('\n', '<br>')


        df['message size'] = df['size']
        df['datetime'] = pd.to_datetime(df['timestamp'])
        result = []
        for m in options.message_sizes:
            result.append((self.convert_size(m), df[df['message size'] == m]))
        return result


    def add_timeseries_from_db(self, timestamps: List[str], options: TimeseriesOptions) -> pd.DataFrame:
        if options.bw:
            value_field = options.bw_value_field
            units = options.bw_units
        else:
            value_field = 'latency_out_of_place'
            value_field = options.latency_value_field
            units = options.latency_units

        colors = ['rgb(99, 110, 250)', 'rgb(239, 85, 59)', 'rgb(0, 204, 150)', 'rgb(171, 99, 250)', 'rgb(255, 161, 90)',
                  'rgb(25, 211, 243)', 'rgb(255, 102, 146)', 'rgb(182, 232, 128)', 'rgb(255, 151, 255)',
                  'rgb(254, 203, 82)']

        markers = ['circle-dot', 'square-dot', 'diamond-dot', 'triangle-up', 'triangle-down']

        dfs = []
        for ts in timestamps:
            data = self._get_db_data(ts, options)
            dfs.append(data)
        df = pd.concat(dfs)

        df['versions'] = df['plugin_version'] + '/' + df['efa_installer'] + '/' + df['nccl_version']
        versions = df['versions'].unique()
        color_map = {versions[i]: colors[i % len(colors)] for i in range(0, len(versions))}
        df['version-color'] = df['versions'].replace(color_map)

        oss = df['os'].unique()
        os_map = {oss[i]: markers[i % len(markers)] for i in range(0, len(oss))}
        df['os-symbol'] = df['os'].replace(os_map)

        results = self.get_timeseries_data(df, options)
        versions = dfs

        first = True
        for result in results:
            name, data = result
            if first:
                x0 = data.iloc[0]['datetime']
                y0 = data.iloc[0][value_field]
                first = False
            trace = go.Scatter(x=data['datetime'], name=name, mode='lines+markers', y=data[value_field],
                               marker_line_color=data['version-color'], marker_color=data['version-color'], marker_symbol=data['os-symbol'],
                               marker_line_width=0, marker_size=5, line=dict(width=1))
            self.fig.add_trace(trace)


        for k,v in color_map.items():
            self.fig.add_shape(showlegend=True, fillcolor=v, name=k, type="rect", line_width=0, x0=x0,y0=y0,x1=x0,y1=y0)
        for k, v in os_map.items():
            self.fig.add_trace(go.Scatter(x=[x0], name=k, mode='markers', y=[y0],
                               marker_line_color='black', marker_color='black', marker_symbol=v,
                               marker_line_width=0, marker_size=5, line=dict(width=1)))

    def _get_db_data(self, timestamp: str, options: PlotOptions) -> pd.DataFrame:
        """Fetch data from the database"""
        query = f"SELECT * FROM runs INNER JOIN perf ON runs.timestamp = perf.timestamp WHERE perf.timestamp='{timestamp}' ORDER BY size ASC"
        #print(query)
        df = pd.read_sql(query, self.db_connection, params={'timestamp': timestamp})
        df = df.loc[:, ~df.columns.duplicated()]
        if options.filter_func:
            df = options.filter_func(df)
        return df

    def _parse_nccl_string(self, data_string: str, options: PlotOptions, run_info: RunInfo) -> pd.DataFrame:
        """Parse NCCL output string into a DataFrame"""
        headers = ['size', 'count', 'type', 'redop', 'root',
                   'latency_out_of_place', 'algbw_out_of_place',
                   'busbw_out_of_place', 'wrong_out_of_place',
                   'latency_inplace', 'algbw_inplace',
                   'busbw_inplace', 'wrong_inplace']

        rows = []
        for line in data_string.split('\n'):
            line = line.strip()
            if not line or not line[0].isdigit():
                continue
            rows.append([self._convert_value(v) for v in line.split() if v])

        df = pd.DataFrame(rows, columns=headers)

        if options.filter_func:
            df = options.filter_func(df)
        df['busbw'] = df['busbw_out_of_place']
        df['latency'] = df['latency_out_of_place']
        df['instance'] = run_info.instance
        df['os'] = run_info.os
        df['timestamp'] = run_info.timestamp
        df['nccl_version'] = run_info.nccl_version
        df['efa_installer'] = run_info.efa_installer
        df['plugin_version'] = run_info.plugin_version
        df['aws_ofi_nccl_commit'] = run_info.aws_ofi_nccl_commit
        return df

    def _convert_value(self, value: str) -> Any:
        """Convert string values to appropriate types"""
        try:
            return int(value) if '.' not in value else float(value)
        except ValueError:
            return value

    def add_plot_from_db(self, timestamp: str, options: PlotOptions) -> pd.DataFrame:
        """Add plot from database data"""
        df = self._get_db_data(timestamp, options)
        df, name = self.get_plot_data(df, options)
        self.add_plot(df, name, timestamp, options)
        return df

    def add_plots_from_db(self, runs: DBRunsInfo, options: PlotOptions) -> pd.DataFrame:
        """Add plots from database data"""

        def percentile_upper(x):
            return x.quantile(options.upper)

        def percentile_lower(x):
            return x.quantile(options.lower)

        dfs = []
        for ts in runs.timestamps:
            dfs.append(self._get_db_data(ts, options))
        df = dfs[0]
        dfs = pd.concat(dfs)
        df_agg = dfs.groupby('size').agg({
            'busbw':[percentile_upper, percentile_lower, 'mean', 'max', 'min'],
            'busbw_out_of_place':[percentile_upper, percentile_lower, 'mean', 'max', 'min'],
            'busbw_inplace': [percentile_upper, percentile_lower, 'mean', 'max', 'min'],
            'latency_out_of_place': [percentile_upper, percentile_lower, 'mean', 'max', 'min'],
            'latency_inplace': [percentile_upper, percentile_lower, 'mean', 'max', 'min'],
        }).pipe(lambda x: x.set_axis(x.columns.map('_'.join), axis=1))
        df_agg['size'] = df_agg.index
        df_agg = df_agg.reset_index(drop=True)
        df = df.merge(df_agg, on='size')
        df, name = self.get_plot_data(df, options)
        self.add_plot(df, name, runs.timestamps[0], options)
        return df

    def add_plot_from_directory(self, dirpath: str, options: PlotOptions) -> pd.DataFrame:
        """Add plot from the first slurmout_*.txt file in the given directory"""
        # Find all slurmout_*.txt files in the directory
        slurm_files = glob.glob(os.path.join(dirpath, 'slurmout_*.txt'))

        # If no slurm files found, raise an error
        if not slurm_files:
            raise FileNotFoundError(f"No slurmout_*.txt files found in {dirpath}")

        # Sort the files (optional, ensures consistent behavior if multiple files exist)
        slurm_files.sort()

        # Get the first slurm file
        filepath = slurm_files[0]

        # Read the contents of the file
        with open(filepath, 'r') as f:
            contents = f.read()

        # Use the existing method to add plot from string
        return self.add_plot_from_string(contents, options)

    def add_plot_from_file(self, filepath: str, options: PlotOptions, run_info: RunInfo) -> pd.DataFrame:
        """Add plot from file data"""
        with open(filepath, 'r') as f:
            contents = f.read()
        return self.add_plot_from_string(contents, options, run_info)

    def add_plots_from_files(self, runs: FileRunsInfo, options: PlotOptions) -> pd.DataFrame:
        """Add plot from file data"""
        contents = []
        for path in runs.files:
            with open(path, 'r') as f:
                contents.append(f.read())
        return self.add_plots_from_strings(contents, options, runs)

    def add_plots_from_strings(self, data_strings: List[str], options: PlotOptions, info: RunInfo) -> pd.DataFrame:
        """Add plot from string data"""

        def percentile_upper(x):
            return x.quantile(options.upper)

        def percentile_lower(x):
            return x.quantile(options.lower)

        dfs = []
        for data_string in data_strings:
            dfs.append(self._parse_nccl_string(data_string, options, info))
        df = dfs[0]
        dfs = pd.concat(dfs)
        df_agg = dfs.groupby('size').agg({
            'busbw': [percentile_upper, percentile_lower, 'mean', 'max', 'min'],
            'busbw_out_of_place': [percentile_upper, percentile_lower, 'mean', 'max', 'min'],
            'busbw_inplace': [percentile_upper, percentile_lower, 'mean', 'max', 'min'],
            'latency_out_of_place': [percentile_upper, percentile_lower, 'mean', 'max', 'min'],
            'latency_inplace': [percentile_upper, percentile_lower, 'mean', 'max', 'min'],
        }).pipe(lambda x: x.set_axis(x.columns.map('_'.join), axis=1))
        df_agg['size'] = df_agg.index
        df_agg = df_agg.reset_index(drop=True)
        df = df.merge(df_agg, on='size')
        df, name = self.get_plot_data(df, options)
        self.add_plot(df, name, info.timestamp, options)
        return df


    def add_plot_from_string(self, data_string: str, options: PlotOptions, info: RunInfo) -> pd.DataFrame:
        """Add plot from string data"""
        df = self._parse_nccl_string(data_string, options, info)
        df, name = self.get_plot_data(df, options)
        self.add_plot(df, name, name, options)
        return df

    def add_plot_from_model(self, options: PlotOptions, run_info: RunInfo):
        models = {
            'p5en.48xlarge': model.LogGP.p5en
        }
        gpus_per_node = {
            '0x0': 8,
            '0x7': 1,
        }
        def latency(row):
            return models[run_info.instance].results(run_info.operation, row['size'], run_info.nodes, gpus_per_node[run_info.mask]).latency
        def busbw(row):
            return models[run_info.instance].results(run_info.operation, row['size'], run_info.nodes, gpus_per_node[run_info.mask]).busbw
        minimum = 0
        if run_info.operation in ('Reduce Scatter', 'All Gather'): minimum = 11
        sizes = [2**x for x in range(minimum,34)]
        df = pd.DataFrame(sizes, columns=['size'])
        df['latency'] = df.apply(latency, axis=1)
        df['busbw'] = df.apply(busbw, axis=1)
        if options.filter_func:
            df = options.filter_func(df)
        return self.add_plot_from_df(df, options, run_info)

    def add_plot_from_df(self, df: DataFrame, options: PlotOptions, run_info: RunInfo) -> pd.DataFrame:
        """Add plot from data frame"""
        """Parse NCCL output string into a DataFrame"""
        if options.filter_func:
            df = options.filter_func(df)

        #df['busbw'] = df['busbw_out_of_place']
        #df['latency'] = df['latency_out_of_place']
        if not 'busbw_out_of_place' in df.columns:
            df['busbw_out_of_place'] = df['busbw']
        if not 'latency_out_of_place' in df.columns:
            df['latency_out_of_place'] = df['latency']
        df['instance'] = run_info.instance
        df['os'] = run_info.os
        df['timestamp'] = run_info.timestamp
        df['nccl_version'] = run_info.nccl_version
        df['efa_installer'] = run_info.efa_installer
        df['plugin_version'] = run_info.plugin_version
        df['aws_ofi_nccl_commit'] = run_info.aws_ofi_nccl_commit

        df, name = self.get_plot_data(df, options)
        self.add_plot(df, name, name, options)
        return df

    def save_plot(self, filename: str):
        """Save the current plot to a file"""
        self.fig.write_image(filename)

    def show_plot(self):
        """Display the current plot"""
        display(self.fig)

    def search_runs(self,
            branch,
            operation,
            instance_type=None,
            os=None,
            num_nodes=None,
            nccl_version=None,
            plugin_version=None,
            efa_version=None,
            cuda_version=None,
            mask=None,
            optimized=None,
            start_time=None,
            end_time=None
    ):
        # Start building the query - select all columns
        query = f'SELECT * FROM runs'
        conditions = [
            f'"operation"=\'{operation}\'',
            f'"branch"=\'{branch}\''
        ]

        # Add conditions based on provided parameters
        if instance_type:
            conditions.append(f'"instance"=\'{instance_type}\'')
        if os:
            conditions.append(f'"os"=\'{os}\'')
        if num_nodes:
            conditions.append(f'"number_of_nodes"={num_nodes}')
        if nccl_version:
            conditions.append(f'"nccl_version" LIKE \'{nccl_version}\'')
        if plugin_version:
            conditions.append(f'"plugin_version"=\'{plugin_version}\'')
        if efa_version:
            conditions.append(f'"efa_installer"=\'{efa_version}\'')
        if cuda_version:
            conditions.append(f'"cuda_version"=\'{cuda_version}\'')
        if mask:
            conditions.append(f'"env_vars" LIKE \'%{mask}%\'')
        if optimized is not None:
            if optimized:
                conditions.append('"env_vars" LIKE \'%TUNER%\'')
            else:
                conditions.append('"env_vars" NOT LIKE \'%TUNER%\'')

        if start_time:
            conditions.append(f'timestamp >= \'{start_time}\'')

        if end_time:
            conditions.append(f'timestamp <= \'{end_time}\'')
        # Combine conditions if they exist
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        # Add ordering and limit to 1 to get only the latest
        query += " ORDER BY datetime(timestamp) DESC LIMIT 100"
        # Execute query
        df = pd.read_sql(query, self.db_connection)

        # print(f'{query} returned results {len(df)}')

        if df.empty:
            return None

        # Get the first row
        timestamps = list(df['timestamp'])
        results_df = pd.DataFrame([{
            'mask': '0x7' if '0x7' in row['env_vars'] else '0x0',
            'optimized': 'TUNER' in row['env_vars']
        } for _, row in df.iterrows()])

        return DBRunsInfo(timestamps, results_df, operation, branch)




