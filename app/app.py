import numpy as np
import xgboost as xgb
import xarray as xr
import pandas as pd
import time
import argparse
import os
import base64
import paramiko
import json
import xarray as xr
import io

import waggle.plugin as plugin
from datetime import datetime, timedelta

# 1. import standard logging module
import logging

# 2. enable debug logging
logging.basicConfig(level=logging.DEBUG)

def open_load_model(model_path):
    print(model_path)
    bst = xgb.Booster()
    bst.load_model(model_path)
    return bst

def load_file(file_path):
    with open(file_path, mode='r') as file_buf:
        fname = file_buf.readline()
        system_id = file_buf.readline()
        number_of_gates = file_buf.readline()
        number_of_gates = int(number_of_gates.split()[-1])
        gate_length = file_buf.readline()
        gate_length = float(gate_length.split()[-1])
        gate_length_pts = file_buf.readline()
        gate_length_pts = int(gate_length_pts.split()[-1])
        pulses_per_ray = file_buf.readline()
        pulses_per_ray = int(pulses_per_ray.split()[-1])
        num_rays = file_buf.readline()
        num_rays = int(num_rays.split()[-1])
        scan_type = file_buf.readline()
        focus_range = file_buf.readline()
        start_time = file_buf.readline()

        start_time = datetime.strptime(start_time[:-4],
            "Start time:	%Y%m%d %H:%M:%S")
        resolution = file_buf.readline()
        resolution = float(resolution.split()[-1])
        altitude = file_buf.readline()
        dataline1 = file_buf.readline()
        dataline1 = file_buf.readline()
        dataline2 = file_buf.readline()
        dataline1 = file_buf.readline()
        skip = file_buf.readline()
        times = []
        snrs = []
        betas = []
        velocities = []
        azimuths = []
        elevations = []
        pitches = []
        rolls = []
        while True:
            file_line = file_buf.readline()
            if not file_line:
                break
            split_line = file_line.split()
            times.append(datetime(
                start_time.year, start_time.month, start_time.day, 0, 0, 0) +
                            timedelta(hours=float(split_line[0])))
            azimuths.append(float(split_line[1]))
            elevations.append(float(split_line[2]))
            pitches.append(float(split_line[3]))
            rolls.append(float(split_line[4]))
            string_buf = []
            for i in range(number_of_gates):
                string_buf.append(file_buf.readline())
            string_buf = io.StringIO("\n".join(string_buf))
            input_data = pd.read_csv(string_buf, nrows=number_of_gates,
                    delim_whitespace=True,
                    header=None, names=["gate_no", "velocity", "intensity", "beta"])

            velocities.append(input_data.values[:, 1])
            snrs.append(input_data.values[:, 2] - 1)
            betas.append(input_data.values[:, 3])

    ranges = (np.arange(0, number_of_gates, 1) + 0.5) * gate_length
    azimuths = np.array(azimuths)
    elevations = np.array(elevations)
    pitches = np.array(pitches)
    rolls = np.array(rolls)
    velocities = np.stack(velocities)
    snrs = np.stack(snrs)
    betas = np.stack(betas)
    times = np.array(times)
    times = xr.DataArray(times, dims=['time'])
    ranges = xr.DataArray(ranges, dims=['range'])

    azimuths = xr.DataArray(azimuths, dims=['time'])
    azimuths.attrs["long_name"] = "Azimuth angle"
    azimuths.attrs["units"] = "degree"
    elevations = xr.DataArray(elevations, dims=['time'])
    elevations.attrs["long_name"] = "Elevation angle"
    elevations.attrs["units"] = "degree"
    pitches = xr.DataArray(pitches, dims=['time'])
    pitches.attrs["long_name"] = "Pitch angle"
    pitches.attrs["units"] = "degree"
    rolls = xr.DataArray(rolls, dims=['time'])
    rolls.attrs["long_name"] = "Roll angle"
    rolls.attrs["units"] = "degree"
    velocities = xr.DataArray(velocities, dims=['time', 'range'])
    velocities.attrs["long_name"] = "Mean Doppler velocity"
    velocities.attrs["units"] = "m/s"
    snr = xr.DataArray(snrs, dims=['time', 'range'])
    snr.attrs["long_name"] = "Signal to noise ratio"
    snr.attrs["units"] = "dB"
    betas = xr.DataArray(betas, dims=['time', 'range'])
    betas.attrs["long_name"] = "Extinction coefficient"
    betas.attrs["units"] = "km-1"
    test_file = xr.Dataset({'azimuth': azimuths,
                            'elevation': elevations,
                            'pitch': pitches,
                            'roll': rolls,
                            'doppler_velocity': velocities,
                            'snr': snr,
                            'beta': betas,
                            'time': times,
                            'range': ranges})
    return test_file


def get_scp(ds, model_name, interval=5):
    range_bins = np.arange(60., 11280., 120.)
    # Parse model string for locations of snr, mean_velocity, spectral_width
    locs = 0
    snr_thresholds = []
    scp_ds = {}
    dates = pd.date_range(ds.time.values[0],
                          ds.time.values[-1], freq='%dmin' % interval)
    times = ds.time.values
    snr = ds['snr'].values
    mname = model_name
    while locs > -1:
        locs = mname.find("snr")
        if locs > -1:
            snr_thresholds.append(float(mname[locs+5:locs+13]))
            scp_ds['snrgt%f' % snr_thresholds[-1]] = np.zeros(
                    (len(dates) - 1, len(range_bins) - 1))
            mname = mname[locs+13:]
    
    for i in range(len(dates) - 1):
        time_inds = np.argwhere(np.logical_and(ds.time.values >= dates[i],
                                               ds.time.values < dates[i + 1]))
        if len(time_inds) == 0:
            continue
        for j in range(len(range_bins) - 1):
            range_inds = np.argwhere(np.logical_and(
                ds.range.values >= range_bins[j], 
                ds.range.values < range_bins[j+1]))
            range_inds = range_inds.astype(int)
            if len(range_inds) == 0:
                continue
            snrs = snr[int(time_inds[0]):int(time_inds[-1]), 
                    int(range_inds[0]):int(range_inds[-1])]
            for snr_thresh in snr_thresholds:
                scp_ds['snrgt%f' % snr_thresh][i, j] = len(np.argwhere(snrs > snr_thresh)) / \
                                                       (len(time_inds) * len(range_inds)) * 100
    scp_ds['input_array'] = np.concatenate(
            [scp_ds[var_keys] for var_keys in scp_ds.keys()], axis=1)
    scp_ds['time_bins'] = dates
    print(scp_ds['input_array'].shape)
    return scp_ds


def progress(bytes_so_far: int, total_bytes: int):
    pct_complete = 100. * float(bytes_so_far) / float(total_bytes)
    if int(pct_complete * 10) % 100 == 0:
        print("Total progress = %4.2f" % pct_complete)  


def download_data(args):
    bt = time.time()
    passwd = base64.b64decode("S3VyQGRvMjM=".encode("utf-8"))
    transport = paramiko.Transport('research.adc.arm.gov', 22)
    username = 'rjackson'

    transport.connect(username=username, password=passwd)
    return_list = []
    with paramiko.SFTPClient.from_transport(transport) as sftp:
        sftp.chdir('/data/datastream/sgp/%s' % args.input)
        if args.date is None and args.time is None:
            file_list = sorted(sftp.listdir())[-5:]
        elif args.time is None:
            file_list = sorted(sftp.listdir())
            for f in file_list:
                if not args.date in f:
                    file_list.remove(f)
        else:
            file_list = ['%s.%s.%s.nc' % (args.input, args.date, args.time)]
        last_file = ""
        for f in file_list:
            print(f)
            if os.path.exists('/app/%s' % f):
                continue
            
            # Only process vertically pointing data for now
            if not 'Stare' in f:
                continue
            print("Downloading %s" % f)
            return_list = '/app/%s' % f
            last_file = f
        sftp.get(last_file, localpath='/app/%s' % last_file, callback=progress)

    transport.close()
    print("Download done in %3.2f minutes" % ((time.time() - bt)/60.0))
    return return_list


def worker_main(args, plugin):
    interval = int(args.interval)
    print('opening input %s' % args.input)
    class_names = ['clear', 'cloudy', 'rain']
    file_name = download_data(args)
    
    model = open_load_model(args.model)
    print("Processing %s" % file_name)
    dsd_ds = load_file(file_name)
    scp = get_scp(dsd_ds, args.model)
    out_predict = model.predict(xgb.DMatrix(scp['input_array']))
    for i in range(len(out_predict)):
        print(scp['time_bins'][i].timestamp())
        print(str(
            scp['time_bins'][i]) + ':' + class_names[int(out_predict[i])])
        plugin.publish("weather.classifier.class",
                        out_predict[i],
                        timestamp=scp['time_bins'][i].timestamp())
        
        if out_predict[i] > 0:
            out_ds = dsd_ds.sel(time=slice(
                str(scp['time_bins'][i]), str(scp['time_bins'][i+1])))
            t = pd.to_datetime(out_ds.time.values[0])
            out_ds.to_netcdf('%s.nc' % 
                t.strftime('%Y%m%d.%H%M%S'))
                 
    dsd_ds.close() 
         
 
def main(args):
    if args.verbose:
        print('running in a verbose mode')
    worker_main(args, plugin)


if __name__ == '__main__':
    plugin.init()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--verbose', dest='verbose',
        action='store_true', help='Verbose')
    parser.add_argument(
        '--input', dest='input',
        action='store', default='sgpdlC1.00',
        help='Path to input device or ARM datastream name')
    parser.add_argument(
        '--model', dest='model',
        action='store', default='/app/modelsnrgt3.000000snrgt5.000000.json',
        help='Path to model')
    parser.add_argument(
        '--interval', dest='interval',
        action='store', default=0,
        help='Time interval in seconds')
    parser.add_argument('--date', dest='date', action='store',
                        default=None,
                        help='Date of record to pull in (YYYY-MM-DD)')
    parser.add_argument('--time', dest='time', action='store',
                        default=None, help='Time of record to pull',
                        type=ascii)

    main(parser.parse_args())
