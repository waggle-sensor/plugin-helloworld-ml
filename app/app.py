import numpy as np
import xgboost as xgb
import xarray as xr
import pandas as pd
import highiq
import time
import argparse
import os
import base64
import paramiko
import json
import xarray as xr
import io

from waggle.plugin import Plugin
from datetime import datetime, timedelta

# 1. import standard logging module
import logging

# 2. enable debug logging
#logging.basicConfig(level=logging.DEBUG)

def open_load_model(model_path):
    print(model_path)
    bst = xgb.Booster()
    bst.load_model(model_path)
    return bst

def load_file(file_path):
    test_file = highiq.io.read_00_data(file_path, 'sgpdlprofcalC1.home_point')
    test_file.to_netcdf('test.nc')
    del test_file
    test_file = xr.open_dataset('test.nc')
    return test_file

def process_file(ds):
    print("Processing lidar moments...")
    ti = time.time()
    my_list = []
    for x in ds.groupby_bins('time', ds.time.values[::5]):
        d = x[1]
        d['acf_bkg'] = d['acf_bkg'].isel(time=1)
        psd = highiq.calc.get_psd(d)
        my_list.append(highiq.calc.get_lidar_moments(psd))
        del psd
    ds_out = xr.concat(my_list, dim='time')

    print("Done in %3.2f minutes" % ((time.time() - ti) / 60.))
    return ds_out


def get_scp(ds, model_name, config, interval=5):
    range_bins = np.arange(60., 11280., 120.)
    # Parse model string for locations of snr, mean_velocity, spectral_width
    locs = 0
    snr_thresholds = []
    scp_ds = {}
    if config == "Stare":
        interval = 5
        dates = pd.date_range(ds.time.values[0],
                          ds.time.values[-1], freq='%dmin' % interval)
    else:
        dates = pd.date_range(ds.time.values[0],
                          ds.time.values[-1], periods=2)

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


def download_data(args, file_name):
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
            if not args.config in f:
                continue
            print("Downloading %s" % f)
            return_list = '/app/%s' % f
            last_file = f
        if last_file == file_name:
            return []
        sftp.get(last_file, localpath='/app/%s' % last_file, callback=progress)

    transport.close()
    print("Download done in %3.2f minutes" % ((time.time() - bt)/60.0))
    return return_list


def worker_main(args, plugin):
    interval = int(args.interval)
    print('opening input %s' % args.input)
    old_file = ""
    run = True
    with Plugin() as plugin:
        while run:
            class_names = ['clear', 'cloudy', 'rain']
            file_name = download_data(args, old_file)
            if file_name == []:
                time.sleep(180)
                continue
            model = open_load_model(args.model)
            print("Processing %s" % file_name)
            dsd_ds = load_file(file_name)
            dsd_ds = process_file(dsd_ds)
            if args.config == "User5":
                dsd_ds["range"] = dsd_ds["range"] * np.sin(np.pi / 3)
            scp = get_scp(dsd_ds, args.model, args.config)
            out_predict = model.predict(xgb.DMatrix(scp['input_array']))
            for i in range(len(out_predict)):
                print(scp['time_bins'][i].timestamp())
                print(str(
                    scp['time_bins'][i]) + ':' + class_names[int(out_predict[i])])
                    plugin.publish("weather.classifier.class",
                            out_predict[i],
                            timestamp=scp['time_bins'][i].timestamp())
            if args.loop == False:
                run = False     
        dsd_ds.close() 
         
 
def main(args):
    if args.verbose:
        print('running in a verbose mode')
    worker_main(args, plugin)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--verbose', dest='verbose',
        action='store_true', help='Verbose')
    parser.add_argument(
        '--input', dest='input',
        action='store', default='sgpdlacfC1.00',
        help='Path to input device or ARM datastream name')
    parser.add_argument(
        '--model', dest='model',
        action='store', default='/app/modelsnrgt3.000000snrgt5.000000.json',
        help='Path to model')
    parser.add_argument(
        '--interval', dest='interval',
        action='store', default=0,
        help='Time interval in seconds')
    parser.add_argument(
            '--loop', action='store_true')
    parser.add_argument('--no-loop', action='store_false')
    parser.set_defaults(loop=True)
    parser.add_argument(
            '--config', dest='config', action='store', default='User5',
            help='Set to User5 for PPI or Stare for VPTs')
    parser.add_argument('--date', dest='date', action='store',
                        default=None,
                        help='Date of record to pull in (YYYY-MM-DD)')
    parser.add_argument('--time', dest='time', action='store',
                        default=None, help='Time of record to pull',
                        type=ascii)

    main(parser.parse_args())
