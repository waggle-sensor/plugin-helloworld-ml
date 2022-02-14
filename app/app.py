import numpy as np
import xgboost as xgb
import xarray as xr
import act
import highiq
import pandas as pd
import time
import datetime
import argparse
import urllib3
import shutil
import multiprocessing
import os
import base64
import paramiko
import json
import xarray as xr
import waggle.plugin as plugin

from datetime import datetime
from itertools import compress
from glob import glob


def open_load_model(model_path):
    print(model_path)
    bst = xgb.Booster()
    bst.load_model(model_path)
    return bst


def load_file(file_path):
    test_file = highiq.io.read_00_data(file_path, 'sgpdlprofcalC1.home_point')
    return test_file


def process_file(ds):
    print("Processing lidar moments...")
    ti = time.time()
    my_list = []
    i = 0
    for x in ds.groupby_bins('time', ds.time.values[::100]):
        d = x[1]
        d['acf_bkg'] = d['acf_bkg'].isel(time=1)
        my_list.append(highiq.calc.get_psd(d))
    ds = xr.concat(my_list, dim='time')
    ds_out = highiq.calc.get_lidar_moments(ds_out)
    print("Done in %3.2f minutes" % ((time.time() - ti) / 60.))
    return ds_out


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
    transport = paramiko.Transport('emerald.adc.arm.gov', 22)
    username = 'rjackson'
    
    transport.connect(username=username, password=passwd)
    return_list = []
    with paramiko.SFTPClient.from_transport(transport) as sftp:
        sftp.chdir('/data/datastream/sgp/%s' % args.input)
        if args.date is None and args.time is None:
            file_list = [sorted(sftp.listdir())[-1]]
        elif args.time is None:
            file_list = sorted(sftp.listdir())
            for f in file_list:
                if not args.date in f:
                    file_list.remove(f)
        else:
            file_list = ['%s.%s.%s.nc' % (args.input, args.date, args.time)]
        for f in file_list:
            if os.path.exists('/app/%s' % f):
                continue
            
            # Only process vertically pointing data for now
            if not 'Stare' in f:
                continue
            print("Downloading %s" % f)
            sftp.get(f, localpath='/app/%s' % f, callback=progress)
            return_list.append('/app/%s' % f)
    transport.close()
    print("Download done in %3.2f minutes" % ((time.time() - bt)/60.0))
    return return_list

def worker_main(args, plugin):
    interval = int(args.interval)
    print('opening input %s' % args.input)
    class_names = ['clear', 'cloudy', 'rain']
    file_list = download_data(args)
    
    model = open_load_model(args.model)
    for file_name in file_list:
        file_name = file_list[-1]
        print("Processing %s" % file_name)
        input_ds = load_file(file_name)
        dsd_ds = process_file(input_ds)
        scp = get_scp(dsd_ds, args.model)
        input_ds.close()
        out_predict = model.predict(xgb.DMatrix(scp['input_array']))
        for i in range(len(out_predict)):
            print(str(
                scp['time_bins'][i]) + ':' + class_names[int(out_predict[i])])
            plugin.publish("weather.classifier.class",
                           class_names[int(out_predict[i])],
                           timestamp=scp['time_bins'][i])
        
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
    plugin.subscribe("weather.classifier.class")
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
    parser.add_argument('--date', dest='date', action='store',
                        default=None,
                        help='Date of record to pull in (YYYY-MM-DD)')
    parser.add_argument('--time', dest='time', action='store',
                        default=None, help='Time of record to pull',
                        type=ascii)

    main(parser.parse_args())
