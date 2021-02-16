import numpy as np
import xgboost as xgb
import xarray as xr
import act
import highiq
import pandas as pd
import time
import datetime
import argparse
import multiprocessing

from itertools import compress
from glob import glob


def open_load_model(model_path):
    bst = xgb.load_model(model_path)
    return bst

def load_file(file_path):
    test_file = highiq.io.load_arm_netcdf(file_path)
    return test_file

def process_file(ds):
    ds_out = highiq.calc.get_psd(ds)
    ds_out = highiq.calc.get_lidar_moments(ds_out)
    return ds_out

def get_scp(ds, model_name, interval=5):
    range_bins = np.arange(60., 11280., 120.)
    # Parse model string for locations of snr, mean_velocity, spectral_width
    locs = 0
    snr_thresholds = []
    scp_ds = {}
    dates = pd.date_range(ds.time.values[0],
                          ds.time.values[-1], freq='%dmin' % interval)
    while locs > -1:
        locs = model_name.find("snr")
        if locs > -1:
            snr_thresholds.append(float(model_name[locs+5:locs+15]))
            scp_ds['snrgt%f' % snr_thresholds[-1]] = np.zeros(len(dates) - 1, len(times) - 1)

    for i in range(len(dates)-1):
        time_inds = np.argwhere(np.logical_and(ds.time.values >= dates[i],
                                               ds.time.values < dates[i+1]))
        for j in range(len(range_bins) - 1):
            range_inds = np.argwhere(np.logical_and(
                ds.range.values >= range_bins[j], ds.range.values < range_bins[j+1]))
            snrs = ds['snr'].values[time_inds, range_inds]
            for snr_thresh in snr_thresholds:
                scp_ds['snrgt%f' % snr_thresh][i, j] = len(np.argwhere(snrs > snr_thresh)) / \
                                                       (len(time_inds) * len(range_inds))
    scp_ds['input_array'] = np.concatenate([scp_ds[var_keys] in scp_ds.keys()], axis=1)

    scp_ds['time_bins'] = dates
    return scp_ds
        
def download_data(date, time):
    act.discovery.download_data('rjackson', '3326641ebc6b55aa',
                                'sgpdlacfC1.a1', date, date, time)
    
def worker_main(args, heartbeat):
    interval = int(args.interval)
    print(f'opening input {args.input}', flush=True)
    
    class_names = ['clear', 'cloudy', 'rain']
    # First get the time period using act

    if args.time is None:
        file_list = glob('sgpdlacfC1.a1.%s*.nc' % args.date)
        file_name = file_list[-1]
    else:
        file_name = 'sgpdlacfC1.a1.%s.%s.nc' % (args.date, args.time)

    input_ds = load_file(file_name)

    model = open_load_model(args.model)
    if args.date == datetime.date.today().strftime('%Y%m%d') and time is None:
        while True:
            file_list = glob('sgpdlacfC1.a1.%s*.nc' % args.date)
            if file_list == []:
                download_data(args.date)
            file_list = glob('sgpdlacfC1.a1.%s*.nc' % args.date)
            file_name = file_list[-1]
            dsd_ds = process_file(input_ds)
            scp = get_scp(dsd_ds)
            input_ds.close()
            dsd_ds.close()
            out_predict = model.predict(scp['input_array'])
            for i in range(len(out_predict)):
                print(scp['time_bins'][i] + ':' + class_names[int(out_predict[i])])
    else:
        download_data(args.date, args.time)
        file_list = glob('sgpdlacfC1.a1.%s*.nc' % args.date)
        file_name = file_list[-1]
        dsd_ds = process_file(input_ds)
        scp = get_scp(dsd_ds)
        input_ds.close()
        dsd_ds.close()
        out_predict = model.predict(scp['input_array'])
        for i in range(len(out_predict)):
            print(scp['time_bins'][i] + ':' + class_names[int(out_predict[i])])


def main(args):
    if args.verbose:
        print('running in a verbose mode', flush=True)

    heartbeat = multiprocessing.Queue()
    worker = multiprocessing.Process(
        target=worker_main, args=(args, heartbeat))

    model = open_load_model(args.model)
    
    try:
        worker.start()
        while True:
            heartbeat.get(timeout=30)  # throws an exception on timeout
    except Exception:
        pass

    # if we reach this point, the worker process has stopped
    worker.terminate()
    raise RuntimeError('worker is no longer responding')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-verbose', dest='verbose',
        action='store_true', help='Verbose')
    parser.add_argument(
        '-input', dest='input',
        action='store', default='sgpdlacfC1.a1',
        help='Path to input device or stream')
    parser.add_argument(
        '-model', dest='model',
        action='store', default='modelsnrgt3.000000snrgt5.000000.json',
        help='Path to model')
    parser.add_argument(
        '-interval', dest='interval',
        action='store', default=0,
        help='Time interval in seconds')
    parser.add_argument('-date', dest='date', action='store',
                        default=datetime.date.today().strftime('%Y%m%d'),
                        help='Date of record to pull')
    parser.add_argument('-time', dest='time', action='store',
                        default=None, help='Time of record to pull')


    main(parser.parse_args())
