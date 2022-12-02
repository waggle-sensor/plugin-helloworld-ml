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
import xarray as xr
import tensorflow as tf

from waggle.plugin import Plugin
from datetime import datetime, timedelta
from scipy.signal import convolve2d
from tensorflow.keras.Model import load_model
from tensorflow.keras.preprocessing.image import load_img
from glob import glob
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
    test_file = highiq.io.load_arm_netcdf(file_path)
    return test_file

def return_convolution_matrix(time_window, range_window):
    return np.ones((time_window, range_window)) / (time_window * range_window)

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


def make_imgs(ds, config, interval=5):
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
    ranges = ds.range.values
    grid['snr'] = grid['snr'] + 2 * np.log10(ranges + 1)
    snr_avg = convolve2d(grid['snr'].values, conv_matrix, mode='same')
    grid['stddev'] = (('time', 'range'), np.sqrt(convolve2d((grid['snr'] - snr_avg) ** 2, conv_matrix, mode='same')))
    Zn = grid.stddev.values
    cur_time = times[0]
    end_time = times[-1]
    time_list = []
    start_ind = 0
    i = 0
    while cur_time < end_time:
        next_time = cur_time + np.timedelta64(interval, 'm')
        print((next_time, end_time))
        if next_time > end_time:
            next_ind = len(times)
        else:
            next_ind = np.argmin(np.abs(next_time - times))
        if (start_ind >= next_ind):
            break
        my_data = Zn[start_ind:next_ind, 0:which_ranges[-1]].T
        my_times = times[start_ind:next_ind]
        if len(my_times) == 0:
            break
        start_ind += next_ind - start_ind + 1

        if first_shape is None:
            first_shape = my_data.shape
        else:
            if my_data.shape[0] > first_shape[0]:
                my_data = my_data[:first_shape[0], :]
            elif my_data.shape[0] < first_shape[0]:
                my_data = np.pad(my_data, [(0, first_shape[0] - my_data.shape[0]), (0, 0)],
                                 mode='constant')

        if not os.path.exists('imgs'):
            os.mkdir('imgs')

        fname = 'imgs/%d.png' % i
        width = first_shape[0]
        height = first_shape[1]
        # norm = norm.SerializeToStri
        fig, ax = plt.subplots(1, 1, figsize=(1 * (height / width), 1))
        # ax.imshow(my_data)
        ax.pcolormesh(my_data, cmap='act_HomeyerRainbow', vmin=20, vmax=150)
        ax.set_axis_off()
        ax.margins(0, 0)
        try:
            fig.savefig(fname, dpi=300, bbox_inches='tight', pad_inches=0)
        except RuntimeError:
            plt.close(fig)
            continue
        plt.close(fig)
        i = i + 1
        del fig, ax
        time_list.append(cur_time)
        cur_time = next_time

    return time_list


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


def worker_main(args):
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
            model = load_model(args.model)
            print("Processing %s" % file_name)
            dsd_ds = load_file(file_name)
            dsd_ds = process_file(dsd_ds)
            if args.config == "User5":
                dsd_ds["range"] = dsd_ds["range"] * np.sin(np.pi / 3)
            time_list = make_imgs(dsd_ds, args.config)
            dsd_ds.close()
            file_list = glob('imgs/*.png')
            i = 0
            for fi in file_list:
                img_gen = load_img(fi)
                out_predict = model.predict(img_gen)
                tstamp = int(time_list[i] * 1e9)
                print(str(
                       scp['time_bins'][i]) + ':' + class_names[int(out_predict[i])])
                plugin.publish("weather.classifier.class",
                            int(out_predict[i]),
                            timestamp=tstamp)
            if args.loop == False:
                run = False     
        dsd_ds.close() 
         
 
def main(args):
    if args.verbose:
        print('running in a verbose mode')
    worker_main(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--verbose', dest='verbose',
        action='store_true', help='Verbose')
    parser.add_argument(
        '--input', dest='input',
        action='store', default='sgpdlacfC1.a0',
        help='Path to input device or ARM datastream name')
    parser.add_argument(
        '--model', dest='model',
        action='store', default='resnet50.hdf5',
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
