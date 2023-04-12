from matplotlib import use 
use('Agg')
import numpy as np
import xarray as xr
import pandas as pd
import time
import argparse
import os
import xarray as xr
import tensorflow as tf
import glob
import globus_sdk
import matplotlib.pyplot as plt
import act

from waggle.plugin import Plugin
from datetime import datetime, timedelta
from scipy.signal import convolve2d
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

from glob import glob
# 1. import standard logging module
import logging
import utils

# 2. enable debug logging

logging.basicConfig(level=logging.DEBUG)

def convert_to_hours_minutes_seconds(decimal_hour, initial_time):
    delta = timedelta(hours=decimal_hour)
    return datetime(initial_time.year, initial_time.month, initial_time.day) + delta

def load_file(file):
    field_dict = utils.hpl2dict(file)
    initial_time = pd.to_datetime(field_dict['start_time'])
    
    time = pd.to_datetime([convert_to_hours_minutes_seconds(x, initial_time) 
        for x in field_dict['decimal_time']])

    ds = xr.Dataset(coords={'range':field_dict['center_of_gates'],
                            'time': time,
                            'azimuth': ('time', field_dict['azimuth'])},
                    data_vars={'radial_velocity':(['range', 'time'],
                                                  field_dict['radial_velocity']),
                               'beta': (('range', 'time'), 
                                        field_dict['beta']),
                               'intensity': (('range', 'time'),
                                             field_dict['intensity'])
                              }
                   )
    ds['snr'] = ds['intensity'] - 1
    return ds


def return_convolution_matrix(time_window, range_window):
    return np.ones((time_window, range_window)) / (time_window * range_window)

def make_imgs(ds, config, interval=5):
    range_bins = np.arange(60., 11280., 120.)
    # Parse model string for locations of snr, mean_velocity, spectral_width
    locs = 0
    snr_thresholds = []
    scp_ds = {}
    interval = 5
    dates = pd.date_range(ds.time.values[0], ds.time.values[-1], freq='%dmin' % interval)
    
    times = ds.time.values
    print(times)
    which_ranges = int(np.argwhere(ds.range.values < 8000.)[-1])
    ranges = np.tile(ds.range.values, (ds['snr'].shape[1], 1)).T
    
    ds['snr'] = ds['snr'] + 2 * np.log10(ranges + 1)
    conv_matrix = return_convolution_matrix(5, 5)
    snr_avg = convolve2d(ds['snr'].values, conv_matrix, mode='same')
    ds['stddev'] = (('range', 'time'), 
            np.sqrt(convolve2d((ds['snr'] - snr_avg) ** 2, conv_matrix, mode='same')))
    Zn = ds.stddev.values.T

    cur_time = times[0]
    end_time = times[-1]
    time_list = []
    start_ind = 0
    i = 0
    first_shape = None

    while cur_time < end_time:
        next_time = cur_time + np.timedelta64(interval, 'm')
        print((next_time, end_time))

        if next_time > end_time:
            next_ind = len(times)
        else:
            next_ind = np.argmin(np.abs(next_time - times))
        if (start_ind >= next_ind):
            break

        my_data = Zn[start_ind:next_ind, 0:which_ranges].T

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


        if not os.path.exists('/app/imgs/'):
            os.mkdir('/app/imgs')
        
        if not os.path.exists('/app/imgs/train'):
            os.mkdir('/app/imgs/train')

        fname = '/app/imgs/train/%d.png' % i
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


def worker_main(args):
    logging.debug("Loading model %s" % args.model)
    model = load_model(args.model)
    interval = int(args.interval)
    logging.debug('opening input %s' % args.input)

   # CLIENT_ID = "2a0ee37f-475d-4b7f-8149-d6a2eab37aab"
   # CLIENT_SECRET = "zQ+Sh52TiNXarGI/xfMeCBZ9ck5o2HSQ2t3K/vY94L4="

   # confidential_client = globus_sdk.ConfidentialAppAuthClient(CLIENT_ID, CLIENT_SECRET)

    # the useful values that you want at the end of this
  #  scopes = "urn:globus:auth:scope:transfer.api.globus.org:all"
  #  authorizer = globus_sdk.ClientCredentialsAuthorizer(confidential_client, scopes)
  #  tc = globus_sdk.TransferClient(authorizer=authorizer)
  #  local_endpoint = globus_sdk.LocalGlobusConnectPersonal()

  #  source_endpoint_id = "d8da717e-ca5c-11ed-9622-4b6fcc022e5a"
  #  target_endpoint_id = local_endpoint.endpoint_id
  #  print(target_endpoint_id)
  #  print("Endpoints Belonging to {}@clients.auth.globus.org:".format(CLIENT_ID))
  #  for ep in tc.endpoint_search(filter_scope="my-endpoints"):
  #      print("[{}] {}".format(ep["id"], ep["display_name"]))

    # create a Transfer task consisting of one or more items
  #  task_data = globus_sdk.TransferData(
  #      source_endpoint=source_endpoint_id, destination_endpoint=target_endpoint_id
  #  )
  #  task_data.add_item(
  #      "/202303/20230323/Background-230323-215053.txt",  # source
  #      "/~/Background-230323-215053.txt",  # dest
  #  )

    # submit, getting back the task ID
  #  task_doc = transfer_client.submit_transfer(task_data)
  #  task_id = task_doc["task_id"]
  #  print(f"submitted transfer, task_id={task_id}")
     
    old_file = ""
    run = True
    already_done = []
    with Plugin() as plugin:
        while run:
            class_names = ['clear', 'cloudy']

            stare_list = glob(os.path.join(args.input, 'Stare*.hpl'))
            
            for fi in stare_list:
                logging.debug("Processing %s" % fi)
                dsd_ds = load_file(fi)
                print(dsd_ds)
                time_list = make_imgs(dsd_ds, args.config)
                dsd_ds.close()
                file_list = glob('/app/imgs/*.png')
                print(file_list)
                
                img_gen = ImageDataGenerator(
                    preprocessing_function=preprocess_input)

                gen = img_gen.flow_from_directory(
                         '/app/imgs/', target_size=(256, 128), shuffle=False)
                out_predict = model.predict(gen).argmax(axis=1)
                
                for i, ti in enumerate(time_list):
                    if ti not in already_done:
                        tstamp = int(ti)
                        
                        if out_predict[i] == 0:
                            string = "clear"
                        else:
                            string = "clouds/rain"
                        logging.debug("%s: %s" % (str(ti), string))

                        plugin.publish("weather.classifier.class",
                                int(out_predict[i]),
                                timestamp=tstamp)
                        already_done.append(ti)
                    
            if args.loop == False:
                run = False


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
        action='store', default='/data',
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
            '--config', dest='config', action='store', default='dlacf',
            help='Set to User5 for PPI or Stare for VPTs')
    parser.add_argument('--date', dest='date', action='store',
                        default=None,
                        help='Date of record to pull in (YYYY-MM-DD)')
    parser.add_argument('--time', dest='time', action='store',
                        default=None, help='Time of record to pull')

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                    gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
        except RuntimeError as e:
            print(e)
    main(parser.parse_args())
                                            
