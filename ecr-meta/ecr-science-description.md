# Algorithm description

This algorithm will classify raw ARM Doppler Lidar data into three categories: clear, cloudy, and rainy. The algorithm was developed by training a gradient boosting tree using the [XGBoost](https://xgboost.readthedocs.io). The tree was trained on three months of ARM Doppler Lidar observations and is based off of the statistical coverage product of signal to noise ratio (May and Lane 2009). On the testing dataset, we report that the algorithm correctly classified images as clear, cloudy, and rainy about 94% of the time.

May, P.T. and Lane, T.P. (2009), A method for using weather radar data to test cloud resolving models. Met. Apps, 16: 425-432. https://doi.org/10.1002/met.150

# Data needed

The input data are autocorrelation function files from the dlacf.a1 dataset. The path to these files must be specified in the app.py file. Support for downloading from a live feed will be added in a later version.

