# Algorithm description

This algorithm will classify raw ARM Doppler Lidar data into three categories: clear, cloudy, and rainy. The algorithm 
was developed by training a gradient boosting tree using [XGBoost](https://xgboost.readthedocs.io). The tree was trained
on three months of ARM Doppler Lidar observations and is based off of the statistical coverage product of signal to
noise ratio (May and Lane 2009). On the testing dataset, we report that the algorithm correctly classified images
as clear, cloudy, and rainy about 94% of the time.

The algorithm will take as inputs the raw Doppler lidar autocorrelation functions. It will then use HighIQ to generate
the signal to noise ratio that provides a view of the cloud, precipitation, and aerosol particles detected by the lidar.
After this is done, then the statistical coverage product will be generated for use by the classifier. Finally, the
gradient boosting tree will classify the time period based off of the generated statistical coverage product.

May, P.T. and Lane, T.P. (2009), A method for using weather radar data to test cloud resolving models.
Met. Apps, 16: 425-432. https://doi.org/10.1002/met.150

Friedman, J.H., (2001), Greedy function approximation: a gradient boosting machine. Annals of statistics, pp. 1189–1232
# Data needed

The input data are autocorrelation function files from the ARM dlacf.a1 dataset. While the algorithm will work on input
SNR data, the design from raw data is there in order to support deciding how to process and store the data from the raw
observations before they are stored on the ARM archive.
The path to these files must be specified in the app.py file.
Support for downloading from a live feed will be added in a later version.

The algorithm will then output scene classifications over five minute time periods. The three classifications supported are
currently 'clear', 'cloudy', and 'rainy'.

