# Science

The DOE Atmospheric Radiation Measurement (ARM) program collects observations of cloud properties
from several Doppler lidars at the Southern Great Plains site in Oklahoma. However, the Doppler lidar spectra
are too large to be transferred over and processed on the current infrastructure. For example, one hour
of Doppler spectra from a vertically pointing Doppler lidar over SGP takes 1.2 GB. Therefore, doing multiyear
analyses of these spectra while including all data is an impossibly large task. However, if we can
filter out the data to only include time periods of interest to cloud modellers, this would vastly
reduce the amount of data throughput while adding more information about the clouds than what is currently
included in the ARM Archive. Therefore, this motived the development of a weather classifier for the vertically
pointing ARM Doppler Lidar.

This classifier will classify raw ARM Doppler Lidar data into three categories: clear, cloudy, and rainy. The algorithm 
was developed by training a gradient boosting tree using [XGBoost](https://xgboost.readthedocs.io). The tree was trained
on three months of ARM Doppler Lidar observations and is based off of the statistical coverage product of signal to
noise ratio (May and Lane 2009). On the testing dataset, we report that the algorithm correctly classified images
as clear, cloudy, and rainy about 94% of the time.

The algorithm will take as inputs the raw Doppler lidar autocorrelation functions. It will then use HighIQ to generate
the signal to noise ratio that provides a view of the cloud, precipitation, and aerosol particles detected by the lidar.
After this is done, then the statistical coverage product will be generated for use by the classifier. Finally, the
gradient boosting tree will classify the time period based off of the generated statistical coverage product. The figure
on the right shows the three classes of Doppler lidar 


# AI at Edge

This model uses gradient boosting machines in order to categorize ARM Doppler Lidar data.
The model takes in a statistical coverage product that provides percent coverage of echoes with SNR > 1, 3, 5, and 10 dB
at 200 evenly spaced height levels ranging 0 to 12 km with 60 m spacing for each five minute period. The gradient boosting machine will then classify this
vertical profile as clear, cloudy, and rainy. The gradient boosting machine is implemented in XGBoost which is
optimized to use a CUDA-compatible GPU when present. The default model uses the statistical coverage of SNR > 3 and 5 dB
as that model provided the best testing accuracy. However, models that classify using all permutations of the 
statistical coverage product are provided.

# Using the code

The input data are autocorrelation function files from the ARM dlacf.a1 dataset. While the algorithm will work on input
SNR data, the design from raw data is there in order to support deciding how to process and store the data from the raw
observations before they are stored on the ARM archive.
The path to these files must be specified in the app.py file.

# Arguments
--verbose: Display more information

--input [ARM datastream]: The ARM datastream to use as input

--model [json file]: The model file to use.

--interval [time interval]: The time interval to classify over

--date: The date to pull data from. Set to None to use latest date/time.

--time: The time to pull data from.

# Ontology

The algorithm will then output scene classifications (weather.classifier.class) 
over five minute time periods. 

weather.classifier.class: The three classifications supported are
currently 'clear', 'cloudy', and 'rainy'.


# Reference

May, P.T. and Lane, T.P. (2009), A method for using weather radar data to test cloud resolving models.
Met. Apps, 16: 425-432. https://doi.org/10.1002/met.150

Friedman, J.H., (2001), Greedy function approximation: a gradient boosting machine. Annals of statistics, pp. 1189–1232

