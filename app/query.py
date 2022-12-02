import sage_data_client
import time


df = sage_data_client.query(start="-168h",
        filter={"name": "weather.classifier.class"}
)
print(df)
df.to_csv('classifications.csv')
