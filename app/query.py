import sage_data_client
import time

while True:
    df = sage_data_client.query(start="-120m",
        filter={"name": "weather.classifier.class"}
    )
    print(df)
    time.sleep(60)