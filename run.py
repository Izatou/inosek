from main import CMD2_PREDICT, bootstrap
import pandas as pd

def main():
    boot = bootstrap()

    # For example, we take the data from csv
    allData = pd.read_csv("data.csv")

    # Convert them to dataframe
    # with header 'MQ2_ADC', 'MQ3_ADC', 'MQ4_ADC', 'TGS2610_ADC', 'TGS2600_ADC', 'TGS822_ADC', 'MQ137_ADC', 'MQ138_ADC'
    allData = allData.loc[:,['PROCESS', 'MQ2_ADC', 'MQ3_ADC', 'MQ4_ADC', 'TGS2610_ADC', 'TGS2600_ADC', 'TGS822_ADC', 'MQ137_ADC', 'MQ138_ADC']]    

    # Split the PROCESS
    (valid, covid) = CMD2_PREDICT(
        boot, 
        allData[allData['PROCESS'] == 'P1'],
        allData[allData['PROCESS'] == 'P2'],
        allData[allData['PROCESS'] == 'P3'],
    )

    print(valid, covid) #7435


if __name__ == "__main__":
    main()

