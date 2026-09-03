# Data Sources Validation

This document outlines the validation and metadata for the three primary datasets used in the Traffic Mortality ML project: MoRTH (India), US Accidents (USA), and STATS19 (UK).

| Source | Dataset Name | URL | Country | Years | Records | Key Columns | License | Download Method |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MoRTH (Govt of India)** | Road Accidents in India | [opencity.in](https://opencity.in), [morth.nic.in](https://morth.nic.in) | India | Variable | Aggregate State-level | State, Year, Total Accidents, Persons Killed, Persons Injured | OGL India | PDF reports / Manual CSV extraction |
| **Sobhan Moosavi** | US Accidents | [Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) | USA | 2016-2023 | ~7.7 Million | ID, Severity, Start_Time, Start_Lat, Start_Lng, Weather_Condition, Visibility, Temperature | CC BY-NC-SA 4.0 | Kaggle API |
| **Department for Transport** | Road Safety Data (STATS19) | [data.dft.gov.uk](https://data.dft.gov.uk/road-accidents-safety-data) | UK | 2023 | ~100k+ | accident_index, accident_severity, date, time, latitude, longitude, weather_conditions | OGL v3.0 | Direct HTTP (CSV) |
