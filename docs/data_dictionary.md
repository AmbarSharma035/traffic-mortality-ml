# Complete Data Dictionary

This document defines the unified schema used for training our machine learning models, ensuring consistency across disparate data sources.

| Column | Meaning | Type | Source | Missing Treatment | Transformation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `accident_id` | Unique identifier for the accident record | String | All | Drop if missing | None (Index) |
| `latitude` | Geographic latitude coordinate | Float | All | Drop if missing | None |
| `longitude` | Geographic longitude coordinate | Float | All | Drop if missing | None |
| `timestamp` | Date and time of the accident | Datetime | All | Drop if missing | Parsed to datetime |
| `severity` | Target variable: accident severity level | String | All | Drop if missing | Mapped to Minor/Serious/Fatal |
| `weather_condition` | Description of weather (e.g., Clear, Rain) | String | US, UK | Mode imputation | Standardized grouping |
| `visibility` | Visibility distance (miles/km) | Float | US | Median imputation | Converted to standard unit |
| `temperature` | Air temperature (F/C) | Float | US | Median imputation | Converted to standard unit |
| `humidity` | Relative humidity percentage | Float | US | Median imputation | None |
| `wind_speed` | Wind speed | Float | US | Median imputation | None |
| `road_surface` | Condition of road (Dry, Wet, Snow) | String | UK, US | Mode imputation | Standardized grouping |
| `lighting_condition` | Lighting (Daylight, Darkness) | String | UK, US | Mode imputation | Standardized grouping |
| `junction_type` | Type of junction | String | UK, US | Mode imputation | Standardized grouping |
| `has_junction` | Presence of a junction (Binary) | Boolean | US | Imputed False | Cast to int (0/1) |
| `has_crossing` | Presence of a pedestrian crossing (Binary) | Boolean | US | Imputed False | Cast to int (0/1) |
| `has_traffic_signal`| Presence of a traffic signal (Binary) | Boolean | US | Imputed False | Cast to int (0/1) |
| `hour` | Hour of the day (0-23) | Integer | Engineered| - | Extracted from timestamp |
| `day_of_week` | Day of the week (0=Mon, 6=Sun) | Integer | Engineered| - | Extracted from timestamp |
| `month` | Month of the year (1-12) | Integer | Engineered| - | Extracted from timestamp |
| `is_weekend` | Whether the day is Saturday or Sunday | Boolean | Engineered| - | Derived from day_of_week |
| `is_night` | Whether the time is night (e.g., 6 PM - 6 AM) | Boolean | Engineered| - | Derived from hour |
| `country` | Country code of the data source | String | All | Constant per source| Label encoding |
| `region` | State/Region of the accident | String | All | Mode imputation | Target or label encoding |

### Engineered Features

| Feature | Meaning | Derivation |
| :--- | :--- | :--- |
| `rush_hour` | Indicates if the accident occurred during typical rush hours (7-9 AM, 4-6 PM) | Derived from `hour` |
| `time_of_day` | Categorical time (Morning, Afternoon, Evening, Night) | Derived from `hour` |
| `season` | Season of the year (Spring, Summer, Fall, Winter) | Derived from `month` |
| `weather_severity`| Ordinal ranking of weather (Clear=0, Rain=1, Snow=2, Severe=3) | Derived from `weather_condition` |
| `visibility_category`| Binned visibility (Poor, Fair, Good) | Derived from `visibility` |
| `temperature_category`| Binned temperature (Freezing, Cold, Mild, Hot) | Derived from `temperature` |
| `has_road_feature`| Flag if ANY road feature (crossing, junction, signal) is present | `has_junction` OR `has_crossing` OR `has_traffic_signal` |
