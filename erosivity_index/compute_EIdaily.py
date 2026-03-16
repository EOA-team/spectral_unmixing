import os
import pandas as pd
import numpy as np

def compute_EI30(input_dir, output_dir, timestep_min=10, min_event_precip=12.7, dry_hours=6):
    """
    Compute EI30 (RUSLE erosive rainfall event erosivity) for CSV files in input_dir.
    If there are gaps in the raw timeseries:
    - less than 30min, the rainfall is set to 0 but doesnt split the rainfall event
    - >30min, the rainfall event is split and rainfall values are set to 0

    Parameters:
        input_dir (str): Folder with CSV files containing 'timestamp' and 'precipitation' (mm) columns.
        output_dir (str): Folder where EI30 CSVs will be saved.
        timestep_min (int): Time resolution of data in minutes (default 10).
        min_event_precip (float): Minimum total rainfall (mm) to consider event erosive.
        dry_hours (float): Dry period (hours) to separate events (default 6).
    """
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if not file.endswith('.csv'):
            continue

        file_path = os.path.join(input_dir, file)
        print(f"Processing {file}")
        df = pd.read_csv(file_path)

        if 'time' not in df.columns or 'rre150z0' not in df.columns:
            print(f"Skipping {file}: missing required columns")
            continue

        # Preprocess
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)

        # Force regular 10min intervals -> if missing, set precip to NaN
        df = df.set_index('time')
        expected_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=f'{timestep_min}min')
        df = df.reindex(expected_index)
        df.index.name = 'time'

        # Deal with missing time gaps
        df['missing'] = df['rre150z0'].isna()
        df['gap_group'] = (df['missing'] != df['missing'].shift()).cumsum()
        gap_lengths = df.groupby('gap_group')['missing'].transform('sum')
        gap_minutes = gap_lengths * timestep_min

        small_gap = 30     # minutes
        large_gap = 180    # minutes
        df.loc[(df['missing']) & (gap_minutes <= small_gap), 'rre150z0'] = 0 # small gpas: set precip to 0
        df['large_gap'] = (df['missing']) & (gap_minutes >= large_gap) # flag large gaps
        df['medium_gap'] = (df['missing']) & (gap_minutes > small_gap) & (gap_minutes < large_gap)
        df['rre150z0'] = df['rre150z0'].fillna(0)
        df = df.reset_index()

        # Remove negative precipitation
        df['rre150z0'] = df['rre150z0'].clip(lower=0)
            
        # Compute rainfall intensity (mm/h)
        df['ir'] = df['rre150z0'] * (60 / timestep_min)

        # Compute unit rainfall energy
        df['er'] = np.where(
            df['rre150z0'] >= 0.1,
            0.29 * (1 - 0.72 * np.exp(-0.05 * df['ir'])),
            0
        )

        # Kinetic energy
        df['E'] = df['er'] * df['rre150z0']

        # Identify events: dry period >= dry_hours separates events
        dry_steps = int(dry_hours * 60 / timestep_min)
        df['dry'] = df['rre150z0'] == 0
        df['dry_group'] = (df['dry'] != df['dry'].shift()).cumsum()
        df['dry_length'] = df.groupby('dry_group')['dry'].transform('sum')
        df['event_break'] = (df['dry'] & (df['dry_length'] >= dry_steps)) | df['large_gap'] | df['medium_gap']
        df['event_id'] = df['event_break'].cumsum()

        # Compute EI30 per event
        results = []
        intervals_30min = int(30 / timestep_min)

        for event_id, event_df in df.groupby('event_id'):

            # Keep only rainfall timesteps
            rain_event = event_df[event_df['rre150z0'] > 0]
            if rain_event.empty:
                continue

            total_precip = event_df['rre150z0'].sum()
            if total_precip < min_event_precip:
                continue

            # Maximum 30-min rainfall intensity
            rain_30min = event_df['rre150z0'].rolling(intervals_30min).sum()
            I30 = (rain_30min * (60 / 30)).max()  # mm/h

            # EI30
            E_total = rain_event['E'].sum()
            EI30 = E_total * I30

            results.append({
                'event_id': event_id,
                'start_time': event_df['time'].iloc[0],
                'end_time': event_df['time'].iloc[-1],
                'total_precipitation': total_precip,
                'I30': I30,
                'EI30': EI30
            })

        # Save results for this file
        results_df = pd.DataFrame(results)
        output_file = os.path.join(output_dir, f"EI30_{file}")
        results_df.to_csv(output_file, index=False)
        print(f"Saved EI30 results to {output_file}")

    return


def compute_EIdaily_avg(input_dir, output_dir):
    """
    Compute average daily EI (climatology) from per-event EI30 CSVs,
    splitting multi-day events proportionally.

    Parameters:
        input_dir (str): Folder containing EI30 CSVs.
        output_dir (str): Folder where average daily EI CSVs will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if not file.endswith('.csv'):
            continue

        file_path = os.path.join(input_dir, file)
        print(f"Processing {file} for multi-day average daily EI")

        df = pd.read_csv(file_path)
        if 'start_time' not in df.columns or 'end_time' not in df.columns or 'EI30' not in df.columns:
            print(f"Skipping {file}: missing required columns")
            continue

        df['start_time'] = pd.to_datetime(df['start_time'])
        df['end_time'] = pd.to_datetime(df['end_time'])

        # Split multi-day events proportionally
        daily_records = []

        for _, row in df.iterrows():
            start = row['start_time']
            end = row['end_time']
            total_ei = row['EI30']

            # Generate all days the event touches
            days = pd.date_range(start=start.date(), end=end.date(), freq='D')

            if len(days) == 1:
                # Single day event
                daily_records.append({'date': start.date(), 'EI_daily': total_ei})
            else:
                # Multi-day event: split EI30 proportionally by duration
                total_seconds = (end - start).total_seconds()
                for day in days:
                    day_start = pd.Timestamp(day)
                    day_end = pd.Timestamp(day + pd.Timedelta(days=1))

                    # Calculate overlap in seconds
                    overlap_start = max(start, day_start)
                    overlap_end = min(end, day_end)
                    overlap_seconds = (overlap_end - overlap_start).total_seconds()

                    ei_fraction = total_ei * (overlap_seconds / total_seconds)
                    daily_records.append({'date': day, 'EI_daily': ei_fraction})

        # Aggregate EI_daily per day
        daily_df = pd.DataFrame(daily_records)
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        daily_df = daily_df.groupby('date', as_index=False)['EI_daily'].sum()
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        daily_df['doy'] = daily_df['date'].dt.dayofyear

        # Average over DOY to get climatology
        avg_doy_df = daily_df.groupby('doy', as_index=False)['EI_daily'].mean()
        avg_doy_df.rename(columns={'EI_daily': 'EI_daily_avg'}, inplace=True)

        # Fill missing DOY with 0
        all_doy = pd.DataFrame({'doy': range(1, 367)})
        avg_doy_complete = all_doy.merge(avg_doy_df, on='doy', how='left')
        avg_doy_complete['EI_daily_avg'] = avg_doy_complete['EI_daily_avg'].fillna(0)

        # Save result
        output_file = os.path.join(output_dir, f"EIdaily_avg_{file}")
        avg_doy_complete.to_csv(output_file, index=False)
        print(f"Saved multi-day average daily EI to {output_file}")

    return
  

def compute_EI_daily_percent(input_dir, output_dir):
    """
    For each average daily EI file, compute daily percentage and cumulative percentage of annual EI.

    Parameters:
        input_dir (str): Folder containing EIdaily_avg CSVs (doy + EI_daily_avg).
        output_dir (str): Folder to save updated CSVs with percentages.
    """
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if not file.endswith('.csv'):
            continue

        file_path = os.path.join(input_dir, file)
        print(f"Processing {file} for daily percentages")

        df = pd.read_csv(file_path)
        if 'doy' not in df.columns or 'EI_daily_avg' not in df.columns:
            print(f"Skipping {file}: missing required columns")
            continue

        # Total annual erosivity
        total_EI = df['EI_daily_avg'].sum()
        if total_EI == 0:
            df['EI_daily_percent'] = 0
            df['EI_daily_cumsum'] = 0
        else:
            # Compute daily percent contribution
            df['EI_daily_percent'] = df['EI_daily_avg'] / total_EI * 100
            # Cumulative percent
            df['EI_daily_cumsum'] = df['EI_daily_percent'].cumsum()

        # Save updated CSV
        output_file = os.path.join(output_dir, f"percent_{file}")
        df.to_csv(output_file, index=False)
        print(f"Saved daily percent file to {output_file}")
        
    return


# Compute EI30 based on 10min data
input_dir = 'stations_data/stations_csv'       
output_dir = 'stations_data/stations_EI30'
compute_EI30(input_dir, output_dir)


# Compute EIdaily: accumulate values daily and compute long term daily mean
input_dir = 'stations_data/stations_EI30'     
output_dir = 'stations_data/stations_EIdaily' 
compute_EIdaily_avg(input_dir, output_dir)

# Conver to daily percentage sum
input_dir = 'stations_data/stations_EIdaily'       # folder with EIdaily_avg CSVs
output_dir = 'stations_data/stations_EIdaily_percent'  # folder to save daily % and cumulative %
compute_EI_daily_percent(input_dir, output_dir)