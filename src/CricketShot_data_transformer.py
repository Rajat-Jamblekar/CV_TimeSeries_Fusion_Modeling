import pandas as pd
import numpy as np

def transform_himanish_to_your_format(input_csv_path, output_csv_path='transformed_cricket_data.csv'):
    """
    Transform Himanish's cricket dataset to match your feature set
    """
    
    # Read the data
    df = pd.read_csv(input_csv_path)
    
    # Filter for T20 matches only
    df = df[df['max_balls'] == 120].copy()
    
    # Filter out rows with NO_SHOT or missing shot data
    df = df[df['shot'].notna() & (df['shot'] != 'NO_SHOT')].copy()
    
    print(f"Total T20 balls with valid shots: {len(df)}")
    
    # Create transformed dataset
    transformed = pd.DataFrame()
    
    # 1. SHOT_TYPE - map from 'shot' column
    transformed['shot_type'] = df['shot'].str.title()
    
    # 2. BOWLER_TYPE - map from 'bowl_kind'
    def map_bowler_type(bowl_kind):
        if pd.isna(bowl_kind):
            return 'fast'
        bowl_kind = str(bowl_kind).lower()
        if 'spin' in bowl_kind or 'chinaman' in bowl_kind or 'googly' in bowl_kind:
            return 'spin'
        return 'fast'
    
    transformed['bowler_type'] = df['bowl_kind'].apply(map_bowler_type)
    
    # 3. INNINGS
    transformed['innings'] = df['inns']
    
    # 4. BALLS_REMAINING
    transformed['balls_remaining'] = df['inns_balls_rem']
    
    # 5. RUNS_REQUIRED (only for innings 2)
    transformed['runs_required_if_innings2'] = df.apply(
        lambda row: row['inns_runs_rem'] if row['inns'] == 2 else np.nan, 
        axis=1
    )
    
    # 6. VENUE_EXPECTED_RUNRATE (only for innings 1)
    # Calculate from historical ground data - use median run rate at this ground
    ground_rr = df.groupby('ground')['inns_rr'].median().to_dict()
    transformed['venue_expected_runrate_if_innings1'] = df.apply(
        lambda row: round(ground_rr.get(row['ground'], 8.0), 1) if row['inns'] == 1 else np.nan,
        axis=1
    )
    
    # 7. PRESSURE_INDEX (only for innings 1)
    # Calculate as: (current_rr - expected_rr) * sqrt(balls_bowled/120)
    def calculate_pressure_index(row):
        if row['inns'] != 1:
            return np.nan
        balls_bowled = 120 - row['inns_balls_rem']
        if balls_bowled == 0:
            return 0.0
        expected_rr = ground_rr.get(row['ground'], 8.0)
        current_rr = row['inns_rr']
        pressure = (current_rr - expected_rr) * np.sqrt(balls_bowled / 120)
        return round(pressure, 2)
    
    transformed['pressure_index_if_innings1'] = df.apply(calculate_pressure_index, axis=1)
    
    # 8. GROUND_SCORING_BUCKET
    # Categorize grounds based on average scoring rate
    ground_avg_score = df.groupby('ground')['inns_runs'].median().to_dict()
    
    def categorize_ground(ground):
        avg = ground_avg_score.get(ground, 160)
        if avg >= 180:
            return 'High-Scoring Ground'
        elif avg <= 150:
            return 'Low-Scoring Ground'
        else:
            return 'Neutral Ground'
    
    transformed['ground_scoring_bucket'] = df['ground'].apply(categorize_ground)
    
    # 9. PITCH_TYPE
    # Infer from bowl_kind distribution and scoring patterns
    def infer_pitch_type(ground, df_subset):
        ground_data = df_subset[df_subset['ground'] == ground]
        if len(ground_data) < 10:
            return 'Two-Paced Surface'
        
        spin_pct = (ground_data['bowl_kind'].str.contains('spin', case=False, na=False).sum() / len(ground_data))
        avg_score = ground_data['inns_runs'].median()
        
        if spin_pct > 0.4:
            return 'Dry Turning Track'
        elif avg_score > 180:
            return 'Flat Batting Pitch'
        elif avg_score < 150:
            return 'Two-Paced Surface'
        else:
            return 'Hard Bounce Surface'
    
    transformed['pitch_type'] = df['ground'].apply(lambda x: infer_pitch_type(x, df))
    
    # 10. WEATHER_CONDITION
    # Use daynight and random assignment for diversity
    def assign_weather(row):
        weather_options = ['Normal Conditions', 'Cloud Cover', 'Dry Heat', 
                          'Humid (Dew Likely)', 'Windy', 'fog']
        
        # Day matches more likely to have dry/cloud
        if row['daynight'] == 'day match':
            return np.random.choice(['Normal Conditions', 'Cloud Cover', 'Dry Heat', 'Windy'], 
                                   p=[0.4, 0.3, 0.2, 0.1])
        # Night matches more likely to have dew
        else:
            return np.random.choice(['Normal Conditions', 'Humid (Dew Likely)', 'Cloud Cover', 'fog'], 
                                   p=[0.3, 0.4, 0.2, 0.1])
    
    np.random.seed(42)  # For reproducibility
    transformed['weather_condition'] = df.apply(assign_weather, axis=1)
    
    # 11. WICKETS_LEFT
    transformed['wickets_left'] = 10 - df['inns_wkts']
    
    # 12. MATCH_FORMAT
    transformed['match_format'] = 'T20'
    
    # 13. SHOT_QUALITY - This is the KEY LABEL
    def determine_shot_quality(row):
        """
        Determine if shot is Good or Bad based on multiple factors
        """
        # Get outcome
        outcome = str(row['outcome']).lower()
        score = row['score']
        out = row['out']
        control = row['control']
        dismissal = str(row['dismissal'])
        
        # BAD shot criteria (prioritize these)
        if out == True or out == 'TRUE':
            return 'Bad'  # Got out = bad shot
        
        if 'bowled' in dismissal.lower() or 'caught' in dismissal.lower():
            return 'Bad'
        
        if 'missed' in row['shot'].lower() or 'misstimed' in row['shot'].lower():
            return 'Bad'
        
        # Control-based (if available)
        if pd.notna(control) and control == 0:
            return 'Bad'  # No control = bad
        
        # Dot balls under pressure
        if score == 0 and row['inns_balls_rem'] < 30:  # Death overs dot
            return 'Bad'
        
        # GOOD shot criteria
        if score >= 4:  # Boundary
            return 'Good'
        
        if score > 0 and pd.notna(control) and control == 1:  # Controlled scoring
            return 'Good'
        
        # Defensive shots when not under pressure
        if 'defense' in row['shot'].lower() or 'leave' in row['shot'].lower():
            if row['inns_balls_rem'] > 60:  # Early in innings
                return 'Good'
            else:
                return 'Bad'
        
        # Default: scoring shots = good, dots = bad
        if score > 0:
            return 'Good'
        else:
            return 'Bad'
    
    transformed['shot_quality'] = df.apply(determine_shot_quality, axis=1)
    
    # Add video filename and shot_id
    transformed.insert(0, 'shot_id', ['shot' + str(i+1) for i in range(len(transformed))])
    transformed.insert(1, 'video_filename', [f'shot{i+1}.mp4' for i in range(len(transformed))])
    
    # Save to CSV
    transformed.to_csv(output_csv_path, index=False)
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"TRANSFORMATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total shots transformed: {len(transformed)}")
    print(f"\nShot Quality Distribution:")
    print(transformed['shot_quality'].value_counts())
    print(f"\nPercentage - Good: {(transformed['shot_quality']=='Good').sum()/len(transformed)*100:.1f}%")
    print(f"Percentage - Bad: {(transformed['shot_quality']=='Bad').sum()/len(transformed)*100:.1f}%")
    print(f"\nInnings Distribution:")
    print(transformed['innings'].value_counts())
    print(f"\nBowler Type Distribution:")
    print(transformed['bowler_type'].value_counts())
    print(f"\nTop 10 Shot Types:")
    print(transformed['shot_type'].value_counts().head(10))
    print(f"\nData saved to: {output_csv_path}")
    
    return transformed

# Example usage:
if __name__ == "__main__":
    # Replace with your actual file path
    input_file = "odi_bbb-25.csv"  # Your downloaded file
    output_file = "transformed_cricket_shots.csv"
    
    try:
        df_transformed = transform_himanish_to_your_format(input_file, output_file)
        print("\n✅ Success! Your dataset is ready for XGBoost modeling.")
        print(f"\nFirst few rows:")
        print(df_transformed.head())
    except FileNotFoundError:
        print(f"❌ Error: Could not find '{input_file}'")
        print("Please update the input_file path to your actual CSV file location.")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your input file format.")