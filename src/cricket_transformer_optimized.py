import pandas as pd
import numpy as np

def transform_himanish_to_optimized_schema(input_csv_path, output_csv_path='optimized_cricket_data.csv', 
                                          match_format='all', min_rows=100):
    """
    Transform Himanish's cricket dataset to the optimized schema
    
    Parameters:
    - match_format: 'T20', 'ODI', 'Test', or 'all'
    - min_rows: Minimum rows required to proceed
    """
    
    # Read the data
    print("Loading data...")
    df = pd.read_csv(input_csv_path)
    print(f"Total rows loaded: {len(df)}")
    
    # Show available match formats
    if 'max_balls' in df.columns:
        print("\nMatch format distribution:")
        print(df['max_balls'].value_counts())
    
    # Filter by match format if specified
    if match_format == 'T20':
        df = df[df['max_balls'] == 120].copy()
        print(f"\nFiltered for T20 matches: {len(df)} rows")
    elif match_format == 'ODI':
        df = df[df['max_balls'] == 300].copy()
        print(f"\nFiltered for ODI matches: {len(df)} rows")
    # else keep all formats
    
    # Filter out rows with NO_SHOT or missing shot data
    initial_count = len(df)
    df = df[df['shot'].notna()].copy()
    df = df[df['shot'] != 'NO_SHOT'].copy()
    df = df[df['shot'] != ''].copy()
    
    print(f"Rows with valid shots: {len(df)} (removed {initial_count - len(df)} NO_SHOT entries)")
    
    if len(df) < min_rows:
        print(f"\n⚠️ WARNING: Only {len(df)} valid shots found. Need at least {min_rows}.")
        print("Try using match_format='all' to include all match types.")
        return None
    
    # Create transformed dataset
    transformed = pd.DataFrame()
    
    # ===== CORE IDENTIFIERS =====
    transformed['shot_id'] = ['shot' + str(i+1) for i in range(len(df))]
    transformed['video_filename'] = [f'shot{i+1}.mp4' for i in range(len(df))]
    
    # ===== SHOT INFORMATION =====
    # 1. Shot Type - Clean and standardize
    transformed['shot_type'] = df['shot'].str.replace('_', ' ').str.title()
    
    # 2. Bowler Type - Map from bowl_kind
    def map_bowler_type(bowl_kind):
        if pd.isna(bowl_kind) or str(bowl_kind).strip() == '':
            return 'fast'
        bowl_kind = str(bowl_kind).lower()
        if 'spin' in bowl_kind or 'chinaman' in bowl_kind or 'wrist' in bowl_kind:
            return 'spin'
        return 'fast'
    
    transformed['bowler_type'] = df['bowl_kind'].apply(map_bowler_type)
    
    # 3. Bowl Style - More granular
    def clean_bowl_style(style):
        if pd.isna(style) or str(style).strip() == '':
            return 'right_arm_fast_medium'
        style = str(style).strip().lower().replace(' ', '_')
        # Standardize common variations
        style_map = {
            'rfm': 'right_arm_fast_medium',
            'rm': 'right_arm_medium',
            'rf': 'right_arm_fast',
            'lfm': 'left_arm_fast_medium',
            'lf': 'left_arm_fast',
            'lm': 'left_arm_medium',
            'ob': 'right_arm_offspin',
            'lb': 'left_arm_orthodox',
            'slob': 'slow_left_arm_orthodox',
            'sla': 'slow_left_arm',
            'leg_spinner': 'right_arm_legspin',
            'leg_spin': 'right_arm_legspin',
            'ls': 'right_arm_legspin',
            'ws': 'right_arm_wristspin'
        }
        return style_map.get(style, style)
    
    transformed['bowl_style'] = df['bowl_style'].apply(clean_bowl_style)
    
    # ===== MATCH CONTEXT =====
    # 4. Innings
    transformed['innings'] = df['inns']
    
    # 5. Balls Remaining - handle both column name possibilities
    if 'inns_balls_rem' in df.columns:
        transformed['balls_remaining'] = df['inns_balls_rem'].fillna(0)
    else:
        # Calculate from max_balls and current balls
        transformed['balls_remaining'] = (df['max_balls'] - df['inns_balls']).fillna(0)
    
    # 6. Current Score
    transformed['current_score'] = df['inns_runs'].fillna(0).astype(int)
    
    # 7. Runs Required (only for innings 2)
    def get_runs_required(row):
        if row['inns'] != 2:
            return np.nan
        if 'inns_runs_rem' in df.columns and pd.notna(row['inns_runs_rem']):
            return int(row['inns_runs_rem'])
        elif 'target' in df.columns and pd.notna(row['target']):
            return int(row['target']) - int(row['inns_runs'])
        return np.nan
    
    transformed['runs_required_if_innings2'] = df.apply(get_runs_required, axis=1)
    
    # 8. Run Rate (current)
    def calculate_run_rate(row):
        if 'inns_rr' in df.columns and pd.notna(row['inns_rr']):
            return round(row['inns_rr'], 2)
        # Calculate manually
        balls_bowled = row['inns_balls'] if pd.notna(row['inns_balls']) else 1
        if balls_bowled > 0:
            overs = balls_bowled / 6
            return round(row['inns_runs'] / overs, 2) if overs > 0 else 0.0
        return 0.0
    
    transformed['run_rate'] = df.apply(calculate_run_rate, axis=1)
    
    # 9. Required Run Rate (only for innings 2)
    def calculate_required_rr(row):
        if row['inns'] != 2:
            return np.nan
        if 'inns_rrr' in df.columns and pd.notna(row['inns_rrr']):
            return round(row['inns_rrr'], 2)
        # Calculate manually
        if 'inns_balls_rem' in df.columns:
            balls_rem = row['inns_balls_rem']
        else:
            balls_rem = row['max_balls'] - row['inns_balls']
        
        runs_req = transformed.loc[row.name, 'runs_required_if_innings2']
        if pd.notna(runs_req) and balls_rem > 0:
            overs_rem = balls_rem / 6
            return round(runs_req / overs_rem, 2)
        return np.nan
    
    transformed['required_run_rate'] = df.apply(calculate_required_rr, axis=1)
    
    # 10. Wickets Left
    transformed['wickets_left'] = 10 - df['inns_wkts'].fillna(0).astype(int)
    
    # ===== BATSMAN STATE =====
    # 11. Batsman Runs (current batsman's score)
    transformed['batsman_runs'] = df['cur_bat_runs'].fillna(0).astype(int)
    
    # 12. Batsman Balls Faced
    transformed['batsman_balls_faced'] = df['cur_bat_bf'].fillna(0).astype(int)
    
    # ===== BALL CHARACTERISTICS (Observable) =====
    # 13. Ball Line
    def map_ball_line(line):
        if pd.isna(line) or str(line).strip() == '':
            return 'middle'
        line = str(line).lower().strip()
        
        # Map various line descriptions
        if 'outside off' in line or 'wide outside off' in line:
            return 'outside_off'
        elif 'off' in line and 'stump' in line:
            return 'off_stump'
        elif 'leg' in line and 'stump' in line:
            return 'leg_stump'
        elif 'down leg' in line or 'outside leg' in line or 'leg side' in line:
            return 'down_leg'
        elif 'middle' in line or 'stumps' in line:
            return 'middle'
        else:
            return 'middle'
    
    transformed['ball_line'] = df['line'].apply(map_ball_line)
    
    # 14. Ball Length
    def map_ball_length(length):
        if pd.isna(length) or str(length).strip() == '':
            return 'good_length'
        length = str(length).lower().strip()
        
        if 'yorker' in length:
            return 'yorker'
        elif 'full' in length or 'half volley' in length or 'overpitched' in length:
            return 'full'
        elif 'short' in length or 'bouncer' in length or 'back of length' in length:
            return 'short'
        elif 'good' in length:
            return 'good_length'
        else:
            return 'good_length'
    
    transformed['ball_length'] = df['length'].apply(map_ball_length)
    
    # ===== SHOT EXECUTION =====
    # 15. Shot Control
    def map_shot_control(control):
        if pd.isna(control):
            return np.nan
        try:
            control_val = int(float(control))
            return control_val if control_val in [0, 1] else np.nan
        except:
            return np.nan
    
    transformed['shot_control'] = df['control'].apply(map_shot_control)
    
    # 16. Shot Outcome
    def determine_shot_outcome(row):
        """Determine actual outcome of the shot"""
        # Check if out
        out_val = row['out']
        if out_val == True or str(out_val).lower() == 'true':
            return 'wicket'
        
        # Check extras
        wide = row['wide'] if pd.notna(row['wide']) else 0
        noball = row['noball'] if pd.notna(row['noball']) else 0
        
        if wide > 0:
            return 'wide'
        if noball > 0:
            return 'noball'
        
        # Check score
        score = row['score']
        if pd.isna(score):
            return '0_runs'
        
        score = int(score)
        if score == 0:
            return '0_runs'
        elif score == 1:
            return '1_run'
        elif score == 2:
            return '2_runs'
        elif score == 3:
            return '3_runs'
        elif score == 4:
            return '4_runs'
        elif score == 6:
            return '6_runs'
        else:
            return f'{score}_runs'
    
    transformed['shot_outcome'] = df.apply(determine_shot_outcome, axis=1)
    
    # ===== MATCH FORMAT =====
    # 17. Match Format - derive from max_balls
    def determine_format(max_balls):
        if pd.isna(max_balls):
            return 'Unknown'
        max_balls = int(max_balls)
        if max_balls == 120:
            return 'T20'
        elif max_balls == 300:
            return 'ODI'
        elif max_balls > 300:
            return 'Test'
        else:
            return 'Unknown'
    
    transformed['match_format'] = df['max_balls'].apply(determine_format)
    
    # ===== TARGET LABEL: SHOT QUALITY =====
    def determine_shot_quality(row):
        """Enhanced shot quality determination"""
        outcome = str(row['outcome']).lower() if pd.notna(row['outcome']) else ''
        score = row['score'] if pd.notna(row['score']) else 0
        out_val = row['out']
        out = (out_val == True or str(out_val).lower() == 'true')
        control = row['control']
        shot_name = str(row['shot']).lower() if pd.notna(row['shot']) else ''
        
        # Get balls remaining - handle different column names
        if 'inns_balls_rem' in df.columns:
            balls_rem = row['inns_balls_rem'] if pd.notna(row['inns_balls_rem']) else 100
        else:
            balls_rem = row['max_balls'] - row['inns_balls']
        
        wickets = 10 - (row['inns_wkts'] if pd.notna(row['inns_wkts']) else 0)
        
        # ===== CLEARLY BAD SHOTS =====
        if out:
            return 'Bad'
        
        if any(word in shot_name for word in ['missed', 'misstimed', 'bowled', 'edged']):
            return 'Bad'
        
        if pd.notna(control) and int(float(control)) == 0:
            return 'Bad'
        
        # Dot balls in death overs (last 20% of innings)
        max_balls = row['max_balls'] if pd.notna(row['max_balls']) else 120
        death_threshold = max_balls * 0.2
        if score == 0 and balls_rem < death_threshold and wickets > 2:
            return 'Bad'
        
        # High pressure dots in innings 2
        if score == 0 and row['inns'] == 2:
            if 'inns_rrr' in df.columns and pd.notna(row['inns_rrr']) and row['inns_rrr'] > 10:
                return 'Bad'
        
        # ===== CLEARLY GOOD SHOTS =====
        if score >= 4:
            return 'Good'
        
        if score > 0 and pd.notna(control) and int(float(control)) == 1:
            return 'Good'
        
        # Rotating strike in middle overs
        middle_start = max_balls * 0.3
        middle_end = max_balls * 0.7
        if score in [1, 2] and middle_start < balls_rem < middle_end:
            return 'Good'
        
        # Defense early in innings
        if 'defense' in shot_name or 'leave' in shot_name:
            if balls_rem > max_balls * 0.75:  # First 25% of innings
                return 'Good'
            else:
                return 'Bad'
        
        # ===== CONTEXT-DEPENDENT =====
        bat_bf = row['cur_bat_bf'] if pd.notna(row['cur_bat_bf']) else 0
        bat_runs = row['cur_bat_runs'] if pd.notna(row['cur_bat_runs']) else 0
        
        # New batsman
        if bat_bf < 5:
            return 'Good' if score >= 1 else 'Bad'
        
        # Set batsman
        if bat_runs > 30 and bat_bf > 20:
            return 'Bad' if score == 0 else 'Good'
        
        # Default
        return 'Good' if score > 0 else 'Bad'
    
    transformed['shot_quality'] = df.apply(determine_shot_quality, axis=1)
    
    # ===== DATA QUALITY CHECKS =====
    print("\n" + "="*70)
    print("DATA QUALITY CHECKS")
    print("="*70)
    print(f"Rows before cleaning: {len(transformed)}")
    
    # Debug: Check balls_remaining values
    print(f"\nBalls remaining stats:")
    print(f"  Non-null count: {transformed['balls_remaining'].notna().sum()}")
    print(f"  Null count: {transformed['balls_remaining'].isna().sum()}")
    print(f"  Min value: {transformed['balls_remaining'].min()}")
    print(f"  Max value: {transformed['balls_remaining'].max()}")
    print(f"  Sample values: {transformed['balls_remaining'].head(10).tolist()}")
    
    # More lenient filtering - just check if balls_remaining exists and is not negative
    transformed = transformed[
        transformed['balls_remaining'].notna()
    ].copy()
    
    # Convert to int and handle any remaining issues
    transformed['balls_remaining'] = transformed['balls_remaining'].astype(int)
    
    print(f"Rows after cleaning: {len(transformed)}")
    
    if len(transformed) == 0:
        print("\n⚠️ ERROR: No valid rows after filtering!")
        print("Possible issues:")
        print("  - No T20 matches in dataset (try match_format='all')")
        print("  - All rows have missing critical data")
        return None
    
    # ===== SAVE TO CSV =====
    transformed.to_csv(output_csv_path, index=False)
    
    # ===== STATISTICS =====
    print(f"\n{'='*70}")
    print(f"✅ TRANSFORMATION COMPLETE - OPTIMIZED SCHEMA")
    print(f"{'='*70}")
    print(f"Total shots: {len(transformed)}")
    
    print(f"\n📊 SHOT QUALITY DISTRIBUTION:")
    quality_dist = transformed['shot_quality'].value_counts()
    for quality, count in quality_dist.items():
        pct = count/len(transformed)*100
        print(f"   {quality}: {count} ({pct:.1f}%)")
    
    print(f"\n🏏 SHOT OUTCOME DISTRIBUTION (Top 10):")
    outcome_dist = transformed['shot_outcome'].value_counts().head(10)
    for outcome, count in outcome_dist.items():
        print(f"   {outcome}: {count}")
    
    print(f"\n⚾ BALL CHARACTERISTICS:")
    print(f"   Line distribution:")
    for line, count in transformed['ball_line'].value_counts().head(5).items():
        print(f"      {line}: {count}")
    print(f"   Length distribution:")
    for length, count in transformed['ball_length'].value_counts().items():
        print(f"      {length}: {count}")
    
    print(f"\n🎯 SHOT CONTROL (where available):")
    control_dist = transformed['shot_control'].value_counts()
    total_with_control = control_dist.sum()
    if total_with_control > 0:
        for control, count in control_dist.items():
            pct = count/total_with_control*100
            label = "Controlled" if control == 1 else "Not Controlled"
            print(f"   {label}: {count} ({pct:.1f}%)")
    else:
        print("   No control data available")
    
    print(f"\n📋 MATCH FORMAT DISTRIBUTION:")
    for fmt, count in transformed['match_format'].value_counts().items():
        print(f"   {fmt}: {count}")
    
    print(f"\n🏃 INNINGS DISTRIBUTION:")
    for innings, count in transformed['innings'].value_counts().items():
        print(f"   Innings {innings}: {count}")
    
    print(f"\n🎳 BOWLER DISTRIBUTION:")
    for bowler, count in transformed['bowler_type'].value_counts().items():
        print(f"   {bowler}: {count}")
    
    print(f"\n🔥 TOP 10 SHOT TYPES:")
    for shot, count in transformed['shot_type'].value_counts().head(10).items():
        print(f"   {shot}: {count}")
    
    print(f"\n💾 Data saved to: {output_csv_path}")
    print(f"{'='*70}\n")
    
    # ===== SAMPLE ROWS =====
    print("📋 SAMPLE ROWS (First 3):")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(transformed.head(3).to_string(index=False))
    
    return transformed


def generate_annotation_template(output_path='annotation_template.csv'):
    """Generate a template CSV for manual annotation"""
    template = pd.DataFrame({
        'shot_id': ['shot1', 'shot2', 'shot3'],
        'video_filename': ['shot1.mp4', 'shot2.mp4', 'shot3.mp4'],
        'shot_type': ['Cover Drive', 'Pull Shot', 'Defensive Block'],
        'bowler_type': ['fast', 'fast', 'spin'],
        'bowl_style': ['right_arm_fast_medium', 'right_arm_fast', 'right_arm_offspin'],
        'innings': [1, 2, 1],
        'balls_remaining': [108, 24, 114],
        'current_score': [45, 156, 12],
        'runs_required_if_innings2': [np.nan, 18, np.nan],
        'run_rate': [7.5, 9.2, 6.3],
        'required_run_rate': [np.nan, 11.5, np.nan],
        'wickets_left': [9, 3, 10],
        'batsman_runs': [23, 67, 5],
        'batsman_balls_faced': [18, 41, 8],
        'ball_line': ['outside_off', 'middle', 'off_stump'],
        'ball_length': ['full', 'short', 'good_length'],
        'shot_control': [1, 1, 0],
        'shot_outcome': ['4_runs', '6_runs', '0_runs'],
        'match_format': ['T20', 'T20', 'T20'],
        'shot_quality': ['Good', 'Good', 'Bad']
    })
    
    template.to_csv(output_path, index=False)
    print(f"✅ Annotation template saved to: {output_path}")


# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    print("="*70)
    print("CRICKET SHOT DATA TRANSFORMER - OPTIMIZED SCHEMA")
    print("="*70)
    
    # Generate annotation template
    print("\n1️⃣ Generating annotation template...")
    generate_annotation_template('annotation_template.csv')
    
    # Transform Himanish's data
    print("\n2️⃣ Transforming Himanish's dataset...")
    input_file = "odi.csv"  # Update this path
    output_file = "transformed_cricket_shots_optimized.csv"
    
    try:
        # Try with ALL match formats first (not just T20)
        df_transformed = transform_himanish_to_optimized_schema(
            input_file, 
            output_file,
            match_format='all',  # Changed from 'T20' to 'all'
            min_rows=100
        )
        
        if df_transformed is not None:
            print("\n✅ SUCCESS! Dataset ready for modeling.")
            print(f"\n📁 Files created:")
            print(f"   1. {output_file} - Transformed dataset ({len(df_transformed)} rows)")
            print(f"   2. annotation_template.csv - Template for manual annotation")
        else:
            print("\n❌ Transformation failed - check error messages above")
        
    except FileNotFoundError:
        print(f"\n❌ ERROR: Could not find '{input_file}'")
        print("Please update the 'input_file' variable with your actual file path.")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()