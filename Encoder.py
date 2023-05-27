from sklearn.preprocessing import OrdinalEncoder

def encode_categorical(df, columns):
    
    # Create a dictionary to map short-form weekday names to full names
    weekday_map = {
        'Mon': 'Monday',
        'Tue': 'Tuesday',
        'Wed': 'Wednesday',
        'Thu': 'Thursday',
        'Fri': 'Friday',
        'Sat': 'Saturday',
        'Sun': 'Sunday'
    }
    
    for col in columns:
        if df[col].dtype == 'object' and col == 'flight_day':
            # Replace short-form names with full names using the weekday_map dictionary
            df[col] = df[col].apply(lambda x: weekday_map[x] if x in weekday_map else x)
            
            # Use the full list of weekdays as categories for the OrdinalEncoder
            categories = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            oe = OrdinalEncoder(categories=[categories])
            df[col] = oe.fit_transform(df[[col]])
        elif df[col].dtype == 'object':
            df[col] = df[col].astype('category').cat.codes
    return df