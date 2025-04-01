import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import os
import re
from datetime import datetime
from sklearn.cluster import KMeans

def select_file():
    """
    Open a file dialog to select a CSV file
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title="Select CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not file_path:
        root.destroy()
        return None
    
    # Ask for machine number
    machine_number = simpledialog.askstring(
        "Machine Number", 
        "Please enter the machine number (e.g., 04):",
        parent=root
    )
    
    # Extract capacity from filename
    filename = os.path.basename(file_path)
    capacity_str, capacity_num = extract_capacity_from_filename(filename)
    
    # Use a simpler approach with messagebox for capacity confirmation
    default_capacity = "340" if capacity_num not in [340, 680] else str(capacity_num)
    message = f"Detected capacity: {capacity_str}\nIs this correct?"
    
    if capacity_num in [340, 680]:
        if not messagebox.askyesno("Confirm Capacity", message):
            # If not correct, ask user to choose
            if capacity_num == 340:
                new_capacity = "680"
            else:
                new_capacity = "340"
                
            if messagebox.askyesno("Select Capacity", f"Use {new_capacity}K instead?"):
                capacity_num = int(new_capacity)
                capacity_str = f"{capacity_num}k"
    else:
        # If capacity not detected, ask user to choose
        if messagebox.askyesno("Select Capacity", "Use 340K?"):
            capacity_num = 340
            capacity_str = "340k"
        else:
            capacity_num = 680
            capacity_str = "680k"
    
    root.destroy()
    return file_path, machine_number, capacity_str, capacity_num

def extract_capacity_from_filename(filename):
    """
    Extract capacity information from filename (e.g., 340K, 680K)
    """
    # Look for patterns like 340K, 680K in the filename
    match = re.search(r'(\d+)[Kk]', filename)
    if match:
        capacity = match.group(1)
        return capacity + "k", int(capacity)
    return "unknown", 0

def process_drops_data(file_path):
    """
    Process the drops data from the CSV file
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    # Import the CSV file, skip the header line
    try:
        data = pd.read_csv(file_path, skiprows=1)
        # Rename columns if they don't have headers
        if 'Label' not in data.columns:
            data.columns = ['Index', 'Label', 'X', 'Y']
        
        # Remove duplicates based on Label, X, Y
        data = data.drop_duplicates(subset=['Label', 'X', 'Y'])
        
        # Group data by Label
        grouped_data = {}
        labels = ["A.tif", "T.tif", "C.tif", "G.tif", "ACT.tif"]
        
        for label in labels:
            label_data = data[data['Label'] == label][['Label', 'X', 'Y']]
            if not label_data.empty:
                grouped_data[label] = label_data
            else:
                print(f"No data found for label: {label}")
        
        return grouped_data
    
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

def rotate_to_vertical(df):
    """
    Rotate the data to make the drop pattern vertical
    Equivalent to rotateToVertical in Mathematica
    """
    if df.empty or len(df) < 2:
        return df
    
    # Sort by X coordinate
    sorted_df = df.sort_values(by='X')
    
    # Calculate the vector from first to last point
    first_point = sorted_df.iloc[0][['X', 'Y']].values
    last_point = sorted_df.iloc[-1][['X', 'Y']].values
    vec = last_point - first_point
    
    # Calculate the angle
    angle = np.arctan2(vec[1], vec[0])
    
    # Create rotation matrix
    rotation_matrix = np.array([
        [np.cos(-angle), -np.sin(-angle)],
        [np.sin(-angle), np.cos(-angle)]
    ])
    
    # Extract X, Y coordinates
    coords = df[['X', 'Y']].values
    
    # Apply rotation
    rotated_coords = np.dot(coords, rotation_matrix.T)
    
    # Create new dataframe with rotated coordinates
    new_df = pd.DataFrame({
        'X': rotated_coords[:, 0],
        'Y': rotated_coords[:, 1]
    })
    
    return new_df

def find_nozzle_340k(values, sn):
    """
    Calculate nozzle positions for 340K data
    Equivalent to findNozzle in Mathematica for 340K
    """
    # Convert to list if it's a numpy array
    if isinstance(values, np.ndarray):
        values = values.tolist()
    
    # Check if values is empty or has less than 2 elements
    if not values or len(values) < 2:
        return []
    
    # Sort the list
    sorted_list = sorted(values)
    start = sorted_list[0]
    end = sorted_list[-1]
    mid = sorted_list[1:-1] if len(sorted_list) > 2 else []
    
    # Calculate step size
    step = (end - start) / (sn - 1)
    
    # Function to approximate nozzle position
    def appro(x):
        if sn == 318:
            # For 318 nozzles
            pos = ((x - start) / step + 1)
            if abs(round(pos) - pos) < 0.25:
                return int(round(pos))
            else:
                return "out of range"
        else:
            # For 317 nozzles
            pos = ((x - start) / step + 1)
            if abs(round(pos) - pos) < 0.25:
                return int(round(pos) + 318)
            else:
                return "out of range"
    
    # Apply the approximation to the middle values
    if not mid:
        return []
    
    return [appro(x) for x in mid]

def find_nozzle_680k(values, sn=636):
    """
    Calculate nozzle positions for 680K data
    Equivalent to findNozzle in Mathematica for 680K
    """
    # Convert to list if it's a numpy array
    if isinstance(values, np.ndarray):
        values = values.tolist()
    
    # Check if values is empty or has less than 2 elements
    if not values or len(values) < 2:
        return []
    
    # Sort the list
    sorted_list = sorted(values)
    start = sorted_list[0]
    end = sorted_list[-1]
    mid = sorted_list[1:-1] if len(sorted_list) > 2 else []
    
    # Calculate step size
    step = (end - start) / (sn - 1)
    
    # Function to approximate nozzle position
    def appro(x):
        pos = ((x - start) / step + 1)
        if abs(round(pos) - pos) < 0.25:
            return int(round(pos))
        else:
            return "out of range"
    
    # Apply the approximation to the middle values
    if not mid:
        return []
    
    return [appro(x) for x in mid]

def get_width(values):
    """
    Calculate the width of a list of values
    Equivalent to wide in Mathematica
    """
    if len(values) == 0:
        return 0
    return max(values) - min(values)

def get_sn(list1, list2):
    """
    Determine which list corresponds to 318 nozzles and which to 317
    Equivalent to getSn in Mathematica
    """
    if get_width(list1) > get_width(list2):
        return [318, 317]
    else:
        return [317, 318]

def auto_cal_flaw_340k(df):
    """
    Automatically calculate flaws in nozzles for 340K data
    Equivalent to autoCalFlaw in Mathematica for 340K
    """
    if df is None or df.empty or len(df) <= 4:
        return []
    
    try:
        # Rotate to vertical
        new_df = rotate_to_vertical(df)
        
        # Use K-means to separate into two clusters (straight line and slant line)
        y_values = new_df['Y'].values.reshape(-1, 1)
        
        # Check if we have enough distinct Y values for clustering
        unique_y = np.unique(y_values)
        if len(unique_y) < 2:
            print("Not enough distinct Y values for clustering")
            return []
        
        kmeans = KMeans(n_clusters=2, random_state=0).fit(y_values)
        cluster_centers = kmeans.cluster_centers_.flatten()
        div = np.mean(cluster_centers)
        
        # Group by cluster
        new_df['cluster'] = np.where(new_df['Y'] > div, 'a', 'b')
        
        # Check if we have both clusters
        if 'a' not in new_df['cluster'].values or 'b' not in new_df['cluster'].values:
            print("Could not separate into two distinct clusters")
            return []
        
        tf = {
            'a': new_df[new_df['cluster'] == 'a'].sort_values(by='X'),
            'b': new_df[new_df['cluster'] == 'b'].sort_values(by='X')
        }
        
        # Check if both clusters have data
        if tf['a'].empty or tf['b'].empty:
            print("One of the clusters is empty")
            return []
        
        # Determine which cluster is 318 and which is 317
        index = get_sn(tf['a']['X'].values, tf['b']['X'].values)
        
        # Find nozzles for each cluster
        nozzles_a = find_nozzle_340k(tf['a']['X'].values, index[0])
        nozzles_b = find_nozzle_340k(tf['b']['X'].values, index[1])
        
        # Combine results and filter out "out of range"
        result = [x for x in nozzles_a + nozzles_b if x != "out of range"]
        return result
    
    except Exception as e:
        print(f"Error in auto_cal_flaw_340k: {e}")
        return []

def auto_cal_flaw_680k(df):
    """
    Automatically calculate flaws in nozzles for 680K data
    Equivalent to autoCalFlaw in Mathematica for 680K
    """
    if df is None or df.empty:
        return []
    
    try:
        # Rotate to vertical
        new_df = rotate_to_vertical(df)
        
        # Find nozzles (fixed at 636 nozzles)
        result = find_nozzle_680k(new_df['X'].values, 636)
        
        # Filter out "out of range"
        result = [x for x in result if x != "out of range"]
        return result
    
    except Exception as e:
        print(f"Error in auto_cal_flaw_680k: {e}")
        return []

def main():
    print("Drop Nozzles Analysis Tool")
    print("==========================")
    
    # Select file using dialog and get machine number and capacity
    result = select_file()
    if not result:
        print("No file selected. Exiting.")
        return
    
    file_path, machine_number, capacity_str, capacity_num = result
    
    # Default machine number if user didn't provide one
    if not machine_number:
        machine_number = "00"
    
    print(f"Processing file: {file_path}")
    print(f"Machine number: {machine_number}")
    print(f"Selected capacity: {capacity_str}")
    
    # Process the data
    grouped_data = process_drops_data(file_path)
    
    if grouped_data:
        # Calculate flaws for each group based on capacity
        flaw_lists = []
        for label, data in grouped_data.items():
            print(f"Processing {label}...")
            
            # Choose processing method based on capacity
            if capacity_num == 340:
                print("Using 340K processing method")
                flaws = auto_cal_flaw_340k(data)
            elif capacity_num == 680:
                print("Using 680K processing method")
                flaws = auto_cal_flaw_680k(data)
            else:
                print(f"Unknown capacity: {capacity_num}, defaulting to 340K processing")
                flaws = auto_cal_flaw_340k(data)
            
            if flaws:
                flaw_lists.extend(flaws)
                print(f"Found {len(flaws)} potential flaws in {label}")
            else:
                print(f"No flaws found in {label}")
        
        # Remove duplicates and sort
        total_flaw_list = sorted(list(set([x for x in flaw_lists if x != "out of range"])))
        
        print("\nTotal Flaw List:")
        print(total_flaw_list)
        
        # Generate output filename in the format: flaw_nozzle_YYYYMMDD_capacity_machine.txt
        today = datetime.now().strftime("%Y%m%d")
        output_filename = f"flaw_nozzle_{today}_{capacity_str}_{machine_number}.txt"
        output_file = os.path.join(os.path.dirname(file_path), output_filename)
        
        # Save results to file - only include the list of flawed nozzle numbers
        with open(output_file, 'w') as f:
            f.write(", ".join(map(str, total_flaw_list)))
        
        print(f"\nResults saved to {output_file}")
    
if __name__ == "__main__":
    main()
