import h5py
import numpy as np
import argparse
import os

def cut_dataset(input_dataset_path, output_dataset_path, num_demos_to_keep):
    with h5py.File(input_dataset_path, 'r') as input_file:
        # Open the input dataset
        data_group = input_file['data']
        
        # Get the demo keys from the dataset (assuming the demos are stored in groups under 'data')
        demo_keys = sorted(data_group.keys())
        
        # Keep only the first 'num_demos_to_keep' demos
        selected_demos = demo_keys[:num_demos_to_keep]
        
        with h5py.File(output_dataset_path, 'w') as output_file:
            output_group = output_file.create_group('data')
            
            # Copy the selected demos to the output file
            for demo_key in selected_demos:
                demo_data = data_group[demo_key]
                output_group.create_group(demo_key)
                # Copy all contents of the demo group
                for dataset_name, dataset in demo_data.items():
                    output_group[demo_key].create_dataset(dataset_name, data=dataset[()])
                    # Optionally, add compression if required
                    # output_group[demo_key].create_dataset(dataset_name, data=dataset[()], compression='gzip')
                    
            print(f"Dataset has been cut to {num_demos_to_keep} demos and saved to {output_dataset_path}")

def main():
    # Setting up argument parser
    parser = argparse.ArgumentParser(description="Cut an HDF5 dataset to a specific number of demos.")
    parser.add_argument('input', type=str, help='Path to the input HDF5 dataset')
    parser.add_argument('output', type=str, help='Path to the output HDF5 dataset')
    parser.add_argument('num_demos', type=int, help='Number of demos to keep in the output dataset')
    
    # Parsing arguments
    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.output_dataset_path), exist_ok=True)

    # Call the function to cut the dataset
    cut_dataset(args.input_dataset_path, args.output_dataset_path, args.num_demos_to_keep)

if __name__ == "__main__":
    main()