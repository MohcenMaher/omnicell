import json
import glob
import argparse
import os
from omnicell.data.catalogue import Catalogue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()

    dd = Catalogue.get_dataset_details(args.dataset)
    if dd.precomputed_DEGs:
        print("Precomputed DEGs found - Terminating")
        exit()

    combined_results = {}

    # Find all partial result files
    part_files = glob.glob(f'{dd.folder_path}/DEGs-*_of_*.json')
    if not part_files:
        raise ValueError(f"No partial results found in {dd.folder_path}")



    first_file = os.path.basename(part_files[0])
    total_parts = int(first_file.split('_of_')[1].split('.json')[0])
    expected_files = {f'{dd.folder_path}/DEGs-{i}_of_{total_parts}.json' for i in range(0, total_parts)}
    missing_files = expected_files - set(part_files)
    if missing_files:
        raise ValueError(f"Missing part files: {missing_files}")

    # Combine partial results
    for filename in part_files:
        with open(filename, 'r') as f:
            partial_results = json.load(f)
            for cell_type, pert_dict in partial_results.items():
                if cell_type not in combined_results:
                    combined_results[cell_type] = {}
                combined_results[cell_type].update(pert_dict)

    
    print(f"Combined {len(part_files)} partial results into {len(combined_results)} cell types")
    # Save combined results
    output_file = f'{dd.folder_path}/DEGs.json'
    with open(output_file, 'w') as f:
        json.dump(combined_results, f)
    
    print(f"Combined results saved to {output_file}")

    # Remove partial results
    for filename in part_files:
        os.remove(filename)

    print(f"Removed {len(part_files)} partial result files")
    print("Done")