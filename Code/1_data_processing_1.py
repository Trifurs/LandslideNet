import sys
import xml.etree.ElementTree as ET
from utils import *


def get_argv(xml_file):
    """Extract preprocessing parameters from XML config"""
    argv_names = [
        'input_factors_dir', 'input_labels_dir',
        'output_factors_dir', 'output_labels_dir',
        'crop_size', 'overlap'
    ]
    argv_values = []
    root = ET.parse(xml_file).getroot()
    
    for argv_name in argv_names:
        for param in root.findall('param'):
            if param.find('name').text == argv_name:
                argv_values.append(param.find('value').text)
                break
        else:
            raise ValueError(f"Parameter {argv_name} not found")
    return argv_values


def main(input_factors_dir, input_labels_dir, output_factors_dir, 
         output_labels_dir, crop_size, overlap):
    """Main landslide data preprocessing workflow"""
    # Convert numeric parameters
    crop_size = int(crop_size)
    overlap = int(overlap)

    # Execute raster cropping
    crop_all_rasters(input_factors_dir, output_factors_dir, crop_size, overlap)
    crop_all_rasters(input_labels_dir, output_labels_dir, crop_size, overlap)

    # Remove invalid data
    move_black_images_in_all_subfolders(output_factors_dir)
    move_black_images_in_all_subfolders(output_labels_dir)

    # Ensure data consistency
    move_missing_images_to_black(output_factors_dir)
    move_missing_images_to_black(output_labels_dir)


if __name__ == '__main__':
    try:
        # Parameter validation
        if len(sys.argv) < 2:
            raise RuntimeError("Missing XML config file parameter")
            
        config_path = sys.argv[1]
        params = get_argv(config_path)
        
        if len(params) != 6:
            raise ValueError("Mismatched configuration parameters count")
            
        # Execute main workflow
        main(*params)
        
        # Output success status
        print('<process_status>0</process_status>')
        print('<process_log>success</process_log>')
        
    except Exception as e:
        # Error handling
        error_msg = str(e).replace('\n', ' ').replace('\t', ' ')
        print('<process_status>1</process_status>')
        print(f'<process_log>{error_msg}</process_log>')
