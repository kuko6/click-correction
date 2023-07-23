#! bin/bash

# Reorder files
python preprocessing/TCIA_data_convert_into_convenient_folder_structure.py --input data/Vestibular-Schwannoma-SEG --output data/VS-dicom/

# Add missing contours and registration matrices into the scan folders
python add_files.py