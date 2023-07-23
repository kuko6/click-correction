#! bin/bash

# Run the data converter
i=1
for d in ./data/VS-dicom/*/; do
    echo "$d"
    mv $d ./data/VS-dicom-tmp
    if [ $(( $i % 2 )) -eq 0 ]; then 
        echo "Importing data"
        echo "--------------"
        
        ~/Applications/Slicer.app/Contents/MacOS/Slicer --no-splash --no-main-window --python-script preprocessing/data_conversion.py --input-folder ~/Developer/School/DP/data/VS-dicom-tmp/ --output-folder ~/Developer/School/DP/data/VS-nii/ --register T2
        #~/Applications/Slicer.app/Contents/MacOS/Slicer --no-splash --no-main-window --python-script preprocessing/data_conversion.py --input-folder ~/Developer/School/DP/data/VS-dicom-tmp/ --output-folder ~/Developer/School/DP/data/VS-nii/ --register T2 --export_all_structures
       
        mv ./data/VS-dicom-tmp/* ./data/VS-dicom

        # if [ $(( $i % 4 )) -eq 0 ]; then 
        #     break
        # fi
    fi
    (( i += 1 ))
done

echo "---------------------------------------"
OUTPUT=$(ls -1 ./data/VS-nii | wc -l)
echo "Succesfully converted: ${OUTPUT} files"
