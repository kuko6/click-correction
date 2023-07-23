import os
import shutil

def main():
    scans_path = 'data/VS-dicom/'
    contours_path = 'data/contours/'
    matrices_path = 'data/registration_matrices/'

    scans = os.listdir(scans_path)

    print('Adding countours and matrices')
    for scan in scans:
        if scan == '.DS_Store': continue
        #print(scan, os.listdir(contours_path + scan), os.listdir(matrices_path + scan))
        print(scan)
        shutil.copy(contours_path + scan + '/contours.json', scans_path + scan)
        shutil.copy(matrices_path + scan + '/' + os.listdir(matrices_path + scan)[0], scans_path + scan)

if __name__ == '__main__':
    main()