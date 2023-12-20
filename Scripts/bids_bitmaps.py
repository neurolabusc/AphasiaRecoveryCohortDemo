#!/usr/bin/python

# Create bitmaps to check defacing of all NIFTI images
#  python bids_deface_bitmaps.py

from pathlib import Path
import os
import os.path
import glob
import sys
import shutil
import fnmatch

def remove_after_final_underscore(input_string):
    # Find the last underscore index
    last_underscore_index = input_string.rfind('_')

    # If an underscore is found, remove characters after it
    if last_underscore_index != -1:
        result_string = input_string[:last_underscore_index]
        return result_string
    else:
        # If no underscore is found, return the original string
        return input_string

if __name__ == "__main__":
    inputRoot = './'
    outRoot = './bmps/'
    exe = '/Applications/MRIcroGL.app/Contents/MacOS/MRIcroGL'
    if not shutil.which(exe):
        sys.exit("Unable to find exe: "+exe)

    nifti_files = [f for f in os.listdir(inputRoot) if fnmatch.fnmatch(f, 'wsub-M4209*.nii.gz')]
    for subname in list(nifti_files):
        sub = Path(subname).stem
        les = os.path.join(inputRoot, subname)
        #wbsub-M2185_ses-388_T1w.nii.gz
        T1 = remove_after_final_underscore(sub)
        bmp = os.path.join(outRoot, T1 + ".png")
        T1 = T1[:1] + 'b' + T1[1:] + '_T1w.nii.gz'
        T1 = os.path.join(inputRoot, T1);
        if not os.path.isfile(T1):
            print('oops: '+ T1);
            continue
        #print(T1) gl.overlayload
        cmd = exe + ' -s \'import gl\ngl.loadimage("' + T1 + '")\ngl.drawload("' + les + '")\ngl.mosaic("A 0.5 S 0.25 C 0.5 S R -0")\ngl.colorbarposition(0)\ngl.savebmp("' + bmp + '")\ngl.quit()\'\n'
        os.system(cmd)
        #sys.exit()
