import pdb
import src
import glob
import importlib.util
import os
import cv2
import numpy as np

np.random.seed(0)


focal_pixels = {"I1": (16.0 * 3264) / 5.74,
                "I2": (5.55 * 653) / 5.74,
                "I3": (5.725 * 730)/ 6.17,
                "I4": (25.5 * 2000) / 23.55,
                "I5": (24.0 * 2000) / 23.55,
                "I6": (30.0 * 602)/ 23.7}


### Change path to images here
# path = 'Images{}*'.format(os.sep)  # Use os.sep, Windows, linux have different path delimiters
path = 'Images{}*'.format(os.sep)  # Use os.sep, Windows, linux have different path delimiters

###

all_submissions = glob.glob('./src/*')
os.makedirs('./results/', exist_ok=True)
for idx,algo in enumerate(all_submissions):
    print('****************\tRunning Awesome Stitcher developed by: {}  | {} of {}\t********************'.format(algo.split(os.sep)[-1],idx,len(all_submissions)))
    try:
        module_name = '{}_{}'.format(algo.split(os.sep)[-1],'stitcher')
        filepath = '{}{}stitcher.py'.format( algo,os.sep,'stitcher.py')
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        PanaromaStitcher = getattr(module, 'PanaromaStitcher')
        inst = PanaromaStitcher()

        ###
        for impaths in glob.glob(path):
            print('\t\t Processing... {}'.format(impaths))
            folder = impaths.split(os.sep)[-1]
            focal = focal_pixels[folder]
            stitched_images, homography_matrix_list = inst.make_panaroma_for_images_in(path=impaths,f=focal)

            outfile =  './results/{}/{}.png'.format(impaths.split(os.sep)[-1],spec.name)
            os.makedirs(os.path.dirname(outfile),exist_ok=True)
            for i in range(len(stitched_images)):
                cv2.imwrite('./intermediate/{}/{}.png'.format(impaths.split(os.sep)[-1],"stitched_image_"+str(i)),stitched_images[i])
            cv2.imwrite(outfile,stitched_images[-1])
            print(homography_matrix_list)
            print('Panaroma saved ... @ ./results/{}.png'.format(spec.name))
            print('\n\n')

    except Exception as e:
        print('Oh No! My implementation encountered this issue\n\t{}'.format(e))
        print('\n\n')
