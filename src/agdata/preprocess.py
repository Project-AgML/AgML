import os
import csv
from utils import get_filelist, get_dirlist, read_txt_file
from utils import convert_txt_to_cocojson, get_label2id,create_dir

from shutil import copyfile
from tqdm import tqdm

class PreprocessData:

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_original_dir = os.path.join(self.data_dir,'original')
        self.data_processed_dir = os.path.join(self.data_dir,'processed')

    def preprocess(self, dataset_name):

        if dataset_name == 'bean_disease_uganda':
            None
        
        if dataset_name == 'carrot_weeds_germany':
            None

        if dataset_name == 'carrot_weeds_macedonia':
            None

        if dataset_name == 'rangeland_weeds_australia':
            dataset_dir = self.data_original_dir + dataset_name + '/'
            imgs_dir = dataset_dir + 'images/'
            labels_dir = dataset_dir + 'labels/'
            labels_path = labels_dir + 'labels.csv'
            labels_unique = []

            # Make directories with class names
            with open(labels_path) as f:
                next(f)
                labels = [row.split(',')[2] for row in f]

            with open(labels_path) as f:
                next(f)
                img_names = [row.split(',')[0].strip().replace(' ', '_') for row in f]

            # Read through list, keep only unique classes, and create directories for each class name
            for k, label in enumerate(labels):
                if label not in labels_unique:
                    labels_unique.append(label)
                    os.mkdir(labels_dir + label)
                os.rename(imgs_dir + img_names[k], labels_dir + label + '/' + img_names[k])

        if dataset_name == 'fruits_classification_worldwide':
            dataset_dir = os.path.join(self.data_original_dir, dataset_name,'datasets')

            # get folder list
            dataset_folders = get_dirlist(dataset_dir)
            label2id = get_label2id(dataset_folders)
            anno_data_all = []
            for folder in dataset_folders:
                annotations = ['test_RGB.txt', 'train_RGB.txt']
                dataset_path = os.path.join(dataset_dir,folder)
                # @TODO: Make separate json files for train and test?
                for anno_file_name in annotations:
                    # get img folder name
                    name = anno_file_name.split('.')[0].upper()

                    # Read annotations
                    try:
                        anno_data = read_txt_file(os.path.join(dataset_path,anno_file_name))
                    except:
                        try:
                            anno_data = read_txt_file(os.path.join(dataset_path,anno_file_name + '.txt'))
                        except:
                            raise

                    # Concat fruit name at head of line
                    for i, anno in enumerate(anno_data):
                        # Change to test path if the text file is test
                        if "test" in anno_file_name and "TRAIN" in anno[0]:
                            anno_data[i][0] = anno[0].replace("TRAIN", "TEST")
                        anno_data[i][0] = os.path.join(dataset_path, anno_data[i][0])

                    anno_data_all += anno_data

            # process files
            save_dir_anno = os.path.join(self.data_processed_dir,dataset_name,'annotations')
            create_dir(save_dir_anno)
            output_json_file = os.path.join(save_dir_anno,'train.json')
            convert_txt_to_cocojson(
                anno_data_all, label2id, output_json_file, False)

            # process img files. it can be replaced as opencv imread->imwrite function
            save_dir_imgs = os.path.join(self.data_processed_dir,dataset_name,'images')
            create_dir(save_dir_imgs)
            for anno in tqdm(anno_data_all):
                img_name = anno[0].split('/')[-1]
                dest_path = os.path.join(save_dir_imgs,img_name)
                try:
                    copyfile(anno[0],dest_path)
                except:
                    # Cannot copy the image file
                    pass
                
# Main function for debugging
if __name__ == '__main__':

    ppdata = PreprocessData(data_dir='/mnt/nas/work/AgData_Datasets')
    ppdata.preprocess(dataset_name='fruits_classification_worldwide')
