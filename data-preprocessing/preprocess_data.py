class PreprocessData:

    def __init__(self):
        self.data_dir = '../data/'
        self.data_original_dir = self.data_dir + 'original/'
        self.data_processed_dir = self.data_dir + 'processed/'
    
    def pp_bean_disease_uganda(self):
        self.dataset_name = 'bean_disease_uganda'
    
    def pp_carrot_weeds_germany(self):
        self.dataset_name = 'carrot_weeds_germany'

    def pp_carrot_weeds_macedonia(self):
        self.dataset_name = 'carrot_weeds_macedonia'