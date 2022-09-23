from step01_prepare_lidc import create_or_load_dataset

create_or_load_dataset(load=False, save=True, annotation_size_perc=0.1, file_name='lidc_processed_10p')
