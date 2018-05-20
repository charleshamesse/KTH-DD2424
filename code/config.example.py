DATA = {
    'dataset_name': 'cifar10',
    'data_dir': '../dataset',
    'sample_dir': '../dataset/generated/',
    'results_dir': 'results',
}

'''
DATA = {
    'dataset_name': 'photos_108',
    'data_dir': '../dataset',
    'sample_dir': '../dataset/generated/',
    'results_dir': 'results',
}
'''

MODEL = {
    'model_dir': 'models',
    'checkpoint_dir': 'checkpoints',
    'batch_size': 8, # 64
    'epochs': 5, # 1000
    'data_limit': 40,# None
    'sample_num': 9,
    'with_sn': True,
    'with_wasserstein': True,
}