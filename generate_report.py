import pandas as pd
from modules.helpers import create_tree_dir, save_report, get_final_report


#ROOT = 'APPLIANCE_Window_Study'
ROOT = 'NFED_ENTROPY_POOL'
# ROOT = 'window_study'
dev_list = [
#                        'television',
#                        'computer',
                       'washing machine',
#                        'kettle',
                        # 'dish washer',
                       'fridge',
                        'microwave',
]
mod_list = [
     # 'VIB_SAED',
     # 'VIBWGRU',
     # 'VIBFNET',
     # 'VIBShortFNET',
     # 'VIBSeq2Point',
     'FNET',
# 'ShortFNET',
#      'S2P',
#      'SimpleGru',
#      'SAED',
#      'WGRU',
]

cat_list = [x for x in ['Single']]
tree_levels = {'root': ROOT, 'l1': ['results'], 'l2': dev_list, 'l3': mod_list, 'experiments': cat_list}

report = get_final_report(tree_levels, save=True, root_dir=ROOT, save_name='entropy_pool_single')

