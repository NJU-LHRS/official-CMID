import argparse as ag
import json

def get_parser_with_args(metadata_json=r'F:\pumpkinCode\Change Detection\metadata.json'):
    parser = ag.ArgumentParser(description='Training change detection network')

    with open(metadata_json, 'r') as fin:
        metadata = json.load(fin)
        parser.set_defaults(**metadata)
        

    parser.add_argument('--backbone', default=None, type=str, choices=['resnet','swin','vitae', "swin_base"], help='type of model')

    parser.add_argument('--dataset', default=None, type=str, choices=['cdd','levir'], help='type of dataset')

    parser.add_argument('--mode', default=None, type=str, choices=['imp','proposed', 'sen12ms', 'mocov2' , 'byol', 'swav', 'seco', "barlowtwins", "swin_base"], help='type of pretrn')

    parser.add_argument('--path', default=None, type=str, help='path of saved model')


    return parser, metadata

