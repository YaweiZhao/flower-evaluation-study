import argparse
import json

def get_param():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config', type=str, help='config path')
    
    args = parser.parse_args()
    config_path = args.config
    config_param = open(config_path, "r")
    param_str = config_param.read()
    param = json.loads(param_str)
    
    return param