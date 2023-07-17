import yaml
def get_conf(conf_path):
    with open(conf_path, "r") as conf:
        cfg = yaml.load(conf, Loader=yaml.Loader)
    return cfg    
