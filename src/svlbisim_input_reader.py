import yaml

def load_yaml(inputsfile):
    with open(inputsfile) as f:
        dict = yaml.load(f, Loader=yaml.FullLoader)
    f.close()

    return dict

