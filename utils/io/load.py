"""
load.py
"""

from pathlib import Path
import toml, yaml, pickle, json
from easydict import EasyDict
import re

def toml_load(path: Path):
    with open(path, "r") as f:
        cfg = toml.load(f)
    cfg = EasyDict(cfg)
    return cfg

def yaml_load(path: Path):
    with open(path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    cfg = EasyDict(cfg)
    return cfg

def pickle_load(path: Path, encoding: str="ASCII"):
    with open(path, "rb") as f:
        cfg = pickle.load(f, encoding=encoding)
    return cfg

def json_load(path: Path):
    with open(path, "r") as f:
        cfg = json.loads(f.read())
    cfg = EasyDict(cfg)
    return cfg

def get_project_root():
    markers = ['requirements.txt']
    current = Path(__file__).resolve().parent
    
    for parent in [current] + list(current.parents):
        if any((parent / marker).exists() for marker in markers):
            return parent
            
    return current

def standardize_file_name(filename):
    replaced = re.sub(r"[ \-,，\-【】（）]", "_", filename)
    replaced = re.sub(r"_+", "_", replaced)
    replaced = replaced.strip("_")
    return replaced