from cypath.data.evalModelResults import *
import yaml
import sys

def oneSetTest(results_dir, result_suffix, annos_dir, anno_suffix, excludes_dir):
    if exclude_dir==None:
        res = evalModelBoxResultsWithDir(results_dir, result_suffix, annos_dir, anno_suffix)
    else:
        res = evalModelBoxResultsWithDirWithExclude(results_dir, result_suffix, annos_dir, anno_suffix, excludes_dir)
    return avgArrDictResults(res)

if __name__ == '__main__':
    yaml_path = sys.argv[1]
    with open(yaml_path, 'r') as f:
        param_sets = yaml.safe_load(f)

    for set_name, param in param_sets.items():
        anno_dir = param.get('anno_dir')
        result_dir = param.get('result_dir')
        exclude_dir = param.get('exclude_dir')
        anno_suffix = param.get('anno_suffix')
        result_suffix = param.get('result_suffix')
        exclude_suffix = param.get('exclude_suffix')
        print(set_name)
        print(oneSetTest(result_dir, result_suffix, anno_dir, anno_suffix, exclude_dir))
        print()
