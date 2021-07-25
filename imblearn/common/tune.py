#!/usr/bin/env python
# hyper parameter trial
# Author: Thamme Gowda [tg (at) isi (dot) edu]
# Created: 7/19/21

from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple, List, Any, Iterator, Callable, Union

from ray import tune
from ruamel.yaml import yaml_object

from imblearn import yaml


@yaml_object(yaml)
class TunableParam:
    yaml_tag = '!tune'

    def __init__(self, *args):
        assert len(args) > 1, 'at least one argument should be given'
        self.args = args
        func_name, *func_args = args
        assert getattr(tune, func_name, None), \
            f'{func_name} unknown; check if ray.tune.<{func_name}> is valid'
        assert callable(getattr(tune, func_name)), \
            f'{func_name} invalid; check if ray.tune.<{func_name}> is callable with args'
        self.param = getattr(tune, func_name)(*func_args)

    @classmethod
    def to_yaml(cls, representer, node):
        return representer.represent_sequence(cls.yaml_tag, node.args)

    @classmethod
    def from_yaml(cls, constructor, node):
        return cls(*constructor.construct_sequence(node, deep=True))

    def __repr__(self):
        return f'!tune.{self.args[0]}' + str(self.args[1:])


def find_tunable_params(node: Union[Dict, List], path=None) -> Iterator[Tuple[Tuple, TunableParam]]:
    """
    Finds tunable params
    :param node: node in conf.
    :param path: path from root; for root node, path is None (default)
    :return:
    """
    if path is None:
        path = []
    if isinstance(node, dict):
        for key, obj in node.items():
            yield from find_tunable_params(obj, path=path + [key])
    elif isinstance(node, list):
        for key, obj in enumerate(node):
            yield from find_tunable_params(obj, path=path + [key])
    else:
        if isinstance(node, TunableParam):
            yield tuple(path), node
        # else ignore

def update_conf_values(conf: Dict, named_values: Mapping[str, Any],
                       name_to_path: Callable[[str], Sequence[str]]):
    """
    Update conf object with values given here
    :param conf: dictionary that needs to be updated
    :param named_values: values to be written into conf
    :param name_to_path: function that maps name to path inside config
    Paths are tuple of segments to identify a node in nested dict and lists, where each segment is
     either key of dict or index in list.
    :return: conf is updated inplace and returned
    """

    for name, value in named_values.items():
        node = conf
        path = name_to_path()
        for segment in path[:-1]:
            assert segment in conf, f'{segment} from {path} not found in conf'
            node = node[segment]
        node[path[-1]] = value
    return conf


if __name__ == '__main__':
    p = Path('/Users/tg/work/papers/011-imb-learn/experiments/txtcls.conf.tune.yml')
    conf = dict(yaml.load(p))
    print(conf)
    # s = StringIO()
    # yaml.dump(conf, s)
    # print(s.getvalue())
    for path, node in find_tunable_params(conf):
        print(path, node)
