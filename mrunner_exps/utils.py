from scripts.train import parse_args


def combine_config_with_defaults(config):
    res = vars(parse_args())
    res.update(config)
    return res