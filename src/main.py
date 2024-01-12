import argparse

from engine import Engine
from params import Params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()

    params = Params.model_validate_json(open(args.config).read())
    engine = Engine(params)
    while True:
        engine.step()
