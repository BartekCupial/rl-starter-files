from mrunner.helpers.client_helper import get_configuration

from scripts.train import main

if __name__ == "__main__":
    config = get_configuration(print_diagnostics=True, with_neptune=False)

    del config["experiment_id"]
    main(config)
