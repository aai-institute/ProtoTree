import argparse
from pathlib import Path

from util.args import save_args


# TODO: won't work on windows, either fix or remove entirely
class Log:

    """
    Object for managing the log directory
    """

    def __init__(self, log_dir: str):  # Store log in log_dir
        log_dir = Path(log_dir)
        self._log_dir = log_dir
        self._logs = dict()

        # Ensure the directories exist
        for required_dir in [self.log_dir, self.checkpoint_dir, self.metadata_dir]:
            required_dir.mkdir(parents=True, exist_ok=True)

        # TODO: the fuck is going on here?
        # make log file empty if it already exists
        open(self.log_dir / "log.txt", "w").close()

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def checkpoint_dir(self):
        return self._log_dir / "checkpoints"

    @property
    def metadata_dir(self):
        return self._log_dir / "metadata"

    # TODO: fix this, my fucking god. We open and close a file at every log!!!
    def log_message(self, msg: str):
        """
        Write a message to the log file
        :param msg: the message string to be written to the log file
        """
        print(msg)
        with open(self.log_dir / "log.txt", "a") as f:
            f.write(msg + "\n")

    def create_log(self, log_name: str, key_name: str, *value_names):
        """
        Create a csv for logging information
        :param log_name: The name of the log. The log filename will be <log_name>.csv.
        :param key_name: The name of the attribute that is used as key (e.g. epoch number)
        :param value_names: The names of the attributes that are logged
        """
        if log_name in self._logs:
            raise KeyError(f"Entry {log_name} already exists!")
        self._logs[log_name] = (key_name, value_names)
        # Create log file. Create columns
        with open(self.log_dir / f"{log_name}.csv", "w") as f:
            f.write(",".join((key_name,) + value_names) + "\n")

    def log_values(self, log_name, key, *values):
        """
        Log values in an existent log file
        :param log_name: The name of the log file
        :param key: The key attribute for logging these values
        :param values: value attributes that will be stored in the log
        """
        if log_name not in self._logs:
            raise KeyError(f"No log with name {log_name} exists!")
        if len(values) != len(self._logs[log_name][1]):
            raise Exception(f"Not all required values for {log_name} are logged!")
        # Write a new line with the given values
        with open(self.log_dir / f"{log_name}.csv", "a") as f:
            f.write(",".join(str(v) for v in (key,) + values) + "\n")

    def log_args(self, args: argparse.Namespace):
        save_args(args, self._log_dir)
