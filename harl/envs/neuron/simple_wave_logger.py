from harl.common.base_logger import BaseLogger


class SimpleWaveLogger(BaseLogger):
    def get_task_name(self):
        return self.env_args["scenario"]
