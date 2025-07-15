from ast import FormattedValue
import time
import os
import logging

class TimeRecord(object):
    def __init__(self, model_name: str, sid) -> None:
        super(TimeRecord, self).__init__()
        self.epoch_: list[tuple[int, int]] = []
        self.temporal_: list[tuple[int, int]] = []

        self.entire: int = 0
        self.epoch: int = 0
        self.temporal: int = 0

        self.special_id = sid

        self.path_prefix = os.path.join(os.getcwd(), f"logs/{sid}/", "timelog/")
        self.score_prefix = os.path.join(os.getcwd(), f"logs/{sid}/", "scorelog/")
        self.model_name = model_name
    
    def set_up_logger(self, name: str = "score_logger"):
        # Determine the file path based on the logger name
        file_path = self.score_prefix if name == "score_logger" else self.path_prefix

        # Create the directory if it doesn't exist
        os.makedirs(file_path, exist_ok=True)

        # Generate current timestamp for the log file name
        current_time = time.strftime("%d-%H-%M", time.localtime())
        log_file_name = f"{self.model_name}_{self.dataset_name}_{current_time}.log"
        log_file_path = os.path.join(file_path, log_file_name)

        # Create or retrieve the logger
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        # Create a file handler for logging
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)

        logging.basicConfig(
            format="%(message)s",
            level=logging.INFO,
            handlers=[file_handler],
        )

        # Define the log message format
        # formatter = logging.Formatter("%(message)s")
        # file_handler.setFormatter(formatter)

        if logger.hasHandlers():
            logger.handlers.clear()

        # Add the file handler to the logger if it doesn't already have one
        if not logger.handlers:
            logger.addHandler(file_handler)

        # Store the logger reference if needed
        if name == "score_logger":
            self.score_log_handler = logger
        else:
            self.log_handler = logger
        

    def get_dataset(self, dataset_name: str):
        self.dataset_name = dataset_name

    def record_start(self):
        self.entire = time.time()

    def record_end(self):
        self.entire = time.time() - self.entire
    
    def temporal_record(self):
        self.temporal = time.time()

    def temporal_end(self, temporal_node: int):
        self.temporal = time.time() - self.temporal
        self.temporal_.append((self.temporal, temporal_node))

    def epoch_record(self):
        self.epoch = time.time()
    
    def epoch_end(self, batch_size: int):
        self.epoch = time.time() - self.epoch
        self.epoch_.append((self.epoch, batch_size))
    
    def score_statement(self, single_: dict[str, float], prefix: str = ""):
        output = []
        for idx, (key, value) in enumerate(single_.items()):
            if (idx + 1) % 4 == 0:
                each_item = f"{key:<10}: {round(value, 4):<8}\n"
            else:
                each_item = f"{key:<10}: {round(value, 4):<8}"
            output.append(each_item)
        return prefix + " | ".join(output)
    
    def score_record(self, temporal_score_: list[dict[list]], node_size: int, temporal_idx: int):
        """
        Attention, socre logger every time will record log info of a temporal result, there is no
        entire score result considering temporal is already the largest unit.

        :param temporal_score_: list[dict[Union[[str, float] | [str, list[float]]]]]
        """
        max_val = max(temporal_score_, key=lambda x: x["val_acc"])
        max_test = max(temporal_score_, key=lambda x: x["test_acc"])

        val_prefix = f"-------------Node size {node_size} | This Temporal {temporal_idx} Highest Val Acc Score Reuslt-------------\n"
        test_prefix = f"-------------Node size {node_size} | This Temporal {temporal_idx} Highest Test Acc Score Reuslt-------------\n"

        self.score_log_handler.info(self.score_statement(max_val, val_prefix))
        self.score_log_handler.info(self.score_statement(max_test, test_prefix))

        # for idx, epoch_eval_ in enumerate(temporal_score_):
        #     # get the max value
        #     self.score_log_handler.info(f"-------------Epoch {idx*5} Score Reuslt-------------\n")
        #     self.score_log_handler.info(self.score_statement(epoch_eval_))

    def record_statement(self, given: dict, prefix: str = ""):
        output = [f"{round(value), 8}" for _, value in given.items()]
        return prefix + " ".join(output)

    def fast_processing(self, entire_score: list[list[dict[str, float]]]):
        largest: list[dict[str, int]] = []
        for tidx, temporal_score in enumerate(entire_score):
            max_val = max(temporal_score, key=lambda x: x["val_acc"])
            max_test = max(temporal_score, key=lambda x: x["test_acc"])
            if len(largest)<2:
                largest+=[max_val, max_test]
            else:
                p1, p2 = largest[:tidx*2], largest[tidx*2:]
                largest = p1+ [max_val, max_test] + p2
            largest += temporal_score

        for score in largest:
            self.log_handler.info(self.record_statement(score)) 

    def to_log(self):
        self.log_handler.info("------------Entire Time-------------")
        self.log_handler.info(f"Entire Time: {self.entire}")

        self.log_handler.info("------------Temporal Time-------------")
        for idx, (temporal_time, temporal_node) in enumerate(self.temporal_):
            self.log_handler.info(f"Temporal Idx {idx} | Temporal node_size {temporal_node} | Temporal Time {temporal_time}")

        self.log_handler.info("------------Epoch Time-------------")
        for epoch_idx, (epoch_time, batch_size) in enumerate(self.epoch_):
            self.log_handler.info(f"Epoch Idx {epoch_idx} | Batch Size {batch_size} | Epoch Time {epoch_time}")


if __name__ == "__main__":
    test_= {"test_acc": 0.85,
            "val_acc": 0.75,
            "train_acc": 0.95,
            "accuacy": 0.85,
            "precision": 0.85,
            "recall": 0.85,
            "f1": 0.85,
            "micro_prec": 0.85,
            "micro_recall": 0.85,
            "micro_f1": 0.85
            }
    # score_statmenet(test_)