import subprocess
from typing import List
import shutil
import sys
import yaml
import os
from datetime import datetime
from itertools import product, chain
import time as time_module
import jinja2
import re
import multiprocessing
import logging
from fractions import Fraction

# Logging
logger = logging.getLogger(__name__)
try:
    if os.environ["LOGGING_LEVEL"].lower() == "debug":
        logger.setLevel(level=logging.DEBUG)
    elif os.environ["LOGGING_LEVEL"].lower() == "warning":
        logger.setLevel(level=logging.WARNING)
    elif os.environ["LOGGING_LEVEL"].lower() == "error":
        logger.setLevel(level=logging.ERROR)
    else:
        logger.setLevel(level=logging.INFO)
except KeyError:
    logger.setLevel(level=logging.INFO)
logger_handler = logging.StreamHandler()
logger.addHandler(logger_handler)
if logger.level == logging.DEBUG:
    logger_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
else:
    logger_handler.setFormatter(logging.Formatter("%(message)s"))

# Jinja2
template_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader("templates"),
    trim_blocks=True,
    lstrip_blocks=True,
)

CSV_DELIMITER = ";"


class Experiment:
    def __init__(self, config) -> None:
        self.config = config
        try:
            self.name = config["experiment_name"]
        except KeyError:
            self.name = "".join(self.config['config_seq'], "experiment")
        self.create_directory_path()
        self.save_configuration(os.path.abspath(os.path.join(self.dir, "configuration.yaml")))
        self.model_checking_results = None
        self.evochecker = None
        self.edit_config()

    def run(self):
        self.log_info("Started Experiment")
        self.check_preconditions()
        self.generate_model()  # may needs PRISM
        if self.config["check_steps_per_round"]:
            self.check_steps_per_round()  # needs PRISM
        if self.config["calculate_aggregated_failure_rate"]:
            self.calculate_aggregated_failure_rate()  # needs PRISM
        if self.config["calculate_baseline"]:
            self.calculate_baseline()  # needs PRISM
        if self.config["model_checking"]["run"]:
            self.run_model_checking()
        if self.config["evochecker"]["run"]:
            self.generate_evochecker_config()
            self.evochecker = self.run_evochecker(os.path.abspath(os.path.join(self.dir, "evochecker", "evochecker.properties")))
            shutil.copyfile(self.evochecker.pareto_front_file, os.path.join(self.dir, "results", "pareto_front"))
            shutil.copyfile(self.evochecker.pareto_set_file, os.path.join(self.dir, "results", "pareto_set"))
        self.log_info("Finished Experiment")

    def check_preconditions(self):
        if not self.config["service1"]["costs"] <= self.config["service2"]["costs"] <= self.config["service3"]["costs"]:
            print("costs must must be in order. s1 <= s2 <= s3")
            sys.exit(1)

    def check_steps_per_round(self):
        steps = self.config['steps_per_round']
        prism_computation_engine = self.config["model_checking"]["prism_computation_engine"]
        model = os.path.join(self.dir, "TAS.prism")
        self.log_debug("Checking steps per round")
        prism_result = get_prism_results(f"prism {model} -pf P>=1[G={2*steps-1}sync_s=6] -maxiters 100000 -{prism_computation_engine}")
        result = True if prism_result[0] == "true" else False
        if not result:
            self.log_error(f"precondition P>=1[G={2*steps-1}sync_s=6] is not true")
            sys.exit(1)

    def save_configuration(self, file_path):
        with open(file_path, 'w') as config_file:
            config_file.write(yaml.safe_dump(self.config))

    def generate_evochecker_config(self):
        if "model_template_file" not in self.config['evochecker']:
            self.config['evochecker'] = os.path.join(self.dir, "evochecker", "TAS_evochecker.prism")
        if "properties_file" not in self.config['evochecker']:
            self.config['evochecker'] = os.path.join(self.dir, "evochecker", "TAS_evochecker.props")
        self.generate_file_from_template("evochecker.properties", os.path.join("evochecker", "evochecker.properties"))

    def generate_model(self):
        self.generate_file_from_template("TAS.prism")
        self.generate_file_from_template("TAS.props")
        self.config['evochecker_template'] = True
        self.generate_file_from_template("TAS.prism", os.path.join("evochecker", "TAS_evochecker.prism"))
        self.generate_file_from_template("TAS.props", os.path.join("evochecker", "TAS_evochecker.props"))
        self.config['evochecker_template'] = False
        self.config['compute_baseline'] = True
        self.generate_file_from_template("TAS.prism", os.path.join("baseline", "TAS_baseline.prism"))
        self.generate_file_from_template("TAS.props", os.path.join("baseline", "TAS_baseline.props"))
        self.config['compute_baseline'] = False

    def generate_file_from_template(self, filename, target_filename=""):
        target_filename = target_filename if target_filename else filename
        model_template = template_env.get_template(filename)
        model = model_template.render(**self.config)
        with open(os.path.join(self.dir, target_filename), "w") as model_file:
            model_file.write(model)

    def create_directory_path(self):
        final_results_path = os.path.join("results", self.config['config_file'].split('.')[0], self.config['datetime'])
        directory_path = os.path.join(final_results_path, "experiments", self.name)
        os.makedirs(directory_path)
        os.makedirs(os.path.join(directory_path, "evochecker"))
        os.makedirs(os.path.join(directory_path, "baseline"))
        os.makedirs(os.path.join(directory_path, "results"))
        self.final_results_dir = final_results_path
        self.dir = directory_path

    def edit_config(self):
        self.config["service1"]["name"] = "service1"
        self.config["service2"]["name"] = "service2"
        self.config["service3"]["name"] = "service3"
        self.config["service1"]["state_name"] = "s1"
        self.config["service2"]["state_name"] = "s2"
        self.config["service3"]["state_name"] = "s3"
        if "model_checking" not in self.config:
            self.config["model_checking"] = {"run": False}
        if "evochecker" not in self.config:
            self.config["evochecker"] = {"run": False}
        if "uac" not in self.config:
            self.config["uac"] = {"strategy": "URC", "double_check_failures": True}
        if "urc" not in self.config:
            self.config["urc"] = {"deactivated": False}
        if "alarm_sender" not in self.config:
            self.config["alarm_sender"] = {"rate": "1/3"}
        if "prism_computation_engine" not in self.config["model_checking"]:
            self.config["model_checking"]["prism_computation_engine"] = "sparse"
        if "processors" not in self.config["evochecker"]:
            self.config["evochecker"]["processors"] = os.cpu_count()
        if "model_template_file" not in self.config["evochecker"]:
            self.config["evochecker"]["model_template_file"] = os.path.abspath(os.path.join(self.dir, "evochecker", "TAS_evochecker.prism"))
        if "properties_file" not in self.config["evochecker"]:
            self.config["evochecker"]["properties_file"] = os.path.abspath(os.path.join(self.dir, "evochecker", "TAS_evochecker.props"))
        if "steps_per_round" not in self.config:
            self.config["steps_per_round"] = 10
        else:
            self.config["steps_per_round"] = int(self.config["steps_per_round"])
        if "rounds" not in self.config["model_checking"]:
            self.config["model_checking"]["rounds"] = 15
        self.config["model_checking"]["steps"] = int(int(self.config["model_checking"]["rounds"]) * self.config["steps_per_round"])
        self.config["model_checking"]["alarms"] = int(self.config["model_checking"]["rounds"]) * float(Fraction(self.config["alarm_sender"]["rate"]))
        if "calculate_aggregated_failure_rate" not in self.config:
            self.config["calculate_aggregated_failure_rate"] = False
        if "calculate_baseline" not in self.config:
            self.config["calculate_baseline"] = False
        if "check_steps_per_round" not in self.config:
            self.config["check_steps_per_round"] = False
        self.calculate_persistent_failue_service()
        self.calculate_services_order()

    def calculate_persistent_failue_service(self):
        s1 = self.config["service1"]
        s2 = self.config["service2"]
        s3 = self.config["service3"]

        for s in [s1, s2, s3]:
            if ("simple_service" in s and s["simple_service"]) or ("failure_rate" in s and "persistent_failure_rate" in s and "persistent_recovery_rate" in s):
                self.log_debug(f"{s['name']}: Service calculation not needed, skipping...")
                continue
            if "failure_rate" in s and "persistent_failure_rate" in s and "persistent_recovery_rate" not in s:
                self.log_info(f"{s['name']}: Service calculation started")
                s['persistent_recovery_rate'] = self.calculate_persistent_recovery_rate(float(s['target_failure_rate']), float(s['failure_rate']), float(s['persistent_failure_rate']))
                self.log_info(f"{s['name']}: Service calculation finished")
                self.log_debug(f"{s['name']}: found persistent_recovery_rate of: {s['persistent_recovery_rate']}")
            elif "failure_rate" in s and "persistent_failure_rate" not in s and "persistent_recovery_rate" in s:
                self.log_info(f"{s['name']}: Service calculation started")
                s['persistent_failure_rate'] = self.calculate_persistent_failure_rate(float(s['target_failure_rate']), float(s['failure_rate']), float(s['persistent_recovery_rate']))
                self.log_info(f"{s['name']}: Service calculation finished")
                self.log_debug(f"{s['name']}: found persistent_failure_rate of: {s['persistent_failure_rate']}")
            elif "failure_rate" not in s and "persistent_failure_rate" in s and "persistent_recovery_rate" in s:
                self.log_info(f"{s['name']}: Service calculation started")
                s['failure_rate'] = self.calculate_failure_rate(float(s['target_failure_rate']), float(s['persistent_failure_rate']), float(s['persistent_recovery_rate']))
                self.log_info(f"{s['name']}: Service calculation finished")
                self.log_debug(f"{s['name']}: found failure_rate of: {s['failure_rate']}")
            else:
                self.log_error(f"{s['name']}: Could not Calulate")
                sys.exit(1)

    # TODO refactor
    def calculate_persistent_recovery_rate(self, target_failure_rate, failure_rate, persistent_failure_rate, precision=5):
        MAX_RETRYS = 25
        lower_bound = round(0.1**precision, precision)
        upper_bound = 1.0
        counter = 0
        prism_computation_engine = self.config["model_checking"]["prism_computation_engine"]

        while lower_bound <= upper_bound:
            mid = (lower_bound + upper_bound) / 2
            command = f'prism models/Persistant_failure_rate_model.prism -pf S=?[s>0] -maxiters 100000 -const failure_rate={failure_rate} -const persistent_failure_rate={persistent_failure_rate} -const persistent_recovery_rate={mid} -{prism_computation_engine}'
            mid_value = float(get_prism_results(command)[0])

            if abs(mid_value - target_failure_rate) < round(0.1**precision, precision):  # Adjust epsilon according to your precision requirement
                return mid  # Found a value within the acceptable range
            elif mid_value > target_failure_rate:
                lower_bound = mid
            else:
                upper_bound = mid
            counter += 1
            if counter >= MAX_RETRYS:
                raise Exception("NoServiceFound", "maximal retrys exceeded")
        return mid

    # TODO refactor
    def calculate_persistent_failure_rate(self, target_failure_rate, failure_rate, persistent_recovery_rate, precision=5):
        MAX_RETRYS = 25
        lower_bound = round(0.1**precision, precision)
        upper_bound = float(target_failure_rate)
        counter = 0
        prism_computation_engine = self.config["model_checking"]["prism_computation_engine"]

        while lower_bound <= upper_bound:
            mid = (lower_bound + upper_bound) / 2
            command = f'prism models/Persistant_failure_rate_model.prism -pf S=?[s>0] -maxiters 100000 -const failure_rate={failure_rate} -const persistent_failure_rate={mid} -const persistent_recovery_rate={persistent_recovery_rate} -{prism_computation_engine}'
            mid_value = float(get_prism_results(command)[0])

            if abs(mid_value - target_failure_rate) < round(0.1**precision, precision):  # Adjust epsilon according to your precision requirement
                return mid  # Found a value within the acceptable range
            elif mid_value < target_failure_rate:
                lower_bound = mid
            else:
                upper_bound = mid
            counter += 1
            if counter >= MAX_RETRYS:
                raise Exception("NoServiceFound", "maximal retrys exceeded")
        return mid

    # TODO refactor
    def calculate_failure_rate(self, target_failure_rate, persistent_failure_rate, persistent_recovery_rate, precision=5):
        MAX_RETRYS = 25
        lower_bound = round(0.1**precision, precision)
        upper_bound = float(target_failure_rate)
        counter = 0
        prism_computation_engine = self.config["model_checking"]["prism_computation_engine"]

        while lower_bound <= upper_bound:
            mid = (lower_bound + upper_bound) / 2
            command = f'prism models/Persistant_failure_rate_model.prism -pf S=?[s>0] -maxiters 100000 -const failure_rate={mid} -const persistent_failure_rate={persistent_failure_rate} -const persistent_recovery_rate={persistent_recovery_rate} -{prism_computation_engine}'
            mid_value = float(get_prism_results(command)[0])

            if abs(mid_value - target_failure_rate) < round(0.1**precision, precision):  # Adjust epsilon according to your precision requirement
                return mid  # Found a value within the acceptable range
            elif mid_value < target_failure_rate:
                lower_bound = mid
            else:
                upper_bound = mid
            counter += 1
            if counter >= MAX_RETRYS:
                raise Exception("NoServiceFound", "maximal retrys exceeded")
        return mid

    def calculate_aggregated_failure_rate(self):
        prism_computation_engine = self.config["model_checking"]["prism_computation_engine"]
        command = f"prism models/3_services.prism models/3_services.props \
            -const s1_failure_rate={self.config['service1']['failure_rate']} \
            -const s1_persistent_failure_rate={self.config['service1']['persistent_failure_rate']} \
            -const s1_persistent_recovery_rate={self.config['service1']['persistent_recovery_rate']} \
            -const s2_failure_rate={self.config['service2']['failure_rate']} \
            -const s2_persistent_failure_rate={self.config['service2']['persistent_failure_rate']} \
            -const s2_persistent_recovery_rate={self.config['service2']['persistent_recovery_rate']} \
            -const s3_failure_rate={self.config['service3']['failure_rate']} \
            -const s3_persistent_failure_rate={self.config['service3']['persistent_failure_rate']} \
            -const s3_persistent_recovery_rate={self.config['service3']['persistent_recovery_rate']} \
            -{prism_computation_engine}"
        self.aggregated_failure_rate = float(get_prism_results(command)[0])
        self.log_debug(f'Calculated aggregated failure rate: {self.aggregated_failure_rate}')
        with open(os.path.join(self.dir, "results", "aggregated_failure_rate.csv"), "w") as result_file:
            result_file.write(str(self.aggregated_failure_rate))

    def calculate_services_order(self):
        class Configuration:
            def __init__(self, s1_np, s2_np, s3_np, s1_pf, s2_pf, s3_pf) -> None:
                self.order = []
                self.mode = 0
                self.s1_np = s1_np
                self.s2_np = s2_np
                self.s3_np = s3_np
                self.s1_pf = s1_pf
                self.s2_pf = s2_pf
                self.s3_pf = s3_pf

        # assuming that the failure rate of a service is known
        # order_config [order, configuration]
        configurations = []
        for s1_np, s2_np, s3_np, s1_pf, s2_pf, s3_pf in product([False, True], repeat=6):
            # np = needs probing
            # pf = (kown) persistant failure
            c = Configuration(s1_np, s2_np, s3_np, s1_pf, s2_pf, s3_pf)
            c.order = ["s1", "s2", "s3"]
            if not s1_np and not s2_np and s3_np:
                c.order = ["s3", "s1", "s2"]
            elif not s1_np and s2_np and not s3_np:
                c.order = ["s2", "s1", "s3"]
            elif not s1_np and s2_np and s3_np:
                if float(self.config["service2"]["target_failure_rate"]) == float(self.config["service3"]["target_failure_rate"]):
                    if float(self.config["service2"]["costs"]) < float(self.config["service3"]["costs"]):
                        c.order = ["s2", "s3", "s1"]
                    else:
                        c.order = ["s3", "s2", "s1"]
                elif float(self.config["service2"]["target_failure_rate"]) < float(self.config["service3"]["target_failure_rate"]):
                    c.order = ["s2", "s3", "s1"]
                else:
                    c.order = ["s3", "s2", "s1"]
            elif s1_np and not s2_np and s3_np:
                if float(self.config["service1"]["target_failure_rate"]) == float(self.config["service3"]["target_failure_rate"]):
                    if float(self.config["service1"]["costs"]) < float(self.config["service3"]["costs"]):
                        c.order = ["s1", "s3", "s2"]
                    else:
                        c.order = ["s3", "s1", "s2"]
                elif float(self.config["service1"]["target_failure_rate"]) < float(self.config["service3"]["target_failure_rate"]):
                    c.order = ["s1", "s3", "s2"]
                else:
                    c.order = ["s3", "s1", "s2"]
            elif s1_np and s2_np and not s3_np:
                if float(self.config["service1"]["target_failure_rate"]) == float(self.config["service2"]["target_failure_rate"]):
                    if float(self.config["service1"]["costs"]) < float(self.config["service2"]["costs"]):
                        c.order = ["s1", "s2", "s3"]
                    else:
                        c.order = ["s2", "s1", "s3"]
                elif float(self.config["service1"]["target_failure_rate"]) < float(self.config["service2"]["target_failure_rate"]):
                    c.order = ["s1", "s2", "s3"]
                else:
                    c.order = ["s2", "s1", "s3"]
            elif s1_np and s2_np and s3_np:
                services_sorted_tfr = sorted(["service1", "service2", "service3"], key=lambda s: float(self.config[s]["target_failure_rate"]))
                if float(self.config[services_sorted_tfr[0]]["target_failure_rate"]) == float(self.config[services_sorted_tfr[1]]["target_failure_rate"]):
                    if float(self.config[services_sorted_tfr[0]]["costs"]) > float(self.config[services_sorted_tfr[1]]["costs"]):
                        services_sorted_tfr[0], services_sorted_tfr[1] = services_sorted_tfr[1], services_sorted_tfr[0]  # swap
                if float(self.config[services_sorted_tfr[1]]["target_failure_rate"]) == float(self.config[services_sorted_tfr[2]]["target_failure_rate"]):
                    if float(self.config[services_sorted_tfr[1]]["costs"]) > float(self.config[services_sorted_tfr[2]]["costs"]):
                        services_sorted_tfr[1], services_sorted_tfr[2] = services_sorted_tfr[2], services_sorted_tfr[1]  # swap
                services_sorted_tfr = [s.replace('service1', 's1') for s in services_sorted_tfr]
                services_sorted_tfr = [s.replace('service2', 's2') for s in services_sorted_tfr]
                services_sorted_tfr = [s.replace('service3', 's3') for s in services_sorted_tfr]
                c.order = services_sorted_tfr
            else:
                c.order = ["s1", "s2", "s3"]

            if not s1_np and s1_pf:
                c.order.remove("s1")
                c.order.append("s1")
            if not s2_np and s2_pf:
                c.order.remove("s2")
                c.order.append("s2")
            if not s3_np and s3_pf:
                c.order.remove("s3")
                c.order.append("s3")

            match c.order:
                case ["s1", "s2", "s3"]:
                    c.mode = 1
                case ["s1", "s3", "s2"]:
                    c.mode = 2
                case ["s2", "s1", "s3"]:
                    c.mode = 3
                case ["s2", "s3", "s1"]:
                    c.mode = 4
                case ["s3", "s1", "s2"]:
                    c.mode = 5
                case ["s3", "s2", "s1"]:
                    c.mode = 6

            configurations.append(c)

        # for so in configurations:
        #     print(so.order, "mode=", so.mode, "s1_np=", so.s1_np, "s2_np=", so.s2_np, "s3_np=", so.s3_np, "s1_pf=", so.s1_pf, "s2_pf=", so.s2_pf, "s3_pf=", so.s3_pf)
        self.config['configurations'] = configurations

    def calculate_baseline(self):
        self.log_info('Starting baseline calculation')
        cmd = (f'prism {os.path.join(self.dir, "baseline", "TAS_baseline.prism")} '
               f'{os.path.join(self.dir, "baseline", "TAS_baseline.props")} '
               f'-{self.config["model_checking"]["prism_computation_engine"]} '
               f'-const s1_latest_probe_max=1:{self.config["service1"]["latest_probe_max"]} '
               f'-const s2_latest_probe_max=1:{self.config["service2"]["latest_probe_max"]} '
               f'-const s3_latest_probe_max=1:{self.config["service3"]["latest_probe_max"]} '
               f'-exportresults {os.path.join(self.dir, "results", "baseline_prism.txt")}')
        results = get_prism_results(cmd)
        # Parley Baseline adapter
        with open(os.path.join(self.dir, "results", "Front"), 'w') as front_file:
            for i in range(0, len(results), 2):
                front_file.write(f'{results[i].split()[0]}\t{results[i+1].split()[0]}')
                if i < len(results) - 2:
                    front_file.write('\n')

    def run_model_checking(self):
        self.log_info('Starting model checking')
        results = get_prism_results(f'prism {os.path.join(self.dir, "TAS.prism")} {os.path.join(self.dir, "TAS.props")} -{self.config["model_checking"]["prism_computation_engine"]}')
        results = list(map(lambda result: re.search(r"(\d+\.\d+|\d+)", result).group(), results))
        self.log_debug(f"model checking results: {results}")
        with open(os.path.join(self.dir, "results", "model_checking.csv"), "w") as results_file:
            results_file.write(results[0])
            for result in results[1:]:
                results_file.write(CSV_DELIMITER + result)

    def run_evochecker(self, config_file):
        pwd = os.getcwd()
        os.chdir('evochecker')
        os.environ["LD_LIBRARY_PATH"] = "libs/runtime"
        # stderr=sys.stderr, stdout=sys.stdout, capture_output=True, text=True
        output_file_path = os.path.join("..", self.dir, "evochecker", "output.txt")
        error_file_path = os.path.join("..", self.dir, "evochecker", "error.txt")
        time_start = time_module.time()
        with open(output_file_path, "w") as output_file:
            with open(error_file_path, "w") as error_file:
                subprocess.run(["nice", "-n", "19", "java", "-jar", "target/EvoChecker-1.1.0.jar", config_file], stdout=output_file, stderr=error_file)
        with open(output_file_path, "r") as output_file:
            with open(error_file_path, "r") as error_file:
                time_end = time_module.time()
                pareto_front_file, pareto_set_file, time = "", "", ""
                for line in chain(output_file.readlines(), error_file.readlines()):
                    if line.startswith("Pareto Front: "):
                        pareto_front_file = os.path.abspath(line[13:].strip())
                    elif line.startswith("Pareto Set: "):
                        pareto_set_file = os.path.abspath(line[11:].strip())
                    elif line.startswith("Time:"):
                        time = line[5:].strip()
        if time == "":
            time = str(time_end - time_start)
        evochecker = EvoChecker(pareto_front_file, pareto_set_file, time)
        os.chdir(pwd)
        return evochecker

    def log_debug(self, msg):
        logger.debug(f'{self.name}: {msg}')

    def log_info(self, msg):
        logger.info(f'{self.name}: {msg}')

    def log_warning(self, msg):
        logger.warning(f'{self.name}: {msg}')

    def log_error(self, msg):
        logger.error(f'{self.name}: {msg}')


class EvoChecker:
    def __init__(self, pareto_front_file, pareto_set_file, time) -> None:
        self.pareto_front_file = pareto_front_file
        self.pareto_set_file = pareto_set_file
        self.time = time

    def get_contents(self):
        with open(self.pareto_front_file, "r") as f:
            self.pareto_front = f.read()
        with open(self.pareto_set_file, "r") as f:
            self.pareto_set = f.read()


def get_prism_results(command: str) -> List[str]:
    result = subprocess.run(command.split(), capture_output=True, text=True)
    Results = []
    if result.returncode == 1:
        print("=== ERROR ====\n")
        print(result.stdout)
        sys.exit(1)
    for line in result.stdout.split('\n'):
        if line.startswith("Result"):
            Results.append(line[8:])
        elif line.startswith("Error:"):
            print("=== ERROR ====\n")
            print(result.stdout)
            sys.exit(1)
    return Results


def merge_configs(base_config, other_config):
    def merge_configs_helper(merged_config, other_config):
        for key, value in other_config.items():
            if key in merged_config and isinstance(merged_config[key], dict) and isinstance(value, dict):
                merged_config[key] = merge_configs(merged_config[key], value)
            else:
                merged_config[key] = value
        return merged_config

    merged_config = base_config.copy()
    return merge_configs_helper(merged_config, other_config)


def collecting_model_checking_results(experiments: List[Experiment]):
    for experiment in experiments:
        if experiment.config["model_checking"]["run"]:
            with open(os.path.join(experiment.final_results_dir, "model_checking.csv"), "a+") as final_file, \
                 open(os.path.join(experiment.dir, "results", "model_checking.csv"), "r") as experiment_file:
                final_file.write(experiment.name + CSV_DELIMITER + experiment_file.read())
                if experiment is not experiments[-1]:
                    final_file.write("\n")


def collecting_aggregated_failure_rate_results(experiments: List[Experiment]):
    for experiment in experiments:
        if experiment.config["calculate_aggregated_failure_rate"]:
            with open(os.path.join(experiment.final_results_dir, "aggregated_failure_rate.csv"), "a+") as final_file, \
                 open(os.path.join(experiment.dir, "results", "aggregated_failure_rate.csv"), "r") as experiment_file:
                final_file.write(experiment.name + CSV_DELIMITER + experiment_file.read())
                if experiment is not experiments[-1]:
                    final_file.write("\n")


def run(config_file):
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiments: List[Experiment] = []
    with open(config_file) as cf:
        configs = yaml.safe_load_all(cf)
        configs_list = [config for config in configs]
        configs = configs_list
        # The first config is always the base config
        for i, config in enumerate(configs):
            config = merge_configs(configs[0], config)
            config['config_file'] = str(os.path.basename(config_file))
            config['config_seq'] = i
            config['datetime'] = now
            experiments.append(Experiment(config))

    processes: List[multiprocessing.Process] = []
    for experiment in experiments:
        process = multiprocessing.Process(target=experiment.run)
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    collecting_model_checking_results(experiments)
    collecting_aggregated_failure_rate_results(experiments)


if __name__ == '__main__':
    try:
        config_file = sys.argv[1].strip()
    except Exception as e:
        logger.warning("Can not opening the config. Fallback to basic.yaml")
        logger.debug(e)
        config_file = "configurations/basic.yaml"
    if not os.path.exists(config_file):
        logger.error(f"Configuration file '{config_file}' does not exist")
        sys.exit(1)
    run(config_file)
