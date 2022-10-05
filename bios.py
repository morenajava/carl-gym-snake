import os.path as osp

# IOT testing environment
CAServerIPAddress = 'localhost'
CAServerPort = 7044
CAServerInterval = 0.001
LOG_BASE_DIRECTORY = "../ConstraintAcquisition/benchmarks/queries/"
RUN_NAME = "minigrid"
# LOGS_PATH = osp.join(LOG_BASE_DIRECTORY, RUN_NAME, "logs.csv")
# EXAMPLE_PATH = osp.join(LOG_BASE_DIRECTORY, RUN_NAME, "queries.txt")
# GYM_LOGGER_PATH = osp.join(LOG_BASE_DIRECTORY, RUN_NAME, "sb_logger")
# GYM_MONITOR_PATH = osp.join(LOG_BASE_DIRECTORY, RUN_NAME, "monitor.csv")

# Production
# CAServerIPAddress = '192.168.1.107'
# CAServerPort = 7044
# CAServerInterval = 0.05
# LOG_BASE_DIRECTORY = "/home/morena/benchmarks/queries/"
LOGS_PATH = osp.join(LOG_BASE_DIRECTORY, RUN_NAME, "logs.csv")
EXAMPLE_PATH = osp.join(LOG_BASE_DIRECTORY, RUN_NAME, "queries.txt")
GYM_LOGGER_PATH = osp.join(LOG_BASE_DIRECTORY, RUN_NAME, "sb_logger")
GYM_MONITOR_PATH = osp.join(LOG_BASE_DIRECTORY, RUN_NAME, "monitor.csv")

#GYM ENVIRONMENT SETUP
GYM_ENVIRONMENT = "Snake-16x16-v0"
# GYM_ENVIRONMENT = "MiniGrid-LavaGapS5-v0"
STEPS = 50000
SEED = 20
CARL = "none"  # none/mask/replace
CA_SERVER_CACHE = 0.01