
import logging


# define a logfile and a level at one point to reuse it in all modules
# actually we log to std out and to /logs/SA.log

logger = logging.getLogger()
hdlr = logging.FileHandler('./logs/' + 'SA' + '.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
# console output
hdlr_console = logging.StreamHandler()
hdlr_console.setFormatter(formatter)
logger.addHandler(hdlr_console)

logger.setLevel(logging.INFO)