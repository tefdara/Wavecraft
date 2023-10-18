import sys, logging

class colors:
    WHITE = '\033[97m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    GREY = '\033[37m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Logger:
    
    def __init__(self):
        self.error_logger = self.make_logger('error', 'wavecraft')
        self.logger = self.make_logger('message', 'wavecraft')
        self.warning_logger = self.make_logger('warning', 'wavecraft')
        self.value_logger = self.make_logger('value', 'wavecraft')
        
    def make_logger(self, type, name):
        logger = logging.Logger(name)
        handler = logging.StreamHandler()
        logger.setLevel(logging.INFO)
        handler.setLevel(logging.INFO)

        if type == 'message':
            formatter = logging.Formatter('%(level)s %(message)s')
        elif type == 'error':
            logger.setLevel(logging.ERROR)
            handler.setLevel(logging.ERROR)
            formatter = logging.Formatter('%(level)s %(message)s')
        elif type == 'warning':
            logger.setLevel(logging.WARNING)
            handler.setLevel(logging.WARNING)
            formatter = logging.Formatter('%(level)s %(message)s')
        elif type == 'value':
            formatter = logging.Formatter('     %(message)-30s %(value)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def get_logger(self, type):
        if type == 'message':
            return self.logger
        elif type == 'error':
            return self.error_logger
        elif type == 'warning':
            return self.warning_logger
        elif type == 'value':
            return self.value_logger
            
    def extra_log_string(self, prepend, append):
        return {'prepend': prepend, 'append': append}   

    def extra_log_value(self, value, unit):
        return {'value': value, 'unit': unit}
    
    def parse_message(self, message, type, any=None):
        #words that are wrapped in <> or any numbers will highlight based on the type of message
        type = type.upper()
        if type == 'INFO':
            message = message.replace('<', f'{colors.CYAN}')
            # find any numbers in the message and highlight them
            level = type.replace('INFO', f'[ {colors.GREEN}INFO{colors.ENDC} ]')
        elif type == 'STAT':
            message = message.replace('<', f'{colors.CYAN}')
            message = ''.join([f'{colors.CYAN}{x}{colors.ENDC}' if x.isdigit() else x for x in message])
            level = type.replace('STAT', f'[ {colors.CYAN}STAT{colors.ENDC} ]')
        elif type == 'WARNING':
            message = message.replace('<', f'{colors.YELLOW}')
            level = type.replace('WARNING', f'[ {colors.YELLOW}WARNING{colors.ENDC} ]')
        elif type == 'ERROR':
            message = colors.RED+message+colors.ENDC
            message = message.replace('<', f'{colors.ENDC}{colors.YELLOW}')
            message = message.replace('>', f'{colors.ENDC}')
            level = type.replace('ERROR', f'[ {colors.RED}ERROR{colors.ENDC} ]')
        elif type == 'VALUE':
            # message = colors.CYAN+message+colors.ENDC
            message = ''.join([f'{colors.GREEN}{x}{colors.ENDC}' if x.isdigit() else x for x in message])
            message = message.replace('<', f'{colors.ENDC}{colors.YELLOW}')
            message = message.replace('>', f'{colors.ENDC}')
            level=''
        elif type == 'DONE':
            message = message.replace('<', f'{colors.CYAN}')
            level = type.replace('DONE', f'[ {colors.GREEN}DONE{colors.ENDC} ]')
        elif type == 'ANY':
            message = message.replace('<', f'{colors.CYAN}')
            message = message.replace('>', f'{colors.ENDC}')
            if any is None: 
                raise ValueError('No value provided for ANY type.')
            else:
                any = any.upper()
            level = type.replace('ANY', f'[ {colors.GREEN}{any}{colors.ENDC} ]')
            
        message = message.replace('>', f'{colors.ENDC}')

        level={'level': level}
        return level, message

logger = Logger() 
class Debug:
    
    @staticmethod
    def log_info(message):
        level, message = logger.parse_message(message, type='info')
        log = logger.get_logger('message')
        log.info(message, extra=level)
        
    @staticmethod
    def log_error(message, exit=True):
        level, message = logger.parse_message(message, type='error')
        log = logger.get_logger('error')
        log.error(message, extra=level)
        if exit:
            sys.exit(1)
    
    @staticmethod
    def log_warning(message):
        level, message = logger.parse_message(message, type='warning')
        log = logger.get_logger('warning')
        log.info(message, extra=level)
    
    @staticmethod
    def log_stat(message):
        level, message = logger.parse_message(message, type='stat')
        log = logger.get_logger('message')
        log.info(message, extra=level)
    
    @staticmethod
    def log_value(message):
        _, message = logger.parse_message(message, type='value')
        log = logger.get_logger('value')
        value = message.split(':')[1]
        message = message.split(':')[0]+':'
        message = colors.CYAN+message+colors.ENDC
        extra = {'value': value}
        log.info(message, extra=extra)
        
    @staticmethod
    def log_done(message):
        level, message = logger.parse_message(message, type='done')
        log = logger.get_logger('message')
        log.info(message, extra=level)
    
    @staticmethod
    def log_any(message, any):
        level, message = logger.parse_message(message, type='any', any=any)
        log = logger.get_logger('message')
        log.info(message, extra=level)
