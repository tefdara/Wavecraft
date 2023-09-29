import logging

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

class Debug:
    
    def __init__(self):
        self.error_logger = self.get_logger('error', 'wavecraft')
        self.logger = self.get_logger('message', 'wavecraft')
        self.warning_logger = self.get_logger('warning', 'wavecraft')
        self.value_logger = self.get_logger('value', 'wavecraft')
        
    def get_logger(self, type, name):
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
            formatter = logging.Formatter('%(message)-18s %(value)s (%(unit)s)')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
            
    def extra_log_string(self, prepend, append):
        return {'prepend': prepend, 'append': append}   

    def extra_log_value(self, value, unit):
        return {'value': value, 'unit': unit}
    
    def parse_message(self, message, type):
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
            message = {colors.RED}+message+{colors.ENDC}
            message = message.replace('<', f'{colors.ENDC}{colors.YELLOW}')
            message = message.replace('>', f'{colors.ENDC}')
            level = type.replace('ERROR', f'[ {colors.RED}ERROR{colors.ENDC} ]')
        elif type == 'VALUE':
            message = {colors.CYAN}+message+{colors.ENDC}
            level = ''
        elif type == 'DONE':
            message = message.replace('<', f'{colors.GREEN}')
            message = ''.join([f'{colors.CYAN}{x}{colors.ENDC}' if x.isdigit() else x for x in message])
            level = type.replace('DONE', f'[ {colors.GREEN}DONE{colors.ENDC} ]')
            
        message = message.replace('>', f'{colors.ENDC}')

        level={'level': level}
        return level, message
    
    @staticmethod
    def log_info(self, message):
        level, message = self.parse_message(message, type='info')
        self.logger.info(message, extra=level)
    @staticmethod    
    def log_error(self, message):
        level, message = self.parse_message(message, type='error')
        self.error_logger.error(message, extra=level)
    @staticmethod
    def log_warning(self, message):
        level, message = self.parse_message(message, type='warning')
        self.warning_logger.info(message, extra=level)
    @staticmethod
    def log_stat(self, message):
        level, message = self.parse_message(message, type='stat')
        self.logger.info(message, extra=level)
    @staticmethod
    def log_value(self, value, unit):
        level, message = self.parse_message(message, type='value')
        self.value_logger.info(message, extra=level)
    @staticmethod    
    def log_done(self, message):
        level, message = self.parse_message(message, type='info')
        self.value_logger.info('', extra=level)
