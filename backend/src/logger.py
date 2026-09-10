import logging
# 配置日志格式
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
# 创建 logger
logger = logging.getLogger("agent")
logger.setLevel(logging.INFO)
# 控制台输出
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
logger.addHandler(console_handler)
# 可选：同时写入文件
file_handler = logging.FileHandler("agent.log", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
logger.addHandler(file_handler)
def log_step(step_name: str, message: str, extra: dict = None):
    """记录 Agent 执行步骤"""
    if extra:
        logger.info(f"[{step_name}] {message} | {extra}")
    else:
        logger.info(f"[{step_name}] {message}")
def log_error(step_name: str, error: Exception):
    """记录错误"""
    logger.error(f"[{step_name}] 错误: {type(error).__name__}: {str(error)}")