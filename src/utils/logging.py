import logging
import sys
from pathlib import Path

def setup_logging(log_dir: Path = None, level: int = logging.INFO) -> logging.Logger:
    """Configures console and file handlers."""
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
        
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "pilot.log")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger

def log_gate_decision(logger: logging.Logger, gate_name: str, decision: str, metrics: dict):
    """Structured log entry for pilot gate decisions."""
    logger.info("="*50)
    logger.info(f"GATE DECISION: {gate_name}")
    logger.info(f"RESULT: {decision}")
    logger.info("-" * 20)
    for k, v in metrics.items():
        logger.info(f"{k}: {v}")
    logger.info("="*50)

def log_step_start(logger: logging.Logger, step: int, description: str):
    logger.info("")
    logger.info("#"*60)
    logger.info(f"# STEP {step}: {description}")
    logger.info("#"*60)

def log_step_end(logger: logging.Logger, step: int):
    logger.info(f"# END STEP {step}")
    logger.info("#"*60 + "\n")
