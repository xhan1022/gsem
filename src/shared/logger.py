"""Logging system with real-time progress monitoring."""
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)


class ProgressLogger:
    """Custom logger with real-time progress monitoring."""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"pipeline_{timestamp}.log"

        # Configure logging
        self.logger = logging.getLogger("GSEM")
        self.logger.setLevel(logging.INFO)

        # File handler with unbuffered output
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        # Force flush after each log
        file_handler.stream.reconfigure(line_buffering=True)

        # Console handler with unbuffered output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        # Force flush after each log
        console_handler.stream.reconfigure(line_buffering=True)

        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.current_case = None
        self.total_cases = 0

    def start_pipeline(self, total_cases: int):
        """Start pipeline logging."""
        self.total_cases = total_cases
        self.logger.info("=" * 80)
        self.logger.info(f"{Fore.CYAN}GSEM Phase 1 Pipeline Started{Style.RESET_ALL}")
        self.logger.info(f"Total cases to process: {total_cases}")
        self.logger.info("=" * 80)

    def start_case(self, case_id: str, case_num: int):
        """Start processing a case."""
        self.current_case = case_id
        msg = f"\n{Fore.GREEN}[Case {case_num}/{self.total_cases}]{Style.RESET_ALL} 开始处理案例 {Fore.YELLOW}{case_id}{Style.RESET_ALL}"
        self.logger.info(msg)

    def log_sampling(self, sample_num: int, total_samples: int, status: str, success: bool):
        """Log sampling progress."""
        status_icon = f"{Fore.GREEN}✓{Style.RESET_ALL}" if success else f"{Fore.RED}✗{Style.RESET_ALL}"
        msg = f"  ├─ [采样 {sample_num}/{total_samples}] {status} {status_icon}"
        self.logger.info(msg)

    def log_pairing(self, num_pairs: int):
        """Log failure-success pairing."""
        msg = f"  ├─ 匹配到 {Fore.CYAN}{num_pairs}{Style.RESET_ALL} 个失败-成功配对"
        self.logger.info(msg)

    def log_experiences(self, indication: int, contraindication: int):
        """Log extracted experiences."""
        total = indication + contraindication
        msg = (f"  └─ 提取经验: {Fore.CYAN}{total}条{Style.RESET_ALL} "
               f"(Indication: {indication}, Contraindication: {contraindication})")
        self.logger.info(msg)

    def log_stage(self, stage_name: str, status: str = "开始"):
        """Log stage progress."""
        msg = f"\n{Fore.MAGENTA}[{stage_name}]{Style.RESET_ALL} {status}"
        self.logger.info(msg)

    def log_error(self, message: str, case_id: Optional[str] = None):
        """Log error."""
        if case_id:
            msg = f"{Fore.RED}[ERROR]{Style.RESET_ALL} Case {case_id}: {message}"
        else:
            msg = f"{Fore.RED}[ERROR]{Style.RESET_ALL} {message}"
        self.logger.error(msg)
        # Force flush to ensure real-time output
        for handler in self.logger.handlers:
            handler.flush()

    def log_warning(self, message: str):
        """Log warning."""
        msg = f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL} {message}"
        self.logger.warning(msg)
        # Force flush to ensure real-time output
        for handler in self.logger.handlers:
            handler.flush()

    def log_info(self, message: str):
        """Log info."""
        self.logger.info(message)
        # Force flush to ensure real-time output
        for handler in self.logger.handlers:
            handler.flush()

    def _renew(self, prefix: str) -> str:
        """Swap the file handler to a new log file with the given prefix.

        Returns the new log file path so callers can print / tail it.
        """
        # Close and remove existing file handlers
        for h in list(self.logger.handlers):
            if isinstance(h, logging.FileHandler):
                h.close()
                self.logger.removeHandler(h)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{prefix}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        ))
        file_handler.stream.reconfigure(line_buffering=True)
        self.logger.addHandler(file_handler)
        return str(log_file)

    def finish_pipeline(self, success_count: int, failed_count: int, total_experiences: int):
        """Finish pipeline logging."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"{Fore.CYAN}Pipeline Completed{Style.RESET_ALL}")
        self.logger.info(f"成功处理: {Fore.GREEN}{success_count}{Style.RESET_ALL} 个案例")
        self.logger.info(f"失败跳过: {Fore.RED}{failed_count}{Style.RESET_ALL} 个案例")
        self.logger.info(f"总计提取: {Fore.CYAN}{total_experiences}{Style.RESET_ALL} 条经验")
        self.logger.info("=" * 80)


# Global logger instance
logger = ProgressLogger()
