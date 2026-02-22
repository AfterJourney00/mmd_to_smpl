from pathlib import Path
import logging

def config_logging(output_log_path: Path):
    output_log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_log_path),
            logging.StreamHandler()
        ]
    )

def format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes} mins {secs:.1f} seconds"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours} hours {minutes} mins {secs:.1f} seconds"