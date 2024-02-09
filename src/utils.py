import logging
import socket
from pathlib import Path

import weaviate

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def check_weaviate_class_exists(
    weaviate_client: weaviate.Client,
    weaviate_class: str,
) -> bool:
    """Check if a class exists in Weaviate."""
    classes = weaviate_client.schema.get()["classes"]
    available_classes = [_class["class"] for _class in classes]
    if weaviate_class not in available_classes:
        logger.error(f"Class {weaviate_class} does not exist in Weaviate.")
        return False

    logger.info(f"Class {weaviate_class} exists in Weaviate.")
    return True


def get_host_ip():
    try:
        # Create a socket object and connect to an external server
        # This step is done to get the local machine's IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        host_ip = s.getsockname()[0]
    except Exception as e:
        print(f"Error while retrieving host IP address: {e}")
        host_ip = None
    finally:
        s.close()

    return host_ip


def create_directory_if_not_exists(path):
    p_base_path = Path(path).resolve()
    p_base_path.mkdir(parents=True, exist_ok=True)
    return str(p_base_path)
