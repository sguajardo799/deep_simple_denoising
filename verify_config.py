from src.config import Config

def verify():
    config = Config.from_yaml("config.yaml")
    print(f"Loaded Device: {config.general.device}")
    print(f"Device ID: {config.general.device_id}")

if __name__ == "__main__":
    verify()
