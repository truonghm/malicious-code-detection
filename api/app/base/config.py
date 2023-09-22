import os


class Settings(object):
    MONGO_DB_URI = os.getenv(
        "SIMCEL_MONGODB_URI",
        )
    API_HOST_PORT = int(os.getenv("API_HOST_PORT", 8501))
    API_HOST_DOMAIN = os.getenv("API_HOST_DOMAIN", "0.0.0.0")
    RELOAD_CODE = os.getenv("RELOAD_CODE", False)
    NUMBER_OF_WORKER = int(os.getenv("NUMBER_OF_WORKER", 4))

settings = Settings()
