import time
from logging.config import dictConfig

import uvicorn
from app.base.config import settings
from app.base.logging import get_log_config
from app.modules.js_detect.routers import router as js_router
from fastapi.applications import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from uvicorn.config import LOGGING_CONFIG

dictConfig(
    get_log_config(
        logger_name="js_detection",
        log_level="DEBUG",
    )
)

app = FastAPI(
    title="Malicious JavaScript Detection API Demo",
    description="Malicious JavaScript Detection API Demo",
    version="0.0.1",
    docs_url="/documentation",
    redoc_url="/redoc",
)

def add_all_routers(app):
    app.include_router(
        js_router, prefix="/js-detection"
    )

add_all_routers(app)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

ALLOWED_CORS_DOMAINS = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_CORS_DOMAINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
async def read_main():
    return {"msg": "Streamlit Backend APIs"}


if __name__ == "__main__":
    LOGGING_CONFIG["formatters"]["access"][
        "fmt"
    ] = '%(asctime)s %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
    LOGGING_CONFIG["formatters"]["access"]["datefmt"] = "%Y-%m-%d %H:%M:%S"

    uvicorn.run(
        "main:app",
        host=settings.API_HOST_DOMAIN,
        port=settings.API_HOST_PORT,
        reload=settings.RELOAD_CODE,
        workers=settings.NUMBER_OF_WORKER,
    )
