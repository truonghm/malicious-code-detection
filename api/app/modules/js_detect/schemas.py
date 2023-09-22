from pydantic import BaseModel


class JavaScriptRequestItem(BaseModel):
    idx: str
    code: str

class JavaScriptRequest(BaseModel):
    javascript: list[JavaScriptRequestItem]


class JavaScriptResponseItem(BaseModel):
    idx: str
    label: str

class JavaScriptResponse(BaseModel):
    results: list[JavaScriptResponseItem]
