import logging

from fastapi import Depends, FastAPI
from fastapi.routing import APIRouter

from . import actions, schemas

logger = logging.getLogger("js_detection")
router = APIRouter()


@router.post("/dummy_predict", response_model=schemas.JavaScriptResponse)
def dummy_predict(request: schemas.JavaScriptRequest):
    examples = [
        schemas.JavaScriptResponseItem(**{"idx": "0", "label": "malicious"}),
        schemas.JavaScriptResponseItem(**{"idx": "1", "label": "benign"}),
    ]
    return schemas.JavaScriptResponse(
        **{"results": examples}
    )

@router.post("/predict", response_model=schemas.JavaScriptResponse)
def predict(request: schemas.JavaScriptRequest, classifier: actions.JavaScriptClassifier = Depends(actions.get_cls)):
    code_list = request.javascript

    cls_input = []
    for item in code_list:
        cls_input.append(
            {
                "idx": item.idx,
                "code": item.code,
                "doc": actions.TextDescription.MALICIOUS.value
            }
        )
    logger.debug(cls_input)
    results = classifier.predict(cls_input)
    results_by_schema = [schemas.JavaScriptResponseItem(**item) for item in results]
    return schemas.JavaScriptResponse(
        **{"results": results_by_schema}
    )
