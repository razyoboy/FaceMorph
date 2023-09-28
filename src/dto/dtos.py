import json
from flask import Response

def create_error_response(e, status_code=500):
    error_response = json.dumps({
        "status": "Error",
        "errorType": type(e).__name__,
        "msg": str(e)
    })

    return Response(error_response, mimetype='application/json'), status_code

def create_response(msg="Up and running!", status_code=200):
    response = json.dumps({
        "status": "Ok" if status_code == (200 or 201) else "Error",
        "msg": msg
    })

    return Response(response, mimetype='application/json'), status_code