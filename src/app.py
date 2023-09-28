from flask import Flask, request, send_file, Response
from dto.dtos import create_response, create_error_response

import os
import main
import json

from waitress import serve

app = Flask(__name__)

@app.route('/status', methods=['GET'])
def status():
    return create_response()

@app.route('/morph', methods=['POST'])
def morph():
    image1 = request.files.get('image1')
    image2 = request.files.get('image2')

    if not image1 or not image2:
        return create_response("Both image1 and image2 are required", 400)

    # Get the alpha parameter from the request, with a default value of 0.5
    alpha = request.form.get('alpha', default=0.5, type=float)

    # Validate alpha value
    if not 0 <= alpha <= 1:
        return create_response("Alpha value must be between 0 and 1", 400)

    # Get the include_borders parameter from the request, with a default value of True
    include_borders = request.form.get('include_borders', default=True, type=lambda x: x.lower() == 'true')

    try:
        result_image_path = main.face_morph_api(
            image1,
            image2,
            alpha=alpha,
            include_borders=include_borders
        )

        return send_file(result_image_path, mimetype='image/png')
    
    except TypeError as e:
        error_response = json.dumps({
            "status": "Error",
            "errorType": type(e).__name__,
            "msg": str(e),
            "cause": "Possible that faces are not detected, please try different images"
        })
        
        return Response(error_response, mimetype='application/json'), 500
    
    except Exception as e:
        return create_error_response(e, 500)

if __name__ == '__main__':
    if os.name == 'nt':
        app.run(debug=True)
    else:
        serve(app, port=int(os.environ.get("PORT", 8080)))
