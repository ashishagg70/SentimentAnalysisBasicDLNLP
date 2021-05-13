from flask import Blueprint
from flask import request
import json
import numpy as np
from .review_processing import predict_ratings

bp = Blueprint('ratings', __name__)

@bp.route('/', methods=['POST'])
def rating():
    req = request.json
    if req['review'] is None or len(req['review'])==0:
        return {
            "status": 403,
            "message": "Please send review"
        }
    
    ratings,html = predict_ratings(req['review'], int(req["num_samples"]))
    ratings = np.around(ratings,3)
    maxr = np.argmax(ratings, axis=1) + 1
    print('completed',maxr)
    return {
        "status":200,
        "review":req['review'],
        "pr1":str(ratings[0,0]),
        "pr2":str(ratings[0,1]),
        "pr3":str(ratings[0,2]),
        "pr4":str(ratings[0,3]),
        "pr5":str(ratings[0,4]),
        "rating": str(maxr[0]),
        "html":html
    }

    