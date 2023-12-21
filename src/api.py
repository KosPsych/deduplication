from flask import Flask, make_response
from utils.raw_preprocessing import create_dataset
from train import train
from test import test
from inference import inference
from flask_restx import Resource, Api, fields
from werkzeug.datastructures import FileStorage
from PIL import Image

app = Flask(__name__)
api = Api(app)

# Define an upload parser for handling file uploads
upload_parser = api.parser()
upload_parser.add_argument('image1', location='files', type=FileStorage, help='First image to compare', required=True)
upload_parser.add_argument('image2', location='files', type=FileStorage, help='Second image to compare', required=True)
upload_parser.add_argument('title1', help='First title to compare', required=False)
upload_parser.add_argument('title2', help='Second title to compare', required=False)


class CreateDataset(Resource):
    def get(self):
        # Creates a pair csv dataset using the data.txt
        return make_response(create_dataset())



class Training(Resource):
    def post(self):
        # Return response for the training endpoint
        return make_response(train())


class Testing(Resource):
    def post(self):
        # Return response for the testing endpoint
        return make_response(test())

class Inference(Resource):
    @api.expect(upload_parser)
    def post(self):
        # Parse arguments from the request
        args = upload_parser.parse_args()
        # Image 1
        image1 = args['image1']
        # Image 2
        image2 = args['image2']
        # Title 1
        title1 = args['title1']
        # Title 2
        title2 = args['title2']
        # Perform inference and get scores
        scores = inference(Image.open(image1), Image.open(image2), title1, title2)
        # Return response for the inference endpoint
        return make_response(scores)

# Add the resource classes to the API
api.add_resource(Training, '/training')
api.add_resource(Testing, '/testing')
api.add_resource(Inference, '/inference')
api.add_resource(CreateDataset, '/create_dataset')

if __name__ == '__main__':
    # Run the Flask application
    app.run(debug=True, host='0.0.0.0')
