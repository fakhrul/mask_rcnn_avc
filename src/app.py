from flask import Flask, request,jsonify
from .config import app_config
from .shared.ImageProcess import ImageProcess
from PIL import Image
import io

def create_app(env_name):
    app = Flask(__name__)
    app.config.from_object(app_config[env_name])
    imageProcess = ImageProcess()

    @app.route('/')
    def index():
        return 'ok'

    @app.route("/predict", methods=["POST"])
    def predict():
        data = {"status": False}
        if request.files.get("image"):
            image = request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            result = imageProcess.predict(image)
            # preprocess the image and prepare it for classification
            # image = prepare_image(image, target=(224, 224))

            # # classify the input image and then initialize the list
            # # of predictions to return to the client
            # preds = model.predict(image)
            # results = imagenet_utils.decode_predictions(preds)
            # data["predictions"] = []

            # # loop over the results and add them to the list of
            # # returned predictions
            # for (imagenetID, label, prob) in results[0]:
            #     r = {"label": label, "probability": float(prob)}
            #     data["predictions"].append(r)

            # indicate that the request was a success
            data["status"] = True
            data["data"] = result

        # return the data dictionary as a JSON response
        return jsonify(data)
        
    return app