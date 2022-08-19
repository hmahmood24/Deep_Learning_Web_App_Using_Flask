from flask import Flask

webapp = Flask(__name__)

from webapp import views, params, xnet

# Set file upload parameters
webapp.config["SECRET_KEY"] = params.SECRET_KEY
webapp.config["UPLOAD_FOLDER"] = params.UPLOAD_FOLDER
webapp.config["ALLOWED_EXTENSIONS"] = params.ALLOWED_EXTENSIONS 
webapp.config['MAX_CONTENT_LENGTH'] = params.MAX_CONTENT_LENGTH

# Load the XNet model into memory
webapp.config['XNET_MODEL'] = xnet.load_model(path = 'webapp/XNet')