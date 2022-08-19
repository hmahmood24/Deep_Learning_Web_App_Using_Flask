import os
from webapp import webapp
from flask import render_template, flash, request, session

from werkzeug.utils import secure_filename

from webapp import xnet

@webapp.route("/", methods = ['GET'])
def index():
    # Routine to render the home page of our website
    return render_template("index.html")

@webapp.route("/predict/", methods = ['GET'])
def predict():
    # Routine to route requests from the predict page to display the upload dialog box
    return render_template("predict.html")

@webapp.route("/upload/", methods = ['POST', 'GET'])
def upload():
    # Routine to deal with the uploaded form data i.e. image in this case
    # Fetches the uploaded image, runs basic validations and saves to the specified directory
    def __allowed_file(filename):
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in webapp.config["ALLOWED_EXTENSIONS"]

    # Clear any existing flash messages from the session
    session.pop('_flashes', None)

    # Run sanity checks
    if request.method == 'POST':
        print("Passed request method sanity check!")
        # First check whether or not the request contains our desired file
        if 'file' not in request.files:
            # Flash an error message and redirect to the same page
            flash('No file part! Try again!', 'error')
            return render_template("predict.html")

        print("Passed file exist sanity check!")
        uploaded_file = request.files['file']

        # Check for the allowed file types and save to the local disk
        if uploaded_file and __allowed_file(uploaded_file.filename):
            path = os.path.join(webapp.config['UPLOAD_FOLDER'], 
                                secure_filename(uploaded_file.filename))
            uploaded_file.save(path)
        else:
            # Flash an error message and redirect to the same page
            flash('The file must be of one of the accepted formats only! Try again!', 'error')
            return render_template("predict.html")
        print("Passed file extension and save check!")

        # Next, run the XNet prediction service
        result = xnet.predict(webapp.config['XNET_MODEL'], path = path)
        print("Passed XNet prediction check!")
        print(result)
        return render_template("upload.html", filename=uploaded_file.filename, result=result) 