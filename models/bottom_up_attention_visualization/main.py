import io
import os
import pickle
import sys

import cv2
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, flash, make_response
from matplotlib.figure import Figure
from wtforms import Form, TextField, validators

app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

with open(sys.argv[1], mode="rb") as in_file:
    print("Loading bottom up features...")
    bottom_up_features = pickle.load(in_file, encoding="latin1")
    print("Bottom up features loaded!")


class ReusableForm(Form):
    name = TextField('Name:', validators=[validators.required()])


@app.route("/", methods=['GET', 'POST'])
def hello():
    form = ReusableForm(request.form)

    if request.method == 'POST':
        img_filename = request.form['name']

        if form.validate():
            fig = Figure()
            im = cv2.imread(os.path.join(sys.argv[2], img_filename))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            img = io.BytesIO()
            plt.imshow(im)
            for i in range(len(bottom_up_features[img_filename]["keep_boxes"])):
                bbox = bottom_up_features[img_filename]["boxes"][i]["coordinates"]
                if bbox[0] == 0:
                    bbox[0] = 1
                if bbox[1] == 0:
                    bbox[1] = 1
                plt.gca().add_patch(
                    plt.Rectangle(
                        (bbox[0], bbox[1]),
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1], fill=False,
                        edgecolor='red', linewidth=2, alpha=0.5
                    )
                )
            plt.savefig(img, format='png')
            img.seek(0)
            response = make_response(img.getvalue())
            response.mimetype = 'image/png'
            return response

        else:
            flash('Error: a valid pair ID is required!')

    return render_template('hello.html', form=form)


if __name__ == "__main__":
    app.run()
