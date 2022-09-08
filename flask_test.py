import json

from flask import Flask, redirect, url_for, request


app = Flask(__name__)


@app.route('/success/<name>')
def success(name):
 return '调用成功 %s' % name


@app.route('/login', methods = ['POST', 'GET'])
def login():

    if request.method == 'POST':
        user = request.form['name']
    else:
        user = request.args.get('name')

    res_text = {
            'result':str(user)
        }
    return json.dumps(res_text)


if __name__ == '__main__':
 app.run(debug=True)