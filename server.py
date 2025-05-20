# from flask import Flask, render_template
# import subprocess

# app = Flask(__name__)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/run-canvas')
# def run_canvas():
#     subprocess.Popen(['python', 'canvas.py'])  # or 'python3' if needed
#     return "Canvas started! and"

# if __name__ == '__main__':
#     app.run(debug=True)




from flask import Flask, render_template
import subprocess
  
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/run-canvas')
def run_canvas():
    subprocess.Popen(["python", "canvas.py"])
    return '', 200

if __name__ == '__main__':
    app.run(debug=True) 