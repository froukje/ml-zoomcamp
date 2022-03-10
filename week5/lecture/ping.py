from flask import Flask

# create a flask app
app = Flask("ping")

# decorator to turn this function into a web service
# the method will live in "/ping"
# access this method with th "GET" method
@app.route("/ping", methods=["GET"])
def ping(): 
    return "PONG"

if __name__ == "__main__":
    # "0.0.0.0"=localhost
    app.run(debug=True, host="0.0.0.0", port=9696)
