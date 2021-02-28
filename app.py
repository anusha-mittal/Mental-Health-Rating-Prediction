from flask import Flask,render_template,redirect,request
import joblib
import mentalPredictor

app=Flask(__name__)

@app.route('/')
def start():
    return render_template("index.html")

@app.route('/',methods=['POST'])
def getPred():
    if request.method=='POST':  
        names=['age','stress','insom','social','head','suicidal','conc','phy','anx','grow','ill']
        arr=[]
        for i in names:
            if i=='insom' or i=='suicidal' or i=='ill':
                arr.append(request.form[i])
                continue

            arr.append(int(request.form[i]))

        pred=mentalPredictor.predict(arr)

        
        return (pred)

if __name__=='__main__':
    app.run(debug=True)
