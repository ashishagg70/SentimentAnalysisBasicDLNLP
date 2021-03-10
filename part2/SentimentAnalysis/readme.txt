Keshav Agarwal (203050039)
Ashish Aggarwal (203050015)
Debabrata Biswal (203050024)


Steps to install and run:

Flask Server:
0. Download the model directory and keep it in SentimentAnalysisAPI directory:
https://drive.google.com/drive/folders/165qgiuEpu_R74ZlkWtYEwIxgQ6t4spoY?usp=sharing

1. Create Python virtual environment and activate it:
    a.  python3 -m venv venv
    b.  Activate for windows: 
            venv\Scripts\activate
        Activate for Linux:
            . venv/bin/activate

2. Install packages 
    a. pip install flask
    b. pip install requests
    c. pip install flask-cors 
    d. pip install pandas
    e. pip install tensorflow

3. cd SentimentAnalysis
4. For PowerShell, run:
       $env:FLASK_APP="SentimentAnalysisAPI"
   For Linux, run:
       export FLASK_APP="SentimentAnalysisAPI"
5. For PowerShell, run:
       $env:FLASK_ENV="developoment"
   For Linux, run:
       export FLASK_ENV="developoment"

6. flask run

Frontend:
7. Open following webpage in browser: SentimentAnalysis/SentimentAnalysisUI/index.html


For testing API (e.g. in Postman)
=============================================
URL: http://127.0.0.1:5000/ratings/
Method: POST
header: Content-Type='application/json'
Body:
{
    "review":"this is a review"
}