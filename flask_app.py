from flask import Flask, request, render_template
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pickle
from sklearn.metrics.pairwise import linear_kernel
import numpy as np

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template("/home.html")

@app.route("/" ,methods=['POST'])
def home():

    jobs_data = pandas.read_csv('https://raw.githubusercontent.com/Nexus-404/object_detection_demo/master/data-6.csv')

    query = request.form['query']

    job = list()
    link = list()
    descript = list()

    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(jobs_data['Description'])
    pickle.dump(tfidf_matrix,open("matrix.npy","wb"))
    pickle.dump(tf.vocabulary_,open("vocabulary.pkl","wb"))
    pickle.dump(tf.idf_,open("idf.pkl","wb"))
    vocabulary = pickle.load(open("vocabulary.pkl", "rb"))
    idfs = pickle.load(open("idf.pkl", "rb"))
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
    tf.vocabulary_ = vocabulary
    tf.idf_ = idfs
    tfidf_matrix = np.load("matrix.npy",allow_pickle=True)
    vector = tf.transform([query])
    cos_sim = linear_kernel(tfidf_matrix,vector)
    res = cos_sim[:,0].argsort()[:-11:-1]

    for i in res:
      job.append(jobs_data['Jobs'][i])
      link.append(jobs_data['Job Url'][i])
      descript.append(jobs_data['Description'][i])

    return render_template("/readpdf.html", jobs = job, links = link, job_description = descript, query = query)
