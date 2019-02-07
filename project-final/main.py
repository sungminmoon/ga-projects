from flask import Flask, render_template
app = Flask(__name__)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('09-19-Amazon-Ranking-Analysis.csv')
df.set_index('Index', inplace=True)
df.ScrapeDate = pd.to_datetime(df.ScrapeDate)
df.fillna("No Information",inplace=True)
df.drop(columns='BBScrapedPrice',inplace=True)
df['BBGroup'] = df.ProductName.factorize()[0]
df['VendorTypeNum'] = df.ScrapedIndexVendorType.map({'Amazon':0, 'FBA':1, 'Other':2})
df['BBVendorTypeNum'] = df.BBVendorType.map({'Amazon':0, 'FBA':1, 'O':2})
df['BBVendorNum'] = df.BBVendor.factorize()[0]
df['VendorNum'] = df.ScrapedIndexVendor.factorize()[0]
list_vendors = df.ScrapedIndexVendor.value_counts().index
df_html = df.to_html()



import pickle 

with open('project_rfr.pkl', 'rb') as picklefile:
	PREDICTOR = pickle.load(picklefile)

import flask
@app.route('/table')
def table():
	return render_template('table.html', table = df_html)

@app.route('/')
def page():
	with open("page.html",'r') as viz_file:
		return viz_file.read()

@app.route('/result', methods=['POST','GET'])
def result():
	if flask.request.method == 'POST':

		inputs = flask.request.form 

		group = inputs['group'][0]
		vendor_text= inputs['vendor']
		vendor = df[df.ScrapedIndexVendor == vendor_text].VendorNum.data[0]
		vendortype_text = inputs['vendortype'][0]
		if vendortype_text == 'Amazon':
			vendortype = 0
		elif vendortype_text == 'FBA':
			vendortype = 1
		else:
			vendortype = 2
		rank = inputs['rank'][0]
		index = inputs['index'][0]

		item = np.array([[group, vendor, vendortype, rank, index]])
		prediction = PREDICTOR.predict(item)
		score = PREDICTOR.oob_score_
		results = {'prediction': prediction[0], 'score': score }
		return flask.jsonify(results)

if __name__ == '__main__':
	HOST = '127.0.0.1'
	PORT = '4000'
	app.run(HOST, PORT, debug = True)

