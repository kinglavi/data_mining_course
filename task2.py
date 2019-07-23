# Imports
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer

# Data Loading:
df = pd.read_csv('https://raw.githubusercontent.com/m-braverman/ta_dm_course_data/master/train3.csv')





# Convert to feature vector
feature_extraction = TfidfVectorizer()
X = feature_extraction.fit_transform(df[f"review_text"].values)

pass
from lime import lime_text
from sklearn.pipeline import make_pipeline
from random import sample

c = make_pipeline(feature_extraction, clf)
print(c.predict_proba([validation[f"final_{text_column}"]]))