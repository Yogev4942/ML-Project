import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

#the dataset
from ucimlrepo import fetch_ucirepo 
#Other Files
from model import model

#Main
def main():
    print("ML Proj\n")
    # fetch dataset 
    automobile = fetch_ucirepo(id=10) 
    # data (as pandas dataframes) 
    x = automobile.data.features 
    y = automobile.data.targets 
    d = automobile.data.labels
    print(x.head) 
    print(y.head) 
    print(d.head)

#special var
if __name__ == "__main__":
    main()