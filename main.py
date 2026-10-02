import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from ucimlrepo import fetch_ucirepo 
  

#Main
def main():
    print("ML Proj")

    # fetch dataset 
    automobile = fetch_ucirepo(id=10) 
  
    # data (as pandas dataframes) 
    X = automobile.data.features 
    y = automobile.data.targets 
    # metadata 
    print(automobile.metadata) 
    # variable information 
    print(automobile.variables) 

#special var
if __name__ == "__main__":
    main()