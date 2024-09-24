import pandas as pd
from sklearn.linear_model import LinearRegression

def main():
    df_1 = pd.read_csv("/home/test/Desktop/Artifical_SRC/artificial_intelligence/data/housing.tab", delimiter="\t")

    print(df_1.head())
    print(df_1.info())

    model = LinearRegression()
    target = "MEDV"
    feature = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX']

    model.fit(df_1[feature], df_1[target])

    # 결과 출력
    print(model.intercept_)
    print()
    print(model.coef_)

    df_2 = pd.read_csv("/home/test/Desktop/Artifical_SRC/artificial_intelligence/data/housing_predict.tab", delimiter="\t")
    
    fitted = model.predict(df_2[feature])
    print(fitted)

if __name__ == "__main__":
    main()