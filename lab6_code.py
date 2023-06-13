import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
from dython.nominal import associations
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split,cross_val_score, LeaveOneOut, GridSearchCV, learning_curve, validation_curve
from sklearn.metrics import  mean_squared_error, mean_absolute_error, r2_score ,recall_score, precision_score
from IPython.core.display import HTML
from sklearn.tree import export_text
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC, NuSVC, LinearSVC, OneClassSVM, SVR, NuSVR, LinearSVR
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from IPython.core.display import HTML
from sklearn.tree import export_text
from IPython.display import Image
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, BaggingRegressor
import graphviz
import pydotplus
from io import StringIO
from heamy.estimator import Regressor, Classifier
from heamy.pipeline import ModelsPipeline
from heamy.dataset import Dataset
from mlxtend.classifier import StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification


st.set_option('deprecation.showPyplotGlobalUse', False)

def ResultsShow(base,predict,):
    return(f"""Mean Squared Error: {str(mean_squared_error(base,predict))}
    Root Mean Squared Error: {str(mean_squared_error(base,predict, squared=False))}
    Mean absolute Error: {str(mean_absolute_error(base, predict))}
    R2_score: {str(r2_score(base, predict))}""")

@st.cache_data
def load_data():
    """
    Загрузка данных
    """
    dataset = pd.read_csv('./diamonds.csv')
    return dataset


@st.cache_data
def preprocess_data(dataset):
    '''
    Масштабирование признаков, функция возвращает X и y для кросс-валидации
    '''
    dataset.cut.replace("b'Fair'", 1, inplace=True)
    dataset.cut.replace("b'Good'", 2, inplace=True)
    dataset.cut.replace("b'Very Good'", 3, inplace=True)
    dataset.cut.replace("b'Ideal'", 4, inplace=True)
    dataset.cut.replace("b'Premium'", 5, inplace=True)
    dataset.clarity.replace("b'I1'", 1, inplace=True)
    dataset.clarity.replace("b'IF'", 2, inplace=True)
    dataset.clarity.replace("b'SI1'", 3, inplace=True)
    dataset.clarity.replace("b'SI2'", 4, inplace=True)
    dataset.clarity.replace("b'VS1'", 5, inplace=True)
    dataset.clarity.replace("b'VS2'", 6, inplace=True)
    dataset.clarity.replace("b'VVS1'", 7, inplace=True)
    dataset.clarity.replace("b'VVS2'", 8, inplace=True)
    dataset.color.replace("b'J'", 1, inplace=True)
    dataset.color.replace("b'I'", 2, inplace=True)
    dataset.color.replace("b'H'", 3, inplace=True)
    dataset.color.replace("b'G'", 4, inplace=True)
    dataset.color.replace("b'F'", 5, inplace=True)
    dataset.color.replace("b'E'", 6, inplace=True)
    dataset.color.replace("b'D'", 7, inplace=True)
    shuffled = dataset.sample(frac=1)
    return shuffled[:1000]

@st.cache_data
def make_split_data(data_in):
    '''
    Масштабирование признаков, функция возвращает X и y для кросс-валидации
    '''
    data_out = data_in.copy()
    data_x = data_out.iloc[:, [0, 1, 2, 3, 4, 5, 7, 8, 9]].values
    data_y = data_out.iloc[:, 6].values
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2,shuffle=True)
    return x_train, x_test, y_train, y_test


def main():
    st.sidebar.header('Метод опорных векторов')
    dataset = load_data()
    fixed = preprocess_data(dataset)
    x_train, x_test, y_train, y_test=make_split_data(fixed)
    svc_kernel = st.sidebar.radio('Ядро:', ('linear', 'rbf', 'poly'))
    num_folds_slider = st.sidebar.slider(
        'Количество фолдов при кросс-валидации":',
        min_value=2,
        max_value=20,
        value=2,
        step=1
    )
    if st.checkbox('Показать корреляционную матрицу'):
        res_corr = associations(
            dataset=dataset,
            nominal_columns=fixed.iloc[:, [0,1,2,3,4,5,7,8,9]].values,
            cmap=sns.diverging_palette(220, 20, as_cmap=True),
            title='Матрица корреляции по всем признакам',
            clustering=True,
            figsize=(28, 28),
            plot=False,
            compute_only=True,
            mark_columns=True,
        )
        corr_matrix = res_corr['corr']
        fig, ax = plt.subplots(figsize=(32, 32))
        sns.heatmap(
            data=corr_matrix,
            square=True,
            center=0,
            cmap=sns.diverging_palette(220, 20, as_cmap=True),
            annot=True,
            vmin=-1.0,
            vmax=1.0
        )
        st.pyplot(fig)


    # Количество строк и столбцов датасета
    num_rows = dataset.shape[0]
    num_cols = dataset.shape[1]
    st.write(f"Количество строк в датасете: {num_rows}")
    st.write(f"Количество столбцов в датасете: {num_cols}")

    # Гиперпараметры метода опорных векторов
    kernel = svc_kernel
    cv = int(num_folds_slider)

    # Обучение
    svc = SVC(kernel=kernel)
    svc.fit(x_train, y_train)

    # Оценка модели
    st.subheader('Оценка качества модели')
    y_pred_test_svc = svc.predict(x_test)
    y_pred_train_svc = svc.predict(x_train)
    st.subheader('На обучающей выборке:')
    st.text(ResultsShow(y_train, y_pred_train_svc))
    st.subheader('На тестовой выборке:')
    st.text(ResultsShow(y_test, y_pred_test_svc))
    scores_svc = cross_val_score(svc, fixed.iloc[:, [0,1,2,3,4,5,7,8,9]].values, fixed.iloc[:, 6].values, cv=cv)
    st.write(f"Оценка accuracy с помощью {cv} фолдной кросс-валидации: {np.mean(scores_svc)}")



if __name__ == '__main__':
    main()
