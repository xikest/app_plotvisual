import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import statsmodels.api as sm
#
def call_example_multi_numeric():
    data = {
        'target': [500, 550, 600, 480, 530, 520, 650, 690, 600, 550],
        'price': [100, 150, 200, 120, 180, 130, 210, 220, 250, 190],
        'customer': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'A', 'B', 'C'],
        'spec1': [1, 2, 3, 1, 2, 1, 3, 3, 2, 1],
        'spec2': ['Good', 'Excellent', 'Good', 'Excellent', 'Average', 'Good', 'Good', 'Excellent', 'Excellent',
                  'Average'],
        'spec3': ['North', 'South', 'North', 'West', 'South', 'North', 'West', 'North', 'South', 'West'],
        'spec4': [2020, 2021, 2022, 2020, 2021, 2020, 2022, 2022, 2021, 2020],

    }
    df = pd.DataFrame(data)
    st.markdown("**Supported Formats: CSV, Excel**")
    st.markdown("Excel (or CSV) Considerations: `target` column is assigned as the dependent variable.")
    return df
def read_numeric_from(data_uploaded, column_target: str = "target") -> pd.DataFrame:
    df = pd.DataFrame()
    supported_formats = ['.csv', '.xlsx', '.txt']
    if data_uploaded.name.endswith(tuple(supported_formats)):
        if data_uploaded.name.endswith('.csv'):
            df = pd.read_csv(data_uploaded)
        elif data_uploaded.name.endswith('.xlsx'):
            df = pd.read_excel(data_uploaded, engine='openpyxl')
    else:
        st.error("This file format is not supported. Please upload a CSV, Excel, or text file.")
        st.stop()
    try:
        y = df.loc[:, column_target]
        X = df.drop(column_target, axis=1)
    except KeyError:
        y = df.iloc[:, 0]
        X = df.iloc[:, 1:]
    # Check if y is not continuous numerical data
    if not pd.api.types.is_numeric_dtype(y):
        raise ValueError("The 'y' column is not continuous numerical data.")
    df_combined = pd.concat([y, X], axis=1)
    return df_combined
def is_na(df):
    return df.isna().sum().mean() > 0


def plot_missing_value(df):
    fig, ax = plt.subplots(figsize=(10, 4))  # 그래프 크기 지정
    msno.matrix(df, ax=ax, sparkline=False)
    st.pyplot(fig, use_container_width=True)
    return None

def split_data_columns(df):
    numerical_columns = df.select_dtypes(include=['number']).columns    # 숫자 데이터를 포함하는 열 선택
    categorical_columns = df.select_dtypes(exclude=['number']).columns   # 숫자 데이터를 포함하지 않는 열 선택
    return numerical_columns, categorical_columns


def plot_distribution(numerical_data):
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(numerical_data)
    sns.set_style("white")
    fig = plt.figure(figsize=(10, 4))
    for i in range(standardized_data.shape[1]):
        sns.histplot(standardized_data[:, i], kde=True, label=numerical_data.columns[i])
    plt.xlabel("Feature Value")
    plt.ylabel("Density")
    plt.title("Distribution of Numerical Features")
    plt.legend()
    sns.despine()
    st.pyplot(fig, use_container_width=True)


def plot_normality(numerical_data):

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    fig.suptitle("Q-Q Plot for Normality Check (Numerical Features)")

    scaler = StandardScaler()  # StandardScaler 객체 생성

    for i, column in enumerate(numerical_data.columns):
        qq_data = numerical_data[column].dropna().values.reshape(-1, 1)  # 1차원 데이터를 2차원으로 변환
        qq_data = scaler.fit_transform(qq_data)  # 데이터 스케일링 (Z-스코어 표준화)
        qq_data = qq_data.flatten()  # 2차원 데이터를 다시 1차원으로 변환
        sm.qqplot(qq_data, line='s', ax=ax, label=column)
        ax.set_title('Q-Q Plot for Numerical Features')
    ax.legend(loc='upper left')
    st.pyplot(fig, use_container_width=True)


def plot_correlation(numerical_data) -> None:
    correlation_matrix = numerical_data.corr(method="spearman")

    # Create a mask to hide the upper triangle
    mask = np.triu(np.ones_like(correlation_matrix), k=1)

    fig, ax = plt.subplots(figsize=(10, 4))

    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='Blues', cbar=False,
                xticklabels=correlation_matrix.columns, yticklabels=correlation_matrix.columns, mask=mask)

    plt.xticks(rotation=45)
    plt.title('Correlation Map')

    # 그래프에서 그리드를 제거
    ax.set_xticks([])
    ax.set_yticks([])

    st.pyplot(fig, use_container_width=True)
    return None


def plot_stacked_bar(df):
    category_counts = df.apply(lambda x: x.value_counts()).T.fillna(0)
    data = []
    for col in category_counts.columns:
        data.append(go.Bar(x=category_counts.index, y=category_counts[col], name=col))
    layout = go.Layout(
        barmode='stack',
        title='Stacked category',
        xaxis=dict(title='Category'),
        yaxis=dict(title='Cumulative Count'),
        legend=dict(title='Subcategory'),
    )
    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig, use_container_width=True)


def pca_reduction(numerical_data, variance_threshold=0.8, show=True):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numerical_data)
    pca = PCA(n_components=2)  # PC1과 PC2만 유지
    principal_components_pca = pca.fit_transform(scaled_data)
    pca_df = pd.DataFrame(data=principal_components_pca, columns=['PC1', 'PC2'])

    if show:
    # PCA 결과 시각화
        plt.figure(figsize=(10, 4))
        plt.scatter(pca_df['PC1'], pca_df['PC2'])
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA Plot')

        # Streamlit에 그래프 출력
        st.pyplot(plt)

    return pca_df


def preprocess_data(df, show=True):
    # st.markdown("#### Preprocess")
    numerical_columns, categorical_columns = split_data_columns(df)
    df.interpolate(method='linear', inplace=True)
    if not numerical_columns.empty:
        pca_df_num = pca_reduction(df[numerical_columns], show=show)
        preprocessed_df = pd.concat([df[categorical_columns], pca_df_num], axis=1)
    else:
        preprocessed_df = df
    if not categorical_columns.empty:
        preprocessed_df = pd.get_dummies(preprocessed_df, columns=categorical_columns, drop_first=True)
    return preprocessed_df


def make_decision_tree(X_train, y_train):
    decision_tree = DecisionTreeRegressor()
    decision_tree.fit(X_train, y_train)
    return decision_tree


def make_random_forest(X_train, y_train):
    random_forest = RandomForestRegressor()
    random_forest.fit(X_train, y_train)
    return random_forest


def select_best_model(X, y):
    """
    주어진 모델 중에서 가장 낮은 MSE를 가진 모델을 선택
    model_list = [
        ("MLP", mlp_model),  # "MLP"는 모델 이름, mlp_model은 모델 인스턴스
        ("Decision Tree", decision_tree_model),
        ("Random Forest", random_forest_model)
    ]
    Parameters:
    - X: 입력 데이터
    - y: 목표 변수
    - models: 모델 리스트, 각 모델은 (model_name, model_instance) 형태.

    Returns:
    - best_model: 가장 낮은 MSE를 가진 모델의 이름
    - best_mse: 가장 낮은 MSE 값
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    models_dict = {
        "Decision Tree": make_decision_tree(X_train, y_train),
        "Random Forest": make_random_forest(X_train, y_train)
    }
    mse_scores = []

    # Streamlit의 progress bar를 생성
    progress_bar = st.progress(0)

    for i, (model_name, model_instance) in enumerate(models_dict.items()):
        # 진행 상황을 업데이트하고 시각화
        progress = (i + 1) / len(models_dict)
        progress_bar.progress(progress)

        model = model_instance
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mse_scores.append((model_name, mse))

    best_model_name, best_mse = min(mse_scores, key=lambda x: x[1])
    best_model = models_dict.get(best_model_name)

    # 진행 상황을 100%로 설정하여 완료
    progress_bar.progress(1.0)

    return best_model_name, best_model, best_mse

def visualize_best_model_performance(X, y, best_model_name, best_model_instance):
    st.markdown("#### Best model")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    best_model_instance.fit(X_train, y_train)
    predictions = best_model_instance.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    fig = plt.figure(figsize=(10, 4))
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"{best_model_name} Model Performance (MSE: {round(mse,2)})")
    st.pyplot(fig, use_container_width=True)

def model_predictions_and_visual(new_data, best_model):
    """
    새로운 입력 데이터와 모델 예측 결과를 함께 시각화합니다.

    Parameters:
    - new_data: 새로운 입력 데이터
    - predictions: 모델의 예측 결과
    """
    predictions = best_model.predict(new_data)
    predictions_df = pd.DataFrame({'Predicted': predictions})
    result_df = new_data.join(predictions_df, how='inner')
    # Plotly를 사용하여 scatter plot 생성
    # Plotly를 사용하여 scatter plot 생성 (로그 스케일)
    fig = px.scatter(result_df, x=result_df.index, y='Predicted', title='Predicted Values',
                     labels={'index': 'Index', 'Predicted': 'Predicted Values'},
                     color_discrete_sequence=['red'])

    # x-축과 y-축을 로그 스케일로 설정
    fig.update_xaxes(type='log')
    fig.update_yaxes(type='log')

    # Streamlit에 그래프 출력
    st.plotly_chart(fig, use_container_width=True)
    return result_df
