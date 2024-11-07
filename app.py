# 필요한 라이브러리를 불러옵니다.
import pandas as pd
from functions.text import *
from functions.timeseries import *
from functions.multi_numeric import *
from functions.image_analysis import *
from pathlib import Path
import os
import json

# 메인 함수 정의
def main():
    # Streamlit 페이지 설정
    st.set_page_config(page_title="plot stream", layout="wide")

    # 세션 상태 초기화를 위한 변수 리스트
    state_vars = ["tab1", "tab2", "tab3", "tab4", "upload_tab1", "upload_tab1_r", "upload_tab2",
                  "upload_tab3", "upload_tab3_a", "upload_tab4"]

    text_dir = Path("text_dir")
    if not text_dir.exists():
        text_dir.mkdir(parents=True)

    text_r_dir = Path("text_r_dir")
    if not text_r_dir.exists():
        text_r_dir.mkdir(parents=True)

    # 각 변수에 대해 세션 상태를 설정하거나 유지합니다.
    for var in state_vars:
        st.session_state.setdefault(var, None)

    # 페이지 헤더 및 설명
    st.header("Plot Visualization")
    st.write("Relaxation with ☕")
    demo_checkbox_clicked = st.checkbox("Demo")
    dir_check = False

    # 각 탭에 대한 컨테이너를 삽입합니다.
    tab1, tab2, tab3, tab4 = st.tabs(["Text", "Time Series", "Multiple Numerical", "Image Analysis"])

    # Text 탭
    with tab1:
        col1_tab1, col2_tab1 = st.columns([1, 3])
        with col1_tab1:
            st.subheader("1. Data Preparation")
            df_example_comments = call_example_comments()
            df_example_replacement = call_example_replacement()

            # 샘플 데이터 다운로드 링크 생성
            download_df_as_csv(df_example_comments, file_name="sample_text_data", key="download_text_sample_csv",
                               label="Sample download")
            download_df_as_csv(df_example_replacement, file_name="replacement_data",
                               key="download_replacement_sample_csv",
                               label="replacement download")
            st.markdown("---")






            if demo_checkbox_clicked:
                st.session_state["upload_tab1"] = df_example_comments
            else:


                file_list_text = os.listdir(text_dir)
                file_list_text_r = os.listdir(text_r_dir)

                # st.session_state["upload_tab1"] = st.file_uploader("Upload Text data", key="time_text_data")
                # st.session_state["upload_tab1_r"] = st.file_uploader("Upload Text replacement", key="time_text_data_r")

                if file_list_text:
                    selected_file_data = st.selectbox("Select a text data file", file_list_text)
                    file_path_data = os.path.join(text_dir, selected_file_data)
                    st.session_state["upload_tab1"] = file_path_data
                    dir_check = True

                else:
                    # st.warning("No text data files found in the specified directory.")
                    st.session_state["upload_tab1"] = st.file_uploader("Upload Text data", key="time_text_data")

                if file_list_text_r:
                    selected_file_data_r = st.selectbox("Select a text replacement file", file_list_text_r)
                    file_path_data_r = os.path.join(text_r_dir, selected_file_data_r)
                    st.session_state["upload_tab1_r"] = file_path_data_r
                    dir_check = True

                else:
                    # st.warning("No text replacement files found in the specified directory.")
                    st.session_state["upload_tab1_r"] = st.file_uploader("Upload Text replacement", key="time_text_data_r")




            if st.session_state["upload_tab1"] is not None:
                # 다른 탭들의 업로드 상태 초기화
                st.session_state["upload_tab2"] = None
                st.session_state["upload_tab3"] = None
                st.session_state["upload_tab4"] = None

                text_data_uploaded = st.session_state["upload_tab1"]

                try:
                    if demo_checkbox_clicked:
                        comments = st.session_state["upload_tab1"].loc[:, "comments"].str.lower()
                    else:
                        if dir_check:
                            comments = read_comments_from_dir(text_data_uploaded, column_name="comments")
                        else:
                            comments = read_comments_from(text_data_uploaded, column_name="comments")

                    # 텍스트 치환
                    if st.session_state["upload_tab1_r"] is not None:
                        text_replacement_uploaded = st.session_state["upload_tab1_r"]
                        if dir_check:
                            text_replacement_dict = read_replacement_from_dir(text_replacement_uploaded,
                                                                          column_name=["key", "replace"])
                        else:
                            text_replacement_dict = read_replacement_from(text_replacement_uploaded,
                                                                          column_name=["key", "replace"])

                        for k, v in text_replacement_dict.items():
                            comments = comments.str.replace(k, v)

                    if demo_checkbox_clicked:
                        words_ds = pd.read_json("https://raw.githubusercontent.com/xikest/app_plotvisual/main/words_ds.json", typ="series", orient="values")
                    else:
                        words_ds = prepare_nouns(comments)

                    
                    st.dataframe(words_ds[:3])

                    nouns = []
                    for words_list in words_ds:
                        nouns.extend(words_list)

                    df_word_freq = prepare_word_freq(nouns)
                    comments_as_string = ' '.join(comments.astype(str))
                    
                    
                    if demo_checkbox_clicked:
                        with open("https://raw.githubusercontent.com/xikest/app_plotvisual/main/corpus.pkl", 'rb') as f:
                            corpus = pickle.load(f)
                        with open("https://raw.githubusercontent.com/xikest/app_plotvisual/main/dictionary.pkl", 'rb') as f:
                            dictionary = pickle.load(f)
                    else:
                        corpus, dictionary = prepare_networkg(comments_as_string)
                    

                    

                    st.session_state["tab1"] = {"plot_df_word_freq": df_word_freq,
                                                "wordcloud_nouns": nouns,
                                                "network_corpus": corpus,
                                                "network_dictionary": dictionary}

                    st.subheader("2. Analysis results")

                    # 결과 다운로드 버튼
                    download_df_as_csv(df_word_freq, file_name="word_freq_analysis", key="download_csv_text_analysis",
                                       label="Result download")
                    st.dataframe(df_word_freq.head(3))

                except Exception as e:
                    st.write(e)
                    st.error('Please verify the file format', icon="🚨")

        with col2_tab1:
            if st.session_state["tab1"] is not None:
                st.subheader("3. Visualization")
                tab1_col2_tab1, tab2_col2_tab1, tab3_col2_tab1 = st.tabs(["Plot", "Word Cloud", "Network Graph"])
                try:
                    with tab1_col2_tab1:
                        df_word_freq = st.session_state["tab1"]["plot_df_word_freq"]
                        plot_freq(df_word_freq)
                    with tab2_col2_tab1:
                        nouns = st.session_state["tab1"]["wordcloud_nouns"]
                        plot_wordcloud(nouns)
                    with tab3_col2_tab1:
                        corpus = st.session_state["tab1"]["network_corpus"]
                        dictionary = st.session_state["tab1"]["network_dictionary"]
                        plot_networkg(corpus, dictionary)
                except:
                    st.warning("Please upload text data first.")

    # Time Series Analysis
    with tab2:
        col1_tab2, col2_tab2 = st.columns([1, 3])
        with col1_tab2:
            st.subheader("1. Data Preparation")
            df_example_timeseries = call_example_timeseries()
            download_df_as_csv(df_example_timeseries, "sample_timeseries_data", key="download_timeseries_sample_csv",
                               label="Sample download")
            st.markdown("---")

            if demo_checkbox_clicked:
                st.session_state["upload_tab2"] = df_example_timeseries
            else:
                st.session_state["upload_tab2"] = st.file_uploader("Upload Time Series", key="time_series_uploader")

            if st.session_state["upload_tab2"] is not None:
                st.session_state["upload_tab1"] = None
                st.session_state["upload_tab3"] = None
                st.session_state["upload_tab4"] = None
                time_data_uploaded = st.session_state["upload_tab2"]
                try:
                    if demo_checkbox_clicked:
                        timeseries = time_data_uploaded.loc[:, ["date", "timeseries"]]
                        timeseries.loc[:, "timeseries"] = timeseries.loc[:, "timeseries"].astype(float)  # 숫자형으로 변경
                        timeseries = timeseries.set_index("date")
                        timeseries.index = pd.to_datetime(timeseries.index)
                        timeseries = timeseries.resample('D').last().ffill()
                    else:
                        timeseries = read_timeseries_from(time_data_uploaded)

                    st.session_state["tab2"] = {"timeseries": timeseries}
                    plot_time_series(timeseries)
                except:
                    st.error('Please verify the file format', icon="🚨")
        with col2_tab2:
            if st.session_state["tab2"] is not None:
                st.subheader("2. Visualization")
                tab1_col2_tab3, tab2_col2_tab3 = st.tabs(["Prophet Plot", "TimeSeries"])
                timeseries = st.session_state["tab2"]["timeseries"]
                with tab1_col2_tab3:
                    plot_prophet(timeseries)
                with tab2_col2_tab3:
                    plot_timesseries_arima(timeseries)

    # Multiple Numerical 탭
    with tab3:
        col1_tab3, col2_tab3 = st.columns(2)
        with col1_tab3:
            st.subheader("1. Data Preparation")
            df_example_multi_numeric = call_example_multi_numeric()
            download_df_as_csv(df_example_multi_numeric, "sample_multi_numeric_data",
                               key="download_multi_numeric_sample_csv",
                               label="Sample download")
            st.markdown("---")

            if demo_checkbox_clicked:
                st.session_state["upload_tab3"] = df_example_multi_numeric
            else:
                st.session_state["upload_tab3"] = st.file_uploader("Upload numeric data", key="multi_numeric_uploader")

            if st.session_state["upload_tab3"] is not None:
                st.session_state["upload_tab1"] = None
                st.session_state["upload_tab2"] = None
                st.session_state["upload_tab4"] = None
                multi_data_uploaded = st.session_state["upload_tab3"]
                try:
                    st.subheader("2. Build Model")
                    if demo_checkbox_clicked:
                        y = multi_data_uploaded.loc[:, "target"]
                        X = multi_data_uploaded.drop("target", axis=1)
                        df_multi = pd.concat([y, X], axis=1)
                    else:
                        df_multi = read_numeric_from(multi_data_uploaded)

                    y_column = df_multi.columns[0]

                    numerical_columns, categorical_columns = split_data_columns(df_multi.drop([y_column], axis=1))

                    tab1_col1_tab3, tab2_col1_tab3, tab3_col1_tab3, tab4_col1_tab3, tab5_col1_tab3 = st.tabs(
                        ["Missing value", "Numeric Features", "Categorical Features", "Pre-process",
                         "Machine learning"])
                    with tab1_col1_tab3:
                        if is_na(df_multi):
                            col1_col1_tab3, col2_col1_tab3 = st.columns(2)
                            with col1_col1_tab3:
                                plot_missing_value(df_multi)
                            with col2_col1_tab3:
                                df_multi = df_multi.interpolate(method='linear', inplace=True)
                                plot_missing_value(df_multi)
                        else:
                            plot_missing_value(df_multi)
                    with tab2_col1_tab3:
                        tab1_tab2_col1_tab3, tab2_tab2_col1_tab3, tab3_tab2_col1_tab3 = st.tabs(
                            ["Distribution", "Correlation", "Normality"])
                        with tab1_tab2_col1_tab3:
                            plot_distribution(df_multi.drop(categorical_columns, axis=1))
                        with tab2_tab2_col1_tab3:
                            plot_correlation(df_multi.drop(categorical_columns, axis=1))
                        with tab3_tab2_col1_tab3:
                            plot_normality(df_multi.drop(categorical_columns, axis=1))
                    with tab3_col1_tab3:
                        plot_stacked_bar(df_multi[categorical_columns])
                    with tab4_col1_tab3:
                        y = df_multi[y_column]
                        X = preprocess_data(df_multi)
                    with tab5_col1_tab3:
                        best_model_name, best_model, best_mse = select_best_model(X, y)
                        visualize_best_model_performance(X, y, best_model_name, best_model)
                        st.session_state["tab3"] = {"ml_model": best_model}
                except:
                    st.error('Please verify the file format', icon="🚨")
        with col2_tab3:
            if st.session_state["tab3"] is not None:
                st.subheader("3. Actual prediction")

                if demo_checkbox_clicked:
                    st.session_state["upload_tab3_a"] = df_multi
                else:
                    st.session_state["upload_tab3_a"] = st.file_uploader("Upload actual data",
                                                                         key="actual_multi_data_uploaded")

                if st.session_state["upload_tab3_a"] is not None:
                    actual_multi_data_uploaded = st.session_state["upload_tab3_a"]
                    try:
                        st.markdown("---")

                        if demo_checkbox_clicked:
                            df_X = df_multi
                        else:
                            df_X = read_numeric_from(actual_multi_data_uploaded)

                        X_new = preprocess_data(df_X, show=False)
                        best_model = st.session_state["tab3"].get("ml_model")
                        results = model_predictions_and_visual(X_new, best_model)

                        # 결과 다운로드 버튼
                        download_df_as_csv(results, file_name="numerical_results",
                                           key="download_csv_numeric_analysis", label="Result download")

                    except:
                        st.error('Please verify the file format', icon="🚨")

    # Image Analysis 탭
    with tab4:
        col1_tab4, col2_tab4 = st.columns([1, 3])
        with col1_tab4:
            st.subheader("1. Data Preparation")

            # 이미지 샘플 다운로드
            _ = download_image_example()

            if demo_checkbox_clicked:
                st.session_state["upload_tab4"] = download_image_example(demo_mode=True)
                if st.session_state["upload_tab4"] == "":
                    st.session_state["upload_tab4"] = None
            else:
                st.session_state["upload_tab4"] = st.file_uploader("Upload image data", key="Image_uploader",
                                                                   type=["jpg", "jpeg", "png"])

            st.markdown("---")

            if st.session_state["upload_tab4"] is not None:
                st.session_state["upload_tab1"] = None
                st.session_state["upload_tab2"] = None
                st.session_state["upload_tab3"] = None
                image_data_uploaded = st.session_state["upload_tab4"]
                st.image(image_data_uploaded, use_column_width=True)
        with col2_tab4:
            if st.session_state["upload_tab4"] is not None:
                st.subheader("2. Analysis results")
                try:
                    if demo_checkbox_clicked:
                        to_lab_image(image_data_uploaded, byte_type=True)
                    else:
                        to_lab_image(image_data_uploaded)
                except Exception as e:
                    st.write(e)


# 메인 함수 실행
if __name__ == "__main__":
    main()
