import pandas as pd
from nltk.probability import FreqDist
from nltk import pos_tag
from wordcloud import WordCloud
import plotly.express as px
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import gensim
from gensim import corpora
import re
import nltk

def call_example_comments() -> pd.DataFrame:
    # creating exmaple data
    comments = {
        'comments': [
            "This is a sample comment about data analysis. Data analysis is a crucial step in any research or business decision-making process. It involves collecting, cleaning, and interpreting data to gain valuable insights. Data analysts use various tools and techniques to uncover patterns and trends in data. In today's data-driven world, data analysis skills are in high demand.",
            "Natural language processing (NLP) is a fascinating field of study. NLP focuses on the interaction between computers and human language. NLP applications include sentiment analysis, machine translation, chatbots, and more. NLP researchers develop algorithms to understand and generate human language. The possibilities in NLP seem endless, and it's an exciting area to explore.",
            "Machine learning is revolutionizing industries across the globe. It's the science of getting computers to learn and act like humans do. Machine learning algorithms are used in recommendation systems, image recognition, autonomous vehicles, and many other areas. As machine learning advances, it continues to shape the future of technology and innovation.",
            "Python is a versatile programming language commonly used in data science and machine learning. Its readability and extensive libraries make it a popular choice among data scientists. Python's simplicity and flexibility make it an excellent language for analyzing and visualizing data. It's no wonder that Python is a go-to language for data professionals.",
            "Artificial intelligence (AI) is a transformative technology with applications in healthcare, finance, and more. AI systems can perform tasks that typically require human intelligence. These systems learn from data, recognize patterns, and make decisions. The growth of AI is expected to drive significant changes in various industries."
        ]
    }
    # dataframe
    df = pd.DataFrame(comments)
    st.markdown("**Supported Formats: CSV, Excel, Text**")
    st.markdown("`comments` column is the subject of analysis.")
    return df


def call_example_replacement() -> pd.DataFrame:
    # creating exmaple data
    replacement = {
        'key': ["sample","this" ],
        'replacement': ["spl","thsi"]}
    # dataframe
    df = pd.DataFrame(replacement)
    st.markdown("`key`,`replacement`: replacement columns ")
    return df


def read_comments_from(data_uploaded, column_name="comments") -> pd.Series:
    df = pd.DataFrame()
    supported_formats = ['.csv', '.xlsx', '.txt']
    if data_uploaded.name.endswith(tuple(supported_formats)):
        if data_uploaded.name.endswith('.csv'):
            df = pd.read_csv(data_uploaded)
        elif data_uploaded.name.endswith('.xlsx'):
            df = pd.read_excel(data_uploaded, engine='openpyxl')
        elif data_uploaded.name.endswith('.txt'):
            df = pd.read_csv(data_uploaded, delimiter='\t')  # Assuming tab-separated text file
    else:
        st.error("This file format is not supported. Please upload a CSV, Excel, or text file.")
        st.stop()
    try:
        comments = df.loc[:, column_name]
    except KeyError:
        comments = df.iloc[:, 0]
    return comments.str.lower()
def read_replacement_from(data_uploaded, column_name=["key","replace"]) -> dict:
    df = pd.DataFrame()
    supported_formats = ['.csv', '.xlsx', '.txt']
    if data_uploaded.name.endswith(tuple(supported_formats)):
        if data_uploaded.name.endswith('.csv'):
            df = pd.read_csv(data_uploaded)
        elif data_uploaded.name.endswith('.xlsx'):
            df = pd.read_excel(data_uploaded, engine='openpyxl')
        elif data_uploaded.name.endswith('.txt'):
            df = pd.read_csv(data_uploaded, delimiter='\t')  # Assuming tab-separated text file
    else:
        st.error("This file format is not supported. Please upload a CSV, Excel, or text file.")
        st.stop()
    try:
        remplacement_dict=  {k:v for k, v in zip(df[column_name[0]].str.lower(), df[column_name[1]].str.lower())}
    except KeyError:
        remplacement_dict = {k:v for k, v in zip(df.iloc[:,0].str.lower(), df.iloc[:,1].str.lower())}
    return remplacement_dict

def prepare_networkg(text) -> "corpus, dictionary":
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    # prepare to topic modeling for ldaLDA
    # extract nouns_adjectives
    nouns_adjectives = [word for word, pos in nltk.pos_tag(filtered_words) if
                        pos in ['NN', 'NNS', 'JJ']]
    # create copus for LDA 
    dictionary = corpora.Dictionary([nouns_adjectives])
    corpus = [dictionary.doc2bow(nouns_adjectives)]
    return corpus, dictionary

def prepare_nouns(comments:pd.Series) ->pd.Series:
    words_list = []

    # nltk data download
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    stop_words = set(stopwords.words('english'))

    for comment in comments:
        tokens = word_tokenize(comment)  # tokenize
        filtered_tokens = [token.lower() for token in tokens if token.isalnum() and token.lower() not in stop_words]
        nouns = [word for (word, tag) in pos_tag(filtered_tokens) if tag.startswith('N')]
        words_list.append(nouns)

    return pd.Series(words_list, name="comments")
    
def prepare_word_freq(nouns) -> pd.DataFrame:
    # nouns frequncy
    noun_counts = FreqDist(nouns)
    df_word_freq = pd.DataFrame(list(noun_counts.items()), columns=['Nouns', 'Frequency'])
    # sorted
    df_word_freq = df_word_freq.sort_values(by='Frequency', ascending=False)
    return df_word_freq


def download_df_as_csv(df: pd.DataFrame, file_name: str, key:str, preview=True, label:str="Download") -> None:

    csv_file = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label,
        csv_file,
        f"{file_name}.csv",
        "text/csv",
        key=key
    )
    # if preview:
    #     st.dataframe(df.head(3))
    return None

def plot_freq(df_word_freq, num_dis: int = 10):
    top_words = df_word_freq.head(num_dis)
    fig = px.bar(top_words, x='Nouns', y='Frequency', title="Top 10 Words Frequency")
    fig.update_xaxes(tickangle=45)
    fig.update_layout(width=800, height=400)
    return st.plotly_chart(fig, use_container_width=True)


def plot_wordcloud(nouns):
    # Create a WordCloud object with the desired settings
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(nouns))
    # Create a Matplotlib figure and axis
    plt.figure(figsize=(8, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    # Display the Matplotlib figure within Streamlit
    st.pyplot(plt, use_container_width=True)


def plot_networkg(corpus, dictionary):
    # LDA model learn
    lda_model = gensim.models.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=10)
    # exteact topic from lda
    topics = lda_model.show_topics(num_topics=2, num_words=8)  # 2 topic & top word 8
    # create graph
    G = nx.Graph()
    # add topic as node
    for topic, words in topics:
        node_label = f"Topic {topic}"
        G.add_node(node_label, node_type='topic')  # add type 'node_type' on tpoc node
        word_list = words.split('+')
        for word in word_list:
            prob, word = word.split('*')
            word = word.strip()
            G.add_node(word, node_type='word')  # add type 'node_type' on word node
            G.add_edge(node_label, word, weight=float(prob))
    # node size define from 100 to 1000 with nomarlize
    node_degrees = dict(G.degree)
    min_size = 100
    max_size = 1000
    node_size = [
        min_size + (max_size - min_size) * (node_degrees[node] - min(node_degrees.values())) / (
                max(node_degrees.values()) - min(node_degrees.values()) + 1) for node in G.nodes]
    # graph viusalize
    pos = nx.spring_layout(G, seed=42)
    edge_width = [data['weight'] * 10 for _, _, data in G.edges(data=True)]
    node_colors = ['lightblue' if G.nodes[node]['node_type'] == 'topic' else 'lightgray' for node in
                   G.nodes]  # set each color to nodes

    plt.figure(figsize=(8, 4))
    nx.draw(G, pos, with_labels=True, node_size=node_size, width=edge_width, node_color=node_colors,
            font_size=8)
    plt.axis('off')
    return st.pyplot(plt, use_container_width=True)