from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import pandas as pd
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import yake
from transformers import TFAutoModel, AutoTokenizer
import plotly.express as px

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_analysis(file_path, text_column):
    df = pd.read_csv(file_path)
    # Extract text data from the selected column
    sentences = df[text_column].astype(str).tolist()

    # Get embeddings using Hugging Face locally stored TensorFlow model
    embeddings = get_embeddings(sentences)

    # Cluster embeddings using K-Means
    num_clusters = 5  # Define the number of clusters
    cluster_labels = cluster_embeddings(embeddings, num_clusters)

    # Reduce dimensions using t-SNE
    embedded_embeddings = reduce_dimensions(embeddings)

    # Auto-name clusters using YAKE
    cluster_names = automatic_cluster_naming(cluster_labels, sentences)

    return sentences, cluster_labels, embedded_embeddings, cluster_names

def get_embeddings(sentences):
    tokenizer = AutoTokenizer.from_pretrained("./models/distilbert-base-uncased-mnli", from_tf=True)
    model = TFAutoModel.from_pretrained("./models/distilbert-base-uncased-mnli")

    inputs = tokenizer(sentences, return_tensors="tf", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    embeddings = np.array(outputs.last_hidden_state)
    
    return embeddings

def cluster_embeddings(embeddings, num_clusters):
    # Reshape the embeddings to make them two-dimensional
    num_samples, embedding_dim = embeddings.shape[0], embeddings.shape[1] * embeddings.shape[2]
    flattened_embeddings = embeddings.reshape(num_samples, embedding_dim)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters)
    cluster_labels = kmeans.fit_predict(flattened_embeddings)
    return cluster_labels

def reduce_dimensions(embeddings):
    # Reshape the embeddings to make them two-dimensional
    num_samples, embedding_dim = embeddings.shape[0], embeddings.shape[1] * embeddings.shape[2]
    flattened_embeddings = embeddings.reshape(num_samples, embedding_dim)

    # Perform t-SNE to reduce dimensions to 2
    tsne = TSNE(n_components=2)
    embedded_embeddings = tsne.fit_transform(flattened_embeddings)
    return embedded_embeddings

def automatic_cluster_naming(cluster_labels, sentences):
    extractor = yake.KeywordExtractor(lan="en", n=1, top=1)
    cluster_names = {}
    for cluster_id in np.unique(cluster_labels):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_text = ' '.join([sentences[i] for i in cluster_indices])
        keywords = extractor.extract_keywords(cluster_text)
        cluster_names[cluster_id] = keywords[0][0]
    return cluster_names

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            df = pd.read_csv(file_path, encoding='latin-1')
            file_details = {
                'filename': filename,
                'content_type': file.content_type,
                'content_length': os.path.getsize(file_path),
                'columns': df.columns.tolist(),
                'data': df.values.tolist(),
                'numerical_columns': [col for col in df.columns if df[col].dtype in ['int64', 'float64']],
                'categorical_columns': [col for col in df.columns if df[col].dtype == 'object']
            }
            sample_data = df['phrases'].tolist()
            embeddings = get_embeddings(sample_data)
            cluster_labels = cluster_embeddings(embeddings, num_clusters=3)
            embedded_embeddings = reduce_dimensions(embeddings)
            cluster_names = automatic_cluster_naming(cluster_labels, sample_data)
            return render_template('upload.html', sample_data=sample_data, 
                                   cluster_labels=cluster_labels, embedded_embeddings=embedded_embeddings,
                                   cluster_names=cluster_names, file_details=file_details)
    return render_template('upload.html')

@app.route('/analysis')
def analysis():
    file_path = request.args.get('file_path')
    text_column = request.args.get('text_column')
    sentences, cluster_labels, embedded_embeddings, cluster_names = process_analysis('./uploads/mcdonalds_sri_lanka_reviews.csv', 'phrases')

    x_data = [x[0] for x in embedded_embeddings]
    y_data = [x[1] for x in embedded_embeddings]
    hover_text = [f"Sentence : {sentence} <br>Cluster Name : {cluster_name}" for sentence, cluster_name in zip(sentences, cluster_names)]

    # # Create the scatter plot
    # scatter_plot = go.Scatter(
    #     x=x_data,
    #     y=y_data,
    #     mode='markers',
    #     name='Scatter Plot',
    #     color=cluster_names,
    #     text=hover_text,  # Set the hover text
    #     hoverinfo='text'  # Show text on hover
    # )

    # # Create the layout
    # layout = go.Layout(
    #     title='Embeddings Scatter Plot',
    #     xaxis=dict(title='X-axis'),
    #     yaxis=dict(title='Y-axis')
    # )

    # Create the figure
    # fig = go.Figure(data=[scatter_plot], layout=layout)
    df = pd.DataFrame()
    df['x_data'] = [x[0] for x in embedded_embeddings]
    df['y_data'] = [x[1] for x in embedded_embeddings]
    df['sentences'] = sentences
    df['cluster_names'] = cluster_names

    fig = px.scatter(df, x = 'x_data', y = 'y_data', color = 'cluster_names')
    # Convert the figure to JSON
    scatter_plot_json = fig.to_json()

    return render_template('analysis.html', sentences=sentences, cluster_labels=cluster_labels,
                            embedded_embeddings=embedded_embeddings, cluster_names=cluster_names, scatter_plot_json=scatter_plot_json,
                            context={"cluster_data":zip(cluster_labels, sentences), "cluster_viz": enumerate(embedded_embeddings)})

if __name__ == '__main__':
    app.run(debug=True)