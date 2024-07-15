from flask import Flask, render_template, send_file
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
import io
import base64

matplotlib.use('Agg')  # Use 'Agg' backend for rendering in scripts

app = Flask(__name__)

# Define feature weights
feature_weights = {
    'Funding': 1.0,
    'Application-Oriented': 1.0,
    'Demos': 1.0,
    'Industrial Collaborations': 1.0,
    'System Maturity': 1.0,
    'Number of Members': 1.0,
    'Academic Collaborations': 1.0
}


def load_and_process_data(file_path):
    df = pd.read_csv(file_path, index_col=0).transpose()
    involvement_mapping = {
        'Strong': 3,
        'Good': 2,
        'Average': 1,
        'None': 0
    }
    involvement_columns = ['Funding', 'Application-Oriented', 'Demos', 'Industrial Collaborations', 'System Maturity',
                           'Number of Members', 'Academic Collaborations']

    for column in involvement_columns:
        if column in df.columns:
            df[column] = df[column].map(involvement_mapping).fillna(0)

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df, df_scaled


def plot_to_img_tag():
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode()
    return f'data:image/png;base64,{img_base64}'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/heatmap')
def heatmap():
    df, _ = load_and_process_data('data/team_activity.csv')
    plt.figure(figsize=(15, 10))
    sns.heatmap(df, annot=True, cmap='viridis')
    plt.title('Heatmap of Raw Strengths Data')
    img_tag = plot_to_img_tag()
    plt.close()
    return render_template('index.html', img_tag=img_tag)


@app.route('/correlation_matrix')
def correlation_matrix():
    df, _ = load_and_process_data('data/team_activity.csv')
    correlation_matrix = df.corr()
    plt.figure(figsize=(14, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    img_tag = plot_to_img_tag()
    plt.close()
    return render_template('index.html', img_tag=img_tag)


@app.route('/factor_analysis')
def factor_analysis():
    df, df_scaled = load_and_process_data('data/team_activity.csv')
    fa = FactorAnalysis(n_components=2, random_state=42)
    fa_results = fa.fit_transform(df_scaled)
    fa_df = pd.DataFrame(data=fa_results, columns=['Factor1', 'Factor2'])
    fa_df['Team'] = df.index
    plt.figure(figsize=(10, 7))
    plt.scatter(fa_df['Factor1'], fa_df['Factor2'])
    for i, txt in enumerate(fa_df['Team']):
        plt.annotate(txt, (fa_df['Factor1'][i], fa_df['Factor2'][i]))
    plt.xlabel('Factor 1')
    plt.ylabel('Factor 2')
    plt.title('Factor Analysis of Team Activities')
    img_tag = plot_to_img_tag()
    plt.close()
    return render_template('index.html', img_tag=img_tag)


@app.route('/performance_distribution')
def performance_distribution():
    df, _ = load_and_process_data('data/team_activity.csv')
    df['Performance Score'] = df.apply(calculate_performance_score, axis=1, weights=feature_weights)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Performance Score'], kde=True, color='blue')
    plt.title('Performance Score Distribution')
    plt.xlabel('Performance Score')
    plt.ylabel('Frequency')
    img_tag = plot_to_img_tag()
    plt.close()
    return render_template('index.html', img_tag=img_tag)


def calculate_performance_score(row, weights):
    score = 0
    for feature, weight in weights.items():
        if feature in row:
            score += row[feature] * weight
    return score


if __name__ == '__main__':
    app.run(debug=True)
