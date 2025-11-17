


import os
import re
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io

DATA_DIR = '.'
PLAYSTORE_CSV = os.path.join(DATA_DIR, 'googleplaystore.csv')
REVIEWS_CSV = os.path.join(DATA_DIR, 'googleplaystore_user_reviews.csv')

app = Flask(__name__, static_folder='static', template_folder='templates')

# Set style for better looking charts
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']


# ---------------------------
# LOAD + CLEAN DATA
# ---------------------------
def load_data():
    apps = pd.read_csv(PLAYSTORE_CSV, low_memory=False)
    reviews = pd.read_csv(REVIEWS_CSV, low_memory=False)
    return apps, reviews


def clean_apps_df(df):
    d = df.copy()

    # Clean Installs
    if "Installs" in d.columns:
        d["Installs"] = (
            d["Installs"].astype(str)
            .str.replace(r"[+,]", "", regex=True)
            .str.replace("Free", "0", regex=False)
        )
        d["Installs"] = pd.to_numeric(d["Installs"], errors="coerce").fillna(0)

    # Clean Rating
    if "Rating" in d.columns:
        d["Rating"] = pd.to_numeric(d["Rating"], errors="coerce")

    # Clean Reviews
    if "Reviews" in d.columns:
        d["Reviews"] = (
            d["Reviews"].astype(str)
            .str.replace(",", "")
        )
        d["Reviews"] = pd.to_numeric(d["Reviews"], errors="coerce").fillna(0)

    # Clean Price
    if "Price" in d.columns:
        d["Price"] = (
            d["Price"]
            .astype(str)
            .str.replace("$", "")
            .str.replace("Free", "0")
        )
        d["Price"] = pd.to_numeric(d["Price"], errors="coerce").fillna(0)

    # Clean Size
    if "Size" in d.columns:
        d["Size_MB"] = d["Size"].apply(convert_size_to_mb)

    d["Category"] = d["Category"].fillna("Unknown")
    
    # Remove duplicates - keep the one with most reviews
    d = d.sort_values('Reviews', ascending=False).drop_duplicates(subset=['App'], keep='first')

    return d


def convert_size_to_mb(size_str):
    """Convert app size to MB"""
    try:
        size_str = str(size_str).strip()
        if size_str == 'Varies with device' or size_str == 'nan':
            return None
        elif 'M' in size_str:
            return float(size_str.replace('M', ''))
        elif 'k' in size_str or 'K' in size_str:
            return float(size_str.replace('k', '').replace('K', '')) / 1024
        else:
            return None
    except:
        return None


def find_app_row(df, name):
    """Find an app by name (exact or partial)."""
    name = name.lower().strip()

    exact = df[df["App"].str.lower() == name]
    if len(exact) > 0:
        return exact.iloc[0]

    contains = df[df["App"].str.lower().str.contains(name, na=False, regex=False)]
    if len(contains) > 0:
        return contains.iloc[0]

    return None


# ---------------------------
# ENHANCED CHART GENERATION WITH BETTER SPACING
# ---------------------------
def generate_enhanced_chart(target, similar_apps, cat_stats):
    """Generate beautiful comparison charts with better spacing"""
    
    # Prepare data
    app_names = [target["App"][:20]] + [a["App"][:20] for a in similar_apps[:9]]
    ratings = [target["Rating"] or 0] + [a["Rating"] or 0 for a in similar_apps[:9]]
    installs = [target["Installs"] or 0] + [a["Installs"] or 0 for a in similar_apps[:9]]
    reviews = [target["Reviews"] or 0] + [a["Reviews"] or 0 for a in similar_apps[:9]]
    
    # Color palette - highlight target app
    colors = ['#667eea'] + ['#95a5f0'] * 9
    
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 1, hspace=0.35)
    
    # 1. Rating Comparison
    ax1 = fig.add_subplot(gs[0])
    bars1 = ax1.barh(range(len(app_names)), ratings, color=colors, edgecolor='white', linewidth=1.5)
    ax1.set_yticks(range(len(app_names)))
    ax1.set_yticklabels(app_names, fontsize=11, weight='bold')
    ax1.set_xlabel('Rating (out of 5)', fontsize=12, weight='bold', labelpad=10)
    ax1.set_title('⭐ Rating Comparison', fontsize=15, weight='bold', pad=20)
    ax1.axvline(cat_stats['Rating'], color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Category Avg: {cat_stats["Rating"]:.2f}')
    ax1.set_xlim(0, 5)
    ax1.legend(fontsize=10, loc='lower right')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, ratings)):
        if val > 0:
            ax1.text(val + 0.1, bar.get_y() + bar.get_height()/2, f'{val:.1f}', 
                    va='center', fontsize=10, weight='bold')
    
    # 2. Installs Comparison (Log scale for better visualization)
    ax2 = fig.add_subplot(gs[1])
    bars2 = ax2.barh(range(len(app_names)), installs, color=colors, edgecolor='white', linewidth=1.5)
    ax2.set_yticks(range(len(app_names)))
    ax2.set_yticklabels(app_names, fontsize=11, weight='bold')
    ax2.set_xlabel('Number of Installs', fontsize=12, weight='bold', labelpad=10)
    ax2.set_title('📥 Installs Comparison', fontsize=15, weight='bold', pad=20)
    ax2.set_xscale('log')
    ax2.axvline(cat_stats['Installs'], color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Category Avg: {cat_stats["Installs"]:,.0f}')
    ax2.legend(fontsize=10, loc='lower right')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels with better formatting
    for i, (bar, val) in enumerate(zip(bars2, installs)):
        if val > 0:
            label = format_number(val)
            ax2.text(val * 1.5, bar.get_y() + bar.get_height()/2, label, 
                    va='center', fontsize=9)
    
    # 3. Reviews Comparison (Log scale)
    ax3 = fig.add_subplot(gs[2])
    bars3 = ax3.barh(range(len(app_names)), reviews, color=colors, edgecolor='white', linewidth=1.5)
    ax3.set_yticks(range(len(app_names)))
    ax3.set_yticklabels(app_names, fontsize=11, weight='bold')
    ax3.set_xlabel('Number of Reviews', fontsize=12, weight='bold', labelpad=10)
    ax3.set_title('💬 Reviews Comparison', fontsize=15, weight='bold', pad=20)
    ax3.set_xscale('log')
    ax3.axvline(cat_stats['Reviews'], color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Category Avg: {cat_stats["Reviews"]:,.0f}')
    ax3.legend(fontsize=10, loc='lower right')
    ax3.grid(axis='x', alpha=0.3)
    
    # Add value labels with better formatting
    for i, (bar, val) in enumerate(zip(bars3, reviews)):
        if val > 0:
            label = format_number(val)
            ax3.text(val * 1.5, bar.get_y() + bar.get_height()/2, label, 
                    va='center', fontsize=9)
    
    plt.suptitle('📊 App Performance Analysis', fontsize=18, weight='bold', y=0.995)
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=150, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    plt.close(fig)

    return encoded


def format_number(num):
    """Format numbers for display"""
    if num >= 1000000000:
        return f'{num / 1000000000:.1f}B'
    if num >= 1000000:
        return f'{num / 1000000:.1f}M'
    if num >= 1000:
        return f'{num / 1000:.1f}K'
    return f'{num:,.0f}'


def analyze_strengths(app, cat_stats):
    """Analyze app strengths and weaknesses with detailed insights"""
    strengths, weaknesses = [], []

    for metric in ["Rating", "Installs", "Reviews"]:
        app_val = app.get(metric) or 0
        avg_val = cat_stats.get(metric) or 0
        
        if avg_val > 0:
            diff_percent = ((app_val - avg_val) / avg_val) * 100
            
            if app_val >= avg_val:
                strengths.append({
                    "metric": metric,
                    "value": app_val,
                    "avg": avg_val,
                    "diff_percent": diff_percent
                })
            else:
                weaknesses.append({
                    "metric": metric,
                    "value": app_val,
                    "avg": avg_val,
                    "diff_percent": diff_percent
                })

    return strengths, weaknesses


def get_sentiment_analysis(reviews_df, app_name):
    """Get sentiment analysis from user reviews"""
    app_reviews = reviews_df[reviews_df['App'] == app_name]
    
    if len(app_reviews) == 0:
        return None
    
    sentiment_data = {
        'Positive': [],
        'Negative': [],
        'Neutral': []
    }
    
    for _, row in app_reviews.iterrows():
        sentiment = row.get('Sentiment', 'Neutral')
        review_text = row.get('Translated_Review', 'No review text')
        
        if pd.notna(review_text) and str(review_text).strip() and str(review_text) != 'nan':
            sentiment_data[sentiment].append(str(review_text))
    
    sentiment_counts = {k: len(v) for k, v in sentiment_data.items()}
    total = sum(sentiment_counts.values())
    
    if total == 0:
        return None
    
    sentiment_percentages = {k: (v/total)*100 for k, v in sentiment_counts.items()}
    
    return {
        'counts': sentiment_counts,
        'percentages': sentiment_percentages,
        'total_reviews': total,
        'reviews': sentiment_data
    }


def get_category_insights(apps_df, category):
    """Get insights about the app category"""
    cat_apps = apps_df[apps_df['Category'] == category]
    
    insights = {
        'total_apps': len(cat_apps),
        'avg_rating': cat_apps['Rating'].mean(),
        'avg_installs': cat_apps['Installs'].mean(),
        'avg_reviews': cat_apps['Reviews'].mean(),
        'free_apps_percent': (len(cat_apps[cat_apps['Price'] == 0]) / len(cat_apps)) * 100,
        'top_rated_app': cat_apps.nlargest(1, 'Rating').iloc[0]['App'] if len(cat_apps) > 0 else None,
        'most_installed_app': cat_apps.nlargest(1, 'Installs').iloc[0]['App'] if len(cat_apps) > 0 else None,
    }
    
    return insights


def get_category_apps(apps_df, category, metric='Rating', limit=20):
    """Get top apps in a category by metric"""
    cat_apps = apps_df[apps_df['Category'] == category]
    top_apps = cat_apps.nlargest(limit, metric)
    
    results = []
    for _, row in top_apps.iterrows():
        results.append({
            'App': row['App'],
            'Rating': float(row['Rating']) if not pd.isna(row['Rating']) else None,
            'Installs': int(row['Installs']) if not pd.isna(row['Installs']) else None,
            'Reviews': int(row['Reviews']) if not pd.isna(row['Reviews']) else None,
            'Price': float(row['Price']) if not pd.isna(row['Price']) else 0,
        })
    
    return results


# Load dataset
apps_df_raw, reviews_df = load_data()
apps_df = clean_apps_df(apps_df_raw)


# ---------------------------
# ROUTES
# ---------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.json
        app_name = data.get("app_name", "").strip()

        if not app_name:
            return jsonify({"error": "Please provide an app name"}), 400

        row = find_app_row(apps_df, app_name)
        if row is None:
            return jsonify({"error": f"App '{app_name}' not found. Try searching for popular apps like Instagram, WhatsApp, or Facebook."}), 404

        # Create target object
        target = {
            "App": str(row["App"]),
            "Category": row["Category"],
            "Rating": float(row["Rating"]) if not pd.isna(row["Rating"]) else None,
            "Installs": int(row["Installs"]) if not pd.isna(row["Installs"]) else None,
            "Reviews": int(row["Reviews"]) if not pd.isna(row["Reviews"]) else None,
            "Size": str(row.get("Size", "N/A")),
            "Type": str(row.get("Type", "N/A")),
            "Price": float(row["Price"]) if not pd.isna(row["Price"]) else 0,
            "Content_Rating": str(row.get("Content Rating", "N/A")),
            "Genres": str(row.get("Genres", "N/A")),
        }

        # Filter category and remove duplicates
        category_df = apps_df[apps_df["Category"] == row["Category"]].copy()
        category_df = category_df[category_df["App"] != target["App"]]

        if len(category_df) == 0:
            return jsonify({"error": "No similar apps found in this category"}), 404

        # Compute similarity with normalization
        cat_feats = category_df[["Rating", "Installs", "Reviews"]].fillna(0)
        
        # Normalize features for better similarity calculation
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        cat_feats_normalized = scaler.fit_transform(cat_feats)
        
        target_vec = np.array([
            target["Rating"] or 0,
            target["Installs"] or 0,
            target["Reviews"] or 0,
        ]).reshape(1, -1)
        target_vec_normalized = scaler.transform(target_vec)
        
        category_df["similarity"] = np.linalg.norm(
            cat_feats_normalized - target_vec_normalized, axis=1
        )

        # Get top 10 similar apps
        top_similar_df = category_df.sort_values("similarity").head(10)

        top_similar = []
        for _, r in top_similar_df.iterrows():
            top_similar.append({
                "App": r["App"],
                "Rating": float(r["Rating"]) if not pd.isna(r["Rating"]) else None,
                "Installs": int(r["Installs"]) if not pd.isna(r["Installs"]) else None,
                "Reviews": int(r["Reviews"]) if not pd.isna(r["Reviews"]) else None,
            })

        # Compute category stats
        cat_stats = {
            "Rating": float(category_df["Rating"].dropna().mean()),
            "Installs": float(category_df["Installs"].dropna().mean()),
            "Reviews": float(category_df["Reviews"].dropna().mean()),
        }

        strengths, weaknesses = analyze_strengths(target, cat_stats)
        
        # Get sentiment analysis
        sentiment = get_sentiment_analysis(reviews_df, target["App"])
        
        # Get category insights
        category_insights = get_category_insights(apps_df, row["Category"])

        # Generate enhanced chart
        chart_base64 = generate_enhanced_chart(target, top_similar, cat_stats)

        return jsonify({
            "target": target,
            "similar_apps": top_similar,
            "category_stats": cat_stats,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "sentiment": sentiment,
            "category_insights": category_insights,
            "chart": chart_base64
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route("/category_top_apps", methods=["GET"])
def category_top_apps():
    """Get top apps in a category by a specific metric"""
    category = request.args.get('category', '')
    metric = request.args.get('metric', 'Rating')
    
    if not category:
        return jsonify({"error": "Category is required"}), 400
    
    apps = get_category_apps(apps_df, category, metric, limit=20)
    return jsonify({"apps": apps, "metric": metric})


@app.route("/categories", methods=["GET"])
def get_categories():
    """Get list of all categories"""
    categories = sorted(apps_df['Category'].unique().tolist())
    return jsonify({"categories": categories})


@app.route("/search", methods=["GET"])
def search_apps():
    """Search for apps by name"""
    query = request.args.get('q', '').lower().strip()
    
    if not query or len(query) < 2:
        return jsonify({"apps": []})
    
    matching_apps = apps_df[
        apps_df['App'].str.lower().str.contains(query, na=False, regex=False)
    ].head(10)
    
    results = [{"name": app, "category": cat} 
               for app, cat in zip(matching_apps['App'], matching_apps['Category'])]
    
    return jsonify({"apps": results})


if __name__ == "__main__":
    app.run(debug=True, port=5000)

