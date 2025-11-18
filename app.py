


import os
import re
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import base64
import io
# Suppress emoji font warnings
import warnings
warnings.filterwarnings('ignore', message='Glyph.*missing from font')

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

def caluculate_success_scores(app, category_stats, app_df):
    """
    Calculate a comprehensive success scores (0-100) based on multiple factors.
    Score Breakdown Factors:
    1. App Rating (20%): How well users rate the application (Scale: 1-5)
    2. Installations (25%): Market penetration and adoption; how many users download and use the application
    3. Reviews (25%): User feedback, rejection and overall engagement;
    4. Category Rank (15%): Application's relative standing/positioning within the category
    5. Price Strategy (15%): Application's accessibility and monetization strategy
    """
    score = 0

    #1. Rating Score (0-20 points)
    if app.get("Rating") is not None:
        rating = float(app['Rating'])
        rating_score = (rating / 5.0) * 20
        score += rating_score

    #2. Installations Score (0-25 points)
    if category_stats.get('Installs') and app.get('Installs'):
        app_installs = float(app['Installs'])
        category_avg_installs = float(category_stats['Installs'])

        # Log scale to prevent extreme outliers
        installs_ratio = min(app_installs / category_avg_installs, 1000)
        installs_score = min((np.log1p(installs_ratio) / np.log1p(1000)) * 25, 25)
        score += installs_score

    #3. Review Score (0-25 points)
    if category_stats.get('Reviews') and app.get('Reviews'):
        app_reviews = float(app['Reviews'])
        category_avg_reviews = float(category_stats['Reviews'])

        # Log scale for reviews
        review_ratio = min(app_reviews / category_avg_reviews, 1000)
        review_score = min((np.log1p(review_ratio) / np.log1p(1000)) * 25, 25)
        score += review_score

    #4. Category Rank Score (0-15 points)
    category = app.get('Category')
    if category:
        category_apps = apps_df[apps_df['Category'] == category]
        category_apps_copy = category_apps.copy()
        category_apps_copy['rank_score'] = (
        (category_apps_copy['Rating'].fillna(0) / 5.0) +
        (np.log1p(category_apps_copy['Installs']) / np.log1p(category_apps_copy['Installs'].max())) +
        (np.log1p(category_apps_copy['Reviews']) / np.log1p(category_apps_copy['Reviews'].max()))
        )

        rank_position = category_apps_copy['rank_score'].rank(ascending=False)
        app_rank = rank_position[category_apps_copy['App'] == app['App']].values

        if len(app_rank) > 0:
            percentile = (1 - (app_rank[0] / len(category_apps))) * 100
            rank_score = (percentile / 100) * 15
            score += rank_score

    #5. Price Strategy Score (0-15 points)
    price = float(app.get('Price', 0))
    # Free applications get full points
    if price == 0:
        score += 15
    else:
        # Paid applications receive partial credits (up to 10 points)
        score += 10

    # Round the score to 2 decimal places
    return round(max(0, min(100, score)), 2)

def get_success_insights(score):
    """
    Provide insights based on the success score (0-100)
    """
    if score >= 90:
        tier = "🟢 Exceptional"
        description = "This application is a top performer with excellent metrics across the board."
        recommendation = "Maintain quality and consider expanding features to attract more users."
    elif score >= 80 and score < 90:
        tier = "🔵 Strong"
        description = "This application performs well above average, but could be improved further."
        recommendation = "Focus on retention and user engagement, and consider adding new features to attract more users."
    elif score >= 55 and score < 80:
        tier = "🟡 Good"
        description = "This application meets category standards, but could be improved further."
        recommendation = "Identify specific areas to improve (rating, installs, or engagement)"
    elif score >= 40 and score < 55:
        tier = "🟠 Moderate"
        description = "This application has potential but needs improvement."
        recommendation = "Address user feedback and improve quality metrics."
    else:
        tier = "🔴 Needs Attention"
        description = "This application lags behind category standards and needs attention."
        recommendation = "Major improvements needed: quality, marketing, or features."

    return {
        "tier": tier,
        "description": description,
        "recommendation": recommendation,
        "score": score,
    }


def analyze_competitive_positioning(target, similar_apps, cat_stats):
    """
    Analyze app positioning in 2x2 matrix:
    - X-axis: Rating (Quality)
    - Y-axis: Installs (Popularity)
    Quadrants:
    - Top Right: Leaders (High quality, high popularity)
    - Top Left: Stars (High quality, lower popularity) - opportunity!
    - Bottom Right: Potential (Lower quality, high popularity) - needs improvement
    - Bottom Left: Niche (Lower quality, lower popularity)
    """
    # Get category averages with proper error handling
    cat_avg_rating = float(cat_stats.get('Rating', 3.5)) if cat_stats.get('Rating') else 3.5
    cat_avg_installs = float(cat_stats.get('Installs', 1000000)) if cat_stats.get('Installs') else 1000000

    # Determine target position
    try:
        target_rating = float(target.get('Rating', 0)) if target.get('Rating') else 0
        target_installs = float(target.get('Installs', 0)) if target.get('Installs') else 0
    except (ValueError, TypeError):
        target_rating = 0
        target_installs = 0

    # Determine target position BEFORE using it
    target_position = "Niche"  # Default position

    if target_rating >= cat_avg_rating and target_installs >= cat_avg_installs:
        target_position = "Leader"
    elif target_rating >= cat_avg_rating and target_installs < cat_avg_installs:
        target_position = "Star"
    elif target_rating < cat_avg_rating and target_installs >= cat_avg_installs:
        target_position = "Potential"
    else:
        target_position = "Niche"

    # Analyze similar apps positioning
    positioning_data = {
        'target': {
            'name': str(target.get('App', 'Unknown')),
            'rating': target_rating,
            'installs': target_installs,
            'position': target_position,
            'category': str(target.get('Category', 'Unknown'))
        },
        'similar_apps': [],
        'quadrant_summary': {
            'Leaders': 0,
            'Stars': 0,
            'Potential': 0,
            'Niche': 0
        }
    }

    # Analyze each similar app
    for app in similar_apps[:10]:
        try:
            app_rating = float(app.get('Rating', 0)) if app.get('Rating') else 0
            app_installs = float(app.get('Installs', 0)) if app.get('Installs') else 0
        except (ValueError, TypeError):
            app_rating = 0
            app_installs = 0

        # Determine position for this app
        if app_rating >= cat_avg_rating and app_installs >= cat_avg_installs:
            position = "Leader"
        elif app_rating >= cat_avg_rating and app_installs < cat_avg_installs:
            position = "Star"
        elif app_rating < cat_avg_rating and app_installs >= cat_avg_installs:
            position = "Potential"
        else:
            position = "Niche"

        positioning_data['similar_apps'].append({
            'name': str(app.get('App', 'Unknown'))[:25],
            'rating': app_rating,
            'installs': app_installs,
            'reviews': int(app.get('Reviews', 0)) if app.get('Reviews') else 0,
            'position': position
        })

        # CRITICAL FIX: Only increment if position is in the dict
        if position in positioning_data['quadrant_summary']:
            positioning_data['quadrant_summary'][position] += 1
        else:
            # Fallback: add position to dict if missing
            positioning_data['quadrant_summary'][position] = 1

    # Generate recommendations
    if target_position == "Leader":
        recommendation = "You're a market leader! Focus on maintaining quality and innovation."
    elif target_position == "Star":
        recommendation = "You have high quality but lower visibility. Invest in marketing and user acquisition."
    elif target_position == "Potential":
        recommendation = "You're popular but have quality concerns. Focus on user experience and ratings."
    else:  # Niche
        recommendation = "Room for growth. Improve quality (rating) and expand user base (installs)."

    positioning_data['recommendation'] = recommendation

    return positioning_data

# In app.py, add after analyze_competitive_positioning()

def analyze_price_strategy(target, apps_df, category):
    """Analyze pricing impact on app success"""

    category_apps = apps_df[apps_df['Category'] == category].copy()

    # Split by price
    free_apps = category_apps[category_apps['Price'] == 0]
    paid_apps = category_apps[category_apps['Price'] > 0]

    # Calculate metrics
    price_analysis = {
        'target': {
            'price': float(target.get('Price', 0)),
            'is_free': float(target.get('Price', 0)) == 0
        },
        'free_apps': {
            'count': len(free_apps),
            'percentage': (len(free_apps) / len(category_apps)) * 100 if len(category_apps) > 0 else 0,
            'avg_rating': float(free_apps['Rating'].mean()) if len(free_apps) > 0 else 0,
            'avg_installs': float(free_apps['Installs'].mean()) if len(free_apps) > 0 else 0,
            'avg_reviews': float(free_apps['Reviews'].mean()) if len(free_apps) > 0 else 0
        },
        'paid_apps': {
            'count': len(paid_apps),
            'percentage': (len(paid_apps) / len(category_apps)) * 100 if len(category_apps) > 0 else 0,
            'avg_rating': float(paid_apps['Rating'].mean()) if len(paid_apps) > 0 else 0,
            'avg_installs': float(paid_apps['Installs'].mean()) if len(paid_apps) > 0 else 0,
            'avg_reviews': float(paid_apps['Reviews'].mean()) if len(paid_apps) > 0 else 0,
            'avg_price': float(paid_apps['Price'].mean()) if len(paid_apps) > 0 else 0
        }
    }

    # Recommendation
    if target.get('Rating'):
        app_rating = float(target['Rating'])
        if price_analysis['target']['is_free']:
            if app_rating >= price_analysis['free_apps']['avg_rating']:
                recommendation = "Good choice! Free apps average lower ratings. Your quality stands out."
            else:
                recommendation = "Consider improving quality to match category average for free apps."
        else:
            target_price = price_analysis['target']['price']
            if target_price > price_analysis['paid_apps']['avg_price']:
                recommendation = f"Your price (${target_price:.2f}) is above category average. Consider lowering."
            else:
                recommendation = f"Your price point is competitive."
    else:
        recommendation = "No rating data available."

    price_analysis['recommendation'] = recommendation

    # Revenue potential estimate
    if target.get('Installs'):
        if price_analysis['target']['is_free']:
            estimated_revenue = (float(target['Installs']) * 0.30 * 0.50) / 1000
            revenue_note = f"Free app - Est. revenue: ${estimated_revenue:,.0f}/month"
        else:
            conversion_rate = 0.01
            estimated_revenue = (float(target['Installs']) * 0.30 * conversion_rate * price_analysis['target']['price'])
            revenue_note = f"Paid app - Est. revenue: ${estimated_revenue:,.0f}"

        price_analysis['revenue_estimate'] = revenue_note
    else:
        price_analysis['revenue_estimate'] = "Insufficient data"

    return price_analysis  # CRITICAL: MUST RETURN THE ANALYSIS DICT


def generate_price_comparison_chart(price_analysis):
    """Generate comparison chart for free vs paid apps"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    categories = ['Free Apps', 'Paid Apps']
    # Rating Comparison
    ratings = [
        price_analysis['free_apps']['avg_rating'],
        price_analysis['paid_apps']['avg_rating']
    ]
    colors = ['#2ecc71' if price_analysis['target']['is_free'] else '#95a5f0',
              '#e74c3c' if not price_analysis['target']['is_free'] else '#95a5f0']

    axes[0].bar(categories, ratings, color=colors, edgecolor='black', linewidth=2)
    axes[0].set_ylabel('Average Rating', fontweight='bold')
    axes[0].set_title('Rating Comparison', fontsize=12, fontweight='bold')
    axes[0].set_ylim(0, 5)
    axes[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(ratings):
        axes[0].text(i, v + 0.1, f'{v:.2f}', ha='center', fontweight='bold')

    # Installs Comparison (Log Scale)
    installs = [
        price_analysis['free_apps']['avg_installs'],
        price_analysis['paid_apps']['avg_installs']
    ]
    axes[1].bar(categories, installs, color=colors, edgecolor='black', linewidth=2)
    axes[1].set_ylabel('Average Installs (Log Scale)', fontweight='bold')
    axes[1].set_title('Installs Comparison', fontsize=12, fontweight='bold')
    axes[1].set_yscale('log')
    axes[1].grid(axis='y', alpha=0.3)

    # Reviews Comparison (Log Scale)
    reviews = [
        price_analysis['free_apps']['avg_reviews'],
        price_analysis['paid_apps']['avg_reviews']
    ]
    axes[2].bar(categories, reviews, color=colors, edgecolor='black', linewidth=2)
    axes[2].set_ylabel('Average Reviews (Log Scale)', fontweight='bold')
    axes[2].set_title('Reviews Comparison', fontsize=12, fontweight='bold')
    axes[2].set_yscale('log')
    axes[2].grid(axis='y', alpha=0.3)

    plt.suptitle('💰 Price Strategy Analysis', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)

    return encoded



def generate_positioning_chart(positioning_data, cat_stats):
    """
    Generate 2D scatter plot showing competitive positioning
    """

    fig, ax = plt.subplots(figsize=(12, 9))

    cat_avg_rating = cat_stats.get('Rating', 3.5)
    cat_avg_installs = cat_stats.get('Installs', 1000000)

    # Quadrant colors
    colors_map = {
        'Leader': '#2ecc71',  # Green
        'Star': '#3498db',  # Blue
        'Potential': '#f39c12',  # Orange
        'Niche': '#e74c3c'  # Red
    }

    # Plot similar apps
    for app in positioning_data['similar_apps']:
        ax.scatter(
            app['rating'],
            np.log10(app['installs']),
            s=300,
            alpha=0.6,
            c=colors_map[app['position']],
            edgecolors='white',
            linewidth=2,
            label=app['position'] if app not in positioning_data['similar_apps'] else ''
        )
        ax.annotate(
            app['name'],
            (app['rating'], np.log10(app['installs'])),
            fontsize=8,
            ha='center',
            va='center'
        )

    # Plot target app (highlighted)
    target = positioning_data['target']
    ax.scatter(
        target['rating'],
        np.log10(target['installs']),
        s=500,
        c='#9b59b6',
        edgecolors='gold',
        linewidth=3,
        marker='*',
        zorder=5,
        label='Your App'
    )
    ax.annotate(
        target['name'],
        (target['rating'], np.log10(target['installs'])),
        fontsize=11,
        fontweight='bold',
        ha='center',
        va='bottom',
        color='#9b59b6'
    )

    # Add quadrant dividing lines
    ax.axvline(cat_avg_rating, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax.axhline(np.log10(cat_avg_installs), color='gray', linestyle='--', alpha=0.5, linewidth=2)

    # Add quadrant labels
    ax.text(4.5, 8.5, 'Leaders\n(Quality & Popularity)', fontsize=11,
            ha='center', alpha=0.5, style='italic', weight='bold')
    ax.text(2.5, 8.5, 'Stars\n(Quality Only)', fontsize=11,
            ha='center', alpha=0.5, style='italic', weight='bold')
    ax.text(4.5, 2.5, 'Potential\n(Popular Only)', fontsize=11,
            ha='center', alpha=0.5, style='italic', weight='bold')
    ax.text(2.5, 2.5, 'Niche', fontsize=11,
            ha='center', alpha=0.5, style='italic', weight='bold')

    ax.set_xlabel('Rating (Quality)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Installs (Log Scale - Popularity)', fontsize=13, fontweight='bold')
    ax.set_title('📊 Competitive Positioning Matrix', fontsize=15, fontweight='bold', pad=20)

    ax.set_xlim(1, 5)
    ax.set_ylim(1, 10)
    ax.grid(True, alpha=0.2)

    # Create legend
    legend_elements = [
        mpatches.Patch(color='#2ecc71', label='Leaders'),
        mpatches.Patch(color='#3498db', label='Stars'),
        mpatches.Patch(color='#f39c12', label='Potential'),
        mpatches.Patch(color='#e74c3c', label='Niche'),
        mpatches.Patch(color='#9b59b6', label='Your App')
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=11)

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)

    return encoded


def analyze_content_rating_impact(target, apps_df, category):
    """
    Analyze how content rating affects app success metrics
    Content ratings: Everyone, Teen, Mature 17+, Unrated
    """
    category_apps = apps_df[apps_df['Category'] == category].copy()

    # Get unique content ratings
    content_ratings = category_apps['Content Rating'].unique()

    content_analysis = {
        'target_rating': str(target.get('Content Rating', 'N/A')),
        'ratings': {}
    }

    for rating in content_ratings:
        if pd.notna(rating) and rating != 'Unrated':
            rating_apps = category_apps[category_apps['Content Rating'] == rating]
            if len(rating_apps) > 0:
                content_analysis['ratings'][str(rating)] = {
                    'count': len(rating_apps),
                    'percentage': (len(rating_apps) / len(category_apps)) * 100,
                    'avg_app_rating': float(rating_apps['Rating'].mean()),
                    'avg_installs': float(rating_apps['Installs'].mean()),
                    'avg_reviews': float(rating_apps['Reviews'].mean())
                }

    # Determine recommendation
    target_rating_data = content_analysis['ratings'].get(content_analysis['target_rating'])
    if target_rating_data and len(content_analysis['ratings']) > 0:
        # Find best performing content rating
        best_rating = max(content_analysis['ratings'].items(),
                          key=lambda x: x['avg_installs'])

        if best_rating == content_analysis['target_rating']:
            recommendation = f"Your content rating ({content_analysis['target_rating']}) is optimal for this category."
        else:
            recommendation = f"Consider '{best_rating}' content rating for broader reach."
    else:
        recommendation = "No data available for your content rating"

    content_analysis['recommendation'] = recommendation

    return content_analysis


def get_overall_sentiment_and_top_reviews(reviews_df, app_name, top_n=5):
    """
    Determine overall sentiment and get top reviews from dominant category
    Args:
        reviews_df: DataFrame with review data
        app_name: Name of the app
        top_n: Number of reviews to return (default 5)
    Returns:
        Dictionary with overall sentiment and top reviews
    """
    # Filter reviews for this app
    app_reviews = reviews_df[reviews_df['App'] == app_name].copy()

    if len(app_reviews) == 0:
        return {
            'overall_sentiment': 'Unknown',
            'sentiment_counts': {'Positive': 0, 'Neutral': 0, 'Negative': 0},
            'dominant_percentage': 0,
            'top_reviews': []
        }

    # Count sentiments
    sentiment_counts = app_reviews['Sentiment'].value_counts().to_dict()

    # Fill in missing sentiments with 0
    for sentiment in ['Positive', 'Neutral', 'Negative']:
        if sentiment not in sentiment_counts:
            sentiment_counts[sentiment] = 0

    # Determine overall/dominant sentiment
    if sentiment_counts['Positive'] > sentiment_counts['Neutral'] and sentiment_counts['Positive'] > sentiment_counts[
        'Negative']:
        overall_sentiment = 'Positive'
    elif sentiment_counts['Negative'] > sentiment_counts['Positive'] and sentiment_counts['Negative'] > \
            sentiment_counts['Neutral']:
        overall_sentiment = 'Negative'
    elif sentiment_counts['Neutral'] > sentiment_counts['Positive'] and sentiment_counts['Neutral'] > sentiment_counts[
        'Negative']:
        overall_sentiment = 'Neutral'
    else:
        # Tie-breaker: default to Positive if equal
        overall_sentiment = 'Positive'

    # Calculate percentage
    total_reviews = sum(sentiment_counts.values())
    dominant_percentage = round((sentiment_counts[overall_sentiment] / total_reviews) * 100,
                                1) if total_reviews > 0 else 0

    # Get reviews from dominant sentiment category
    dominant_reviews = app_reviews[app_reviews['Sentiment'] == overall_sentiment].copy()

    # Drop NaN values and convert to numeric
    dominant_reviews = dominant_reviews.dropna(
        subset=['Sentiment_Polarity', 'Sentiment_Subjectivity', 'Translated_Review'])
    dominant_reviews['Sentiment_Polarity'] = pd.to_numeric(dominant_reviews['Sentiment_Polarity'], errors='coerce')
    dominant_reviews['Sentiment_Subjectivity'] = pd.to_numeric(dominant_reviews['Sentiment_Subjectivity'],
                                                               errors='coerce')
    dominant_reviews = dominant_reviews.dropna(subset=['Sentiment_Polarity', 'Sentiment_Subjectivity'])

    # Sort by polarity (descending for Positive/Neutral, ascending for Negative to show "most negative")
    if overall_sentiment == 'Negative':
        # For negative reviews, lowest polarity = most negative
        top_reviews = dominant_reviews.nsmallest(top_n, 'Sentiment_Polarity')
    else:
        # For positive/neutral, highest polarity = most positive
        top_reviews = dominant_reviews.nlargest(top_n, 'Sentiment_Polarity')

    # Format results
    review_list = []
    for idx, row in top_reviews.iterrows():
        review_list.append({
            'review': str(row['Translated_Review'])[:250],  # First 250 chars
            'polarity': round(float(row['Sentiment_Polarity']), 3),
            'subjectivity': round(float(row['Sentiment_Subjectivity']), 3),
            'sentiment': str(row['Sentiment'])
        })

    return {
        'overall_sentiment': overall_sentiment,
        'sentiment_counts': sentiment_counts,
        'dominant_percentage': dominant_percentage,
        'top_reviews': review_list
    }


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

        # Calculate Success Score
        success_score = caluculate_success_scores(target, cat_stats, apps_df)
        success_insights = get_success_insights(success_score)

        # Competitive Positioning Analysis
        positioning_data = analyze_competitive_positioning(target, top_similar, cat_stats)
        positioning_chart = generate_positioning_chart(positioning_data, cat_stats)

        # Price Strategy Analysis
        price_strategy = analyze_price_strategy(target, apps_df, row['Category'])
        price_chart = generate_price_comparison_chart(price_strategy)

        # Content Rating Analysis
        content_analysis = analyze_content_rating_impact(target, apps_df, row['Category'])

        # Get overall sentiment and top reviews
        # Load reviews data
        overall_sentiment_data = get_overall_sentiment_and_top_reviews(reviews_df, row['App'], top_n=5)

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
            "chart": chart_base64,
            "success_score": success_score,
            "success_insights": success_insights,
            "positioning_data": positioning_data,
            "positioning_chart": positioning_chart,
            "price_strategy": price_strategy,
            "price_chart": price_chart,
            "content_analysis": content_analysis,
            "overall_sentiment_data": overall_sentiment_data
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

