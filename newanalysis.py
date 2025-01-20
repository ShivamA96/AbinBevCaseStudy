import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os
from datetime import datetime


def validateData(df):
    required_columns = {
        'Country', 'Zone', 'Product_ID', 'Product_Category', 'Units_Sold',
        'Revenue', 'Cost_of_Goods_Sold', 'Profit', 'Profit_Margin',
        'Marketing_Spend', 'Brand_Awareness_Score', 'Customer_Rating',
        'Volume_of_Feedback', 'Competitor_Price', 'Competitor_Market_Share', 'Date'
    }

    missing_cols = required_columns - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    numeric_cols = ['Units_Sold', 'Revenue', 'Cost_of_Goods_Sold', 'Profit',
                    'Profit_Margin', 'Marketing_Spend', 'Brand_Awareness_Score',
                    'Customer_Rating', 'Volume_of_Feedback', 'Competitor_Price',
                    'Competitor_Market_Share']

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['Date'] = pd.to_datetime(df['Date'])

    return df


def calculateProfitibalityMetrics(df):

    metrics['total_revenue'] = df['Revenue'].sum()
    print(metrics['total_revenue'])
    metrics['total_cost'] = df['Cost_of_Goods_Sold'].sum()
    metrics['total_profit'] = df['Profit'].sum()
    metrics['overall_margin'] = (
        metrics['total_profit'] / metrics['total_revenue']) * 100

    category_metrics = df.groupby('Product_Category').agg({
        'Revenue': 'sum',
        'Profit': 'sum',
        'Profit_Margin': 'mean',
        'Units_Sold': 'sum'
    }).round(4)

    zone_metrics = df.groupby('Zone').agg({
        'Revenue': 'sum',
        'Profit': 'sum',
        'Profit_Margin': 'mean'
    }).round(4)

    return metrics, category_metrics, zone_metrics


def brandMetricsAnalyze(df):
    brand_metrics = {}

    brand_metrics['avg_rating'] = df['Customer_Rating'].mean()
    brand_metrics['total_feedback'] = df['Volume_of_Feedback'].sum()
    brand_metrics['avg_awareness'] = df['Brand_Awareness_Score'].mean()

    # Brand metrics by category
    category_brand = df.groupby('Product_Category').agg({
        'Customer_Rating': 'mean',
        'Brand_Awareness_Score': 'mean',
        'Volume_of_Feedback': 'sum',
        'Marketing_Spend': 'sum'
    }).round(4)

    category_brand['marketing_efficiency'] = (
        category_brand['Brand_Awareness_Score'] /
        category_brand['Marketing_Spend']
    ).round(3)

    return brand_metrics, category_brand


def analyzeCompetitivePosition(df):
    comp_metrics = {}

    comp_metrics['avg_market_share'] = 100 - \
        df['Competitor_Market_Share'].mean()

    price_position = df.groupby('Product_Category').agg({
        'Revenue': lambda x: x.mean(),
        'Units_Sold': lambda x: x.mean(),  # Our price
        'Competitor_Price': 'mean',
        'Competitor_Market_Share': 'mean'
    })

    price_position['Product_Price'] = price_position['Revenue'] / \
        price_position['Units_Sold']

    price_position['price_difference'] = (
        (price_position['Product_Price'] - price_position['Competitor_Price']) /
        price_position['Competitor_Price'] * 100
    ).round(4)

    return comp_metrics, price_position


def createVisualizations(df, metrics,  save_path='outputs/'):
    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(20, 15))
    gs = GridSpec(2, 2)

    ax1 = plt.subplot(gs[0, :])
    prof_by_cat = df.pivot_table(
        values=['Profit', 'Revenue'],
        index='Product_Category',
        aggfunc='sum'
    ).sort_values('Profit', ascending=True)
    prof_by_cat.plot(kind='barh', ax=ax1)
    ax1.set_title('Profitability by Category')
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(gs[1, 0])
    monthly_margins = df.groupby([
        pd.Grouper(key='Date', freq='M'),
        'Product_Category'
    ])['Profit_Margin'].mean().unstack()
    monthly_margins.plot(ax=ax2)
    ax2.set_title('Profit Margin Trends')
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(gs[1, 1])
    zone_perf = df.pivot_table(
        values='Profit_Margin',
        index='Zone',
        columns='Product_Category',
        aggfunc='mean'
    )
    sns.heatmap(zone_perf, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax3)
    ax3.set_title('Zone Performance Heat Map')

    plt.tight_layout()
    plt.savefig(f'{save_path}profitability_analysis.png')
    plt.close()

    plt.figure(figsize=(20, 15))
    gs = GridSpec(2, 2)

    ax1 = plt.subplot(gs[0, 0])
    sns.barplot(
        data=df,
        x='Brand_Awareness_Score',
        y='Customer_Rating',
        hue='Product_Category',
        alpha=0.6,
        ax=ax1
    )
    ax1.set_title('Brand Awareness vs Customer Rating')

    correlation_data = df[['Brand_Awareness_Score',
                           'Customer_Rating', 'Volume_of_Feedback']].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_data, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(f'{save_path}correlation_matrix.png')
    plt.close()

    popularity_metrics = df.groupby('Product_Category').agg({
        'Revenue': 'sum',
        'Units_Sold': 'sum',
        'Competitor_Market_Share': 'mean'
    }).sort_values(by='Revenue', ascending=False)

    popularity_metrics.plot(kind='bar', figsize=(10, 7))
    plt.title('Brand Popularity by Revenue, Units Sold, and Competitor Market Share')
    plt.ylabel('Values')
    plt.tight_layout()
    plt.savefig(f'{save_path}brand_popularity.png')
    plt.close()

    ax2 = plt.subplot(gs[0, 1])
    marketing_efficiency = df.groupby('Product_Category').agg({
        'Marketing_Spend': 'sum',
        'Brand_Awareness_Score': 'mean'
    })
    marketing_efficiency['Efficiency'] = (
        marketing_efficiency['Brand_Awareness_Score'] /
        marketing_efficiency['Marketing_Spend']
    )
    marketing_efficiency['Efficiency'].sort_values().plot(
        kind='barh',
        ax=ax2
    )
    ax2.set_title('Marketing Efficiency by Category')

    ax3 = plt.subplot(gs[1, :])
    feedback_trend = df.groupby([
        pd.Grouper(key='Date', freq='M'),
        'Product_Category'
    ])['Volume_of_Feedback'].sum().unstack()
    feedback_trend.plot(ax=ax3)
    ax3.set_title('Feedback Volume Trends')

    plt.tight_layout()
    plt.savefig(f'{save_path}brand_analysis.png')
    plt.close()

    plt.figure(figsize=(20, 15))
    gs = GridSpec(2, 2)

    ax1 = plt.subplot(gs[0, :])
    market_share = df.groupby('Zone')['Competitor_Market_Share'].mean()
    our_share = 100 - market_share
    pd.DataFrame({
        'Our Share': our_share,
        'Competitor Share': market_share
    }).plot(kind='bar', stacked=False, ax=ax1)
    ax2 = plt.subplot(gs[1, 0])

    df['Unit_Price'] = df['Revenue'] / df['Units_Sold']

    price_comp = df.groupby('Product_Category').agg({
        'Unit_Price': 'mean',
        'Competitor_Price': 'mean'
    }).round(4)

    price_comp.plot(kind='bar', ax=ax2)
    ax2.set_title('Average Unit Price Comparison with Competitors')
    ax2.set_ylabel('Price per Unit')
    ax2.tick_params(axis='x', rotation=45)

    for i in ax2.containers:
        ax2.bar_label(i, fmt='%.2f', padding=3)

    ax2.legend(['Our Price', 'Competitor Price'])

    ax2.grid(True, axis='y', linestyle='--', alpha=0.7)

    ax3 = plt.subplot(gs[1, 1])
    sns.barplot(
        data=df,
        x='Revenue',
        y='Competitor_Market_Share',
        hue='Product_Category',
        alpha=0.6,
        ax=ax3
    )
    ax3.set_title('Price vs Market Share')

    plt.tight_layout()
    plt.savefig(f'{save_path}competitive_analysis.png')
    plt.close()

    average_pricing = (
        df.groupby('Product_Category')['Competitor_Price']
        .mean()
        .reset_index()
        .rename(columns={'Product_Category': 'Competitor', 'Competitor_Price': 'Price'})
    )

    market_share = (
        df.groupby('Product_Category')['Competitor_Market_Share']
        .mean()
        .reset_index()
        .rename(columns={'Product_Category': 'Competitor', 'Competitor_Market_Share': 'Market_Share'})
    )

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=average_pricing, x='Competitor',
                     y='Price', palette='viridis')
    plt.title('Average Pricing by Competitor')
    plt.ylabel('Average Price')
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f'),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=8, color='black', xytext=(0, 5),
                    textcoords='offset points')
    plt.ylim(0, average_pricing['Price'].max() * 1.1)
    plt.tight_layout()
    plt.savefig(f'{save_path}average_pricing.png')
    plt.close()
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=market_share, x='Competitor',
                     y='Market_Share', palette='magma')
    plt.title('Market Share by Competitor')
    plt.ylabel('Market Share (%)')
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f'),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=8, color='black', xytext=(0, 5),
                    textcoords='offset points')
    plt.ylim(0, market_share['Market_Share'].max() * 1.1)
    plt.tight_layout()
    plt.savefig(f'{save_path}market_share.png')
    plt.close()

    current_price = df['Competitor_Price'].mean()
    competitor_average = average_pricing['Price'].mean()

    if current_price > competitor_average:
        pricing_strategy = "Consider reducing prices to be more competitive."
    elif current_price < competitor_average:
        pricing_strategy = "Potential to increase prices to improve margins."
    else:
        pricing_strategy = "Maintain current pricing strategy."

    with open(f'{save_path}pricing_strategy.txt', 'w') as f:
        f.write(pricing_strategy)


def generateReport(df, save_path='outputs/'):
    df = validateData(df)

    prof_metrics, category_metrics, zone_metrics = calculateProfitibalityMetrics(
        df)
    brand_metrics, category_brand = brandMetricsAnalyze(df)
    comp_metrics, price_position = analyzeCompetitivePosition(df)

    createVisualizations(df, {
        'profitability': prof_metrics,
        'brand': brand_metrics,
        'competitive': comp_metrics
    }, save_path)

    metrics_path = os.path.join(save_path, 'detailed_metrics')
    os.makedirs(metrics_path, exist_ok=True)

    category_metrics.to_csv(f'{metrics_path}/category_metrics.csv')
    zone_metrics.to_csv(f'{metrics_path}/zone_metrics.csv')
    category_brand.to_csv(f'{metrics_path}/category_brand_metrics.csv')
    price_position.to_csv(f'{metrics_path}/price_position_metrics.csv')

    return {
        'profitability': prof_metrics,
        'brand': brand_metrics,
        'competitive': comp_metrics,
        'category_metrics': category_metrics,
        'zone_metrics': zone_metrics,
        'category_brand': category_brand,
        'price_position': price_position
    }



if __name__ == "__main__":
    df = pd.read_csv('processed_data.csv', parse_dates=['Date'])

    analysis_results = generateReport(df, save_path='outputs/')

    print("Analysis completed. Check the 'outputs' directory for visualizations and detailed metrics.")
