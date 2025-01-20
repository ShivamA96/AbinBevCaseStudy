import pandas as pd
import numpy as np
from scipy import stats
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import logging

ExcelColumns = {
    'Sales_Data': [
        'Country', 'Zone', 'Product_ID', 'Product_Category',
        'Units_Sold', 'Revenue', 'Cost_of_Goods_Sold', 'Date'
    ],
    'Marketing_Data': [
        'Country', 'Zone', 'Marketing_Spend', 'Campaign_Type',
        'Customer_Reach', 'Brand_Awareness_Score', 'Date'
    ],
    'Customer_Feedback_Data': [
        'Country', 'Zone', 'Product_Category', 'Customer_Rating',
        'Feedback_Type', 'Volume_of_Feedback', 'Date'
    ],
    'Competitor_Data': [
        'Country', 'Zone', 'Competitor_Name', 'Competitor_Product_Category',
        'Competitor_Price', 'Competitor_Market_Share'
    ]
}


def load_DFs(excel_path='Command_Centre_Dataset.xlsx'):
    excel_sheets = pd.read_excel(excel_path, sheet_name=None)
    dataframes = {}

    for sheet_name, required_cols in ExcelColumns.items():
        if sheet_name not in excel_sheets:
            raise ValueError(f"Sheet '{sheet_name}' not found in file.")
        df = excel_sheets[sheet_name]

        # Check missing columns
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Missing columns in {sheet_name}: {missing_cols}")

        dataframes[sheet_name] = df

    return (
        dataframes['Sales_Data'],
        dataframes['Marketing_Data'],
        dataframes['Customer_Feedback_Data'],
        dataframes['Competitor_Data']
    )


logging.basicConfig(filename='cleaning.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')


def validateAndClean(df):
    logging.info(f"Starting cleaning for DataFrame with shape {df.shape}")

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                median = df[col].median()
                df[col].fillna(median, inplace=True)
                logging.info(
                    f"Filled NaNs in {col} with median value {median}")
            else:
                mode = df[col].mode()[0]
                df[col].fillna(mode, inplace=True)
                logging.info(f"Filled NaNs in {col} with mode value '{mode}'")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    df = df[(z_scores < 3).all(axis=1)]
    logging.info(f"Removed outliers. New shape: {df.shape}")

    return df


def cleanData(salesDF, marketingDF, customerDF, competitorDF):
    salesDF = validateAndClean(salesDF)
    marketingDF = validateAndClean(marketingDF)
    customerDF = validateAndClean(customerDF)
    competitorDF = validateAndClean(competitorDF)

    for df in [salesDF, marketingDF, customerDF]:
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            logging.info(
                f"Converted 'Date' column to datetime in DataFrame with shape {df.shape}")

    for df in [salesDF, marketingDF, customerDF, competitorDF]:
        if 'Country' in df.columns:
            df['Country'] = df['Country'].astype(str).str.strip().str.title()
            logging.info(
                f"Standardized 'Country' column in DataFrame with shape {df.shape}")
        if 'Zone' in df.columns:
            df['Zone'] = df['Zone'].astype(str).str.strip().str.title()
            logging.info(
                f"Standardized 'Zone' column in DataFrame with shape {df.shape}")

    initial_shape = salesDF.shape
    salesDF.drop_duplicates(
        subset=['Country', 'Zone', 'Date', 'Product_ID'], inplace=True)
    logging.info(
        f"Dropped duplicates in Sales Data from {initial_shape} to {salesDF.shape}")

    salesDF['Profit'] = salesDF['Revenue'] - salesDF['Cost_of_Goods_Sold']
    salesDF['Profit_Margin'] = salesDF['Profit'] / salesDF['Revenue']
    logging.info("Added 'Profit' and 'Profit_Margin' columns to Sales Data")

    if 'Competitor_Product_Category' in competitorDF.columns:
        competitorDF.rename(
            columns={'Competitor_Product_Category': 'Product_Category'}, inplace=True)
        logging.info(
            "Renamed 'Competitor_Product_Category' to 'Product_Category' in Competitor Data")

    merged_df = salesDF.merge(marketingDF, on=['Country', 'Zone', 'Date'], how='left') \
        .merge(customerDF, on=['Country', 'Zone', 'Product_Category', 'Date'], how='left') \
        .merge(competitorDF, on=['Country', 'Zone', 'Product_Category'], how='left')
    logging.info(f"Merged DataFrame shape: {merged_df.shape}")

    imputer = IterativeImputer(max_iter=10, random_state=42)
    cols_to_impute = ['Marketing_Spend',
                      'Customer_Rating', 'Volume_of_Feedback']
    merged_df[cols_to_impute] = imputer.fit_transform(
        merged_df[cols_to_impute])
    logging.info("Imputed missing values using Iterative Imputer")

    merged_df.to_csv('processed_data.csv', index=False)
    logging.info("Saved processed data to 'processed_data.csv'")

    return merged_df


def processAndSaveCSV():
    salesDF, marketingDF, customerDF, competitorDF = load_DFs()

    merged_df = cleanData(salesDF, marketingDF, customerDF, competitorDF)

    merged_df.to_csv('processed_data.csv', index=False)

    print("Data cleaning and merging completed. 'processed_data.csv' is ready for analysis.")


processAndSaveCSV()
