import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# Part 1: Data Loading and Basic Python Operations

df = pd.read_csv('data/Churn_Modelling.csv')

# View basic data information

Shows column data types and non-null counts
print(df.info())   
Gives summary statistics for numerical columns       
print(df.describe())     
print(df.head()) 


# Create a function to calculate the average balance for different age groups.

def calculate_age_group_balance(df):
    df['age_group'] = None
    for idx in df.index:
        age = df.loc[idx, 'Age']
        if age < 30:
            df.loc[idx, 'age_group'] = 'Young'
        elif age < 50:
            df.loc[idx, 'age_group'] = 'Middle-aged'
        else:
            df.loc[idx, 'age_group'] = 'Senior'
    return df.groupby('age_group')['Balance'].mean()
avg_balances = calculate_age_group_balance(df)
print(avg_balances)

# Use a dictionary to count customers by country.

country_counts = df['Geography'].value_counts().to_dict()
print(country_counts)

# Part 2: Data Structure Manipulation

#1. Create lists of churned and retained customers
churned_customers = df[df['Exited'] == 1]['CustomerId'].tolist()
retained_customers = df[df['Exited'] == 0]['CustomerId'].tolist()

#2. Use list comprehension to filter high-value customers (balance > 100,000)
high_value_customers = [
    customer_id for customer_id, balance in zip(df['CustomerId'], df['Balance']) if balance > 100000
]
# 3. Create a dictionary with customer statistics by country
country_stats = {
    country: {
        'avg_balance': df[df['Geography'] == country]['Balance'].mean(),
        'churn_rate': df[df['Geography'] == country]['Exited'].mean() * 100
    }
    for country in df['Geography'].unique()
}

# Part 3: Data Cleaning and Preparation

def prepare_data(df):

    # Handle missing values
    df['Balance'].fillna(df['Balance'].mean(), inplace=True)
    
    # Create new features

    df['balance_per_product'] = df['Balance'] / df['NumOfProducts']
    df['is_high_value'] = df['Balance'] > df['Balance'].mean()

    # Convert categorical variables
    df = pd.get_dummies(df, columns=['Gender', 'Geography'], drop_first=True)
    return df

# Part 4: Exploratory Data Analysis and Visualization

def create_visualizations(df):
    # Set up the matplotlib figure with a 3x2 grid
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    
    # Plot Age Distribution by Churn Status
    sns.histplot(data=df, x='Age', hue='Exited', ax=axes[0, 0])
    axes[0, 0].set_title('Age Distribution by Churn Status')
    
    # Plot Churn Rate by Country
    sns.barplot(data=df, x='Geography', y='Exited', ax=axes[0, 1])
    axes[0, 1].set_title('Churn Rate by Country')
    
    # Plot Balance Distribution by Product Number
    sns.boxplot(data=df, x='NumOfProducts', y='Balance', ax=axes[1, 0])
    axes[1, 0].set_title('Balance Distribution by Product Number')
    
    # Duplicate Plot (Consider changing this to avoid repetition)
    sns.boxplot(data=df, x='NumOfProducts', y='Balance', ax=axes[1, 1])
    axes[1, 1].set_title('Balance Distribution by Product Number')
    
    # Plot Correlation Heatmap
    numeric_cols = ['Age', 'Balance', 'CreditScore', 'Tenure']
    sns.heatmap(df[numeric_cols].corr(), annot=True, ax=axes[2, 0])
    axes[2, 0].set_title('Correlation Heatmap')
    
    # Plot Churn Rate by Credit Score Range
    df['CreditScoreRange'] = pd.cut(df['CreditScore'], bins=4)
    sns.barplot(data=df, x='CreditScoreRange', y='Exited', ax=axes[2, 1])
    axes[2, 1].set_title('Churn Rate by Credit Score Range')
    
    # Adjust layout to prevent overlapping
    plt.tight_layout()
    
    # Return the figure object
    return fig




# Create and display visualizations
fig = create_visualizations(df)
plt.show()


# Part 5: Basic Predictive Analysis
# 1. Prepare features for modeling
features = df[['CreditScore', 'Age', 'Balance', 'NumOfProducts', 'IsActiveMember']]
target = df['Exited']
# 2. Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)



# 3. Create a Simple Prediction Model:
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)



print('Accuracy:', accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

