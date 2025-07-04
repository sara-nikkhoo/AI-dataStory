import streamlit as st
import pandas as pd
import altair as alt
from openai import OpenAI
import openai
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
openai.api_key = os.getenv("openai_api_key")

# Load dataset
df = pd.read_csv("Superstore.csv", encoding="ISO-8859-1")


##kpi
# Filter Technology category
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Year'] = df['Order Date'].dt.year
tech_df = df[df['Category'] == "Technology"]

# Extract last two years in dataset
years = sorted(tech_df['Year'].unique())
this_year = years[-1]
prev_year = years[-2]

# Group total sales and profit by year
yearly_summary = tech_df.groupby('Year')[['Sales', 'Profit']].sum().reset_index()

# Get sales and profit values
sales_now = yearly_summary[yearly_summary['Year'] == this_year]['Sales'].values[0]
sales_prev = yearly_summary[yearly_summary['Year'] == prev_year]['Sales'].values[0]
profit_now = yearly_summary[yearly_summary['Year'] == this_year]['Profit'].values[0]
profit_prev = yearly_summary[yearly_summary['Year'] == prev_year]['Profit'].values[0]

# Compute total sales/profit overall (not just this year)
total_sales = tech_df['Sales'].sum()
total_profit = tech_df['Profit'].sum()
profit_margin = (total_profit / total_sales) * 100

# Compute deltas
delta_sales = ((sales_now - sales_prev) / sales_prev) * 100
delta_profit = ((profit_now - profit_prev) / profit_prev) * 100

# Find top region by total profit
best_region = tech_df.groupby('Region')['Profit'].sum().idxmax()

# Streamlit app
st.markdown("## Key Performance Indicators (Technology Category)")

# First row with dynamic deltas
row1_col1, row1_col2 = st.columns(2)
row1_col1.metric("Total Sales", f"${total_sales:,.0f}", f"{delta_sales:+.2f}%")
row1_col2.metric("Total Profit", f"${total_profit:,.0f}", f"{delta_profit:+.2f}%")

row2_col1, row2_col2 = st.columns(2)
row2_col1.metric("Profit Margin", f"{profit_margin:.2f}%")
row2_col2.metric("Top Region", best_region)


#---------------------------------------------------------------------------------------
# Preprocess data

sales_summary = df.groupby(['Year', 'Category'])['Sales'].sum().reset_index()

# Create chart
chart = alt.Chart(sales_summary).mark_line(point=True).encode(
    x='Year:O',
    y='Sales:Q',
    color=alt.Color(
        'Category:N',
        scale=alt.Scale(
            domain=['Furniture', 'Office Supplies', 'Technology'],
            range=['#1b9e77', '#d95f02', '#666666']
        )
    )
).properties(
    title='Sales Trend by Product Category'
)

# Display chart
st.title("E-commerce Sales Dashboard")
st.altair_chart(chart, use_container_width=True)

#----------------------------------------------------------------------------------------------------
#### percentage increase

# Extract first and last year in the dataset
first_year = df['Year'].min()
last_year = df['Year'].max()

# Initialize list to store growth values
growth_data = []

# Loop through each category
for cat in df['Category'].unique():
    mask_cat = df['Category'] == cat
    sales_first = df[(df['Year'] == first_year) & mask_cat]['Sales'].sum()
    sales_last = df[(df['Year'] == last_year) & mask_cat]['Sales'].sum()
    growth_percent = ((sales_last - sales_first) / sales_first) * 100
    growth_data.append({
        'Category': cat,
        'Growth (%)': growth_percent
    })

# Create DataFrame
df_growth = pd.DataFrame(growth_data)

# Prepare for plotting
df_plot = pd.DataFrame({
    'Year': [first_year, last_year] * len(df_growth),
    'Category': df_growth['Category'].tolist() * 2,
    'Growth (%)': [0 if i % 2 == 0 else val for i, val in enumerate(df_growth['Growth (%)'].repeat(2))]
})

# Altair chart
chart_growth = alt.Chart(df_plot).mark_line(point=True).encode(
    x=alt.X('Year:O', title='', axis=alt.Axis(labelAngle=0)),
    y=alt.Y('Growth (%):Q', title='Growth from First to Last Year'),
    color='Category:N'
).properties(
    width=500,
    height=350,
    title=f'Yes, the Office Supplies segment appears to be a strong candidate for investment'
)

# Display chart
#st.subheader(" % Growth in Sales by Category")
#st.altair_chart(chart_growth)


# adding an annotation

# Assume 'df_growth' has 'Category' and 'Growth (%)'
df_ann = pd.DataFrame({
    'Text': df_growth.apply(lambda row: f"{row['Category']}: {row['Growth (%)']:.2f}%", axis=1),
    'Y': df_growth['Growth (%)'],
    'X': [last_year] * len(df_growth),  # or the last year in your data
    'Category': df_growth['Category']
})

 
pi = alt.Chart(df_ann).mark_text(
    dx=5, #A
    align='left', #B
    baseline='middle',
    fontSize=18 #C
 ).encode(
    text='Text:N',
    y='Y:Q',
    x='X:O',
    color=alt.Color('Category:N', 
                    scale=alt.Scale(domain=['Office Supplies', 'Furniture', 'Technology'], 
                    range=['grey','lightgrey' ,'orange']),
                    legend=None)
 )


##----adding annotation gp


# Create annotation DataFrame
df_text = pd.DataFrame([{'text': 'vvvv'}])


text = alt.Chart(df_text).mark_text(
            lineBreak='\n',
            align='left',
            fontSize=20,
            y=100,
            color='orange'
        ).encode(
            text='text:N'
        )

total = (chart_growth + pi | text).configure_view(strokeWidth=0).configure_title(
 fontSize=20,
 offset=25
 )
st.altair_chart(total)




#### Generate insight with GPT
client = OpenAI()
if st.button("Show Insight"):
    with st.sidebar:
        st.subheader("GPT Insight")
        prompt = f"""Here is a dataset summarizing yearly sales by product category from a retail store:
    
        {df_growth.round(2).to_string(index=False)}

    Please provide a short analysis as an  Invitor to Actionin **3-4 lines maximum**. Highlight only the most important trend or insight."""

        response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

        st.info(response.choices[0].message.content)



##### --------------------------------------------------------------------------------------chart by reagion an Office Supplies
off_data = df[df['Category']=='Office Supplies']
rigion_summary = off_data.groupby(['Year', 'Region'])['Sales'].sum().reset_index()
region_chart = alt.Chart(rigion_summary).mark_line(point=True).encode(
    x='Year:O',
    y='Sales:Q',
    color=alt.Color(
        'Region:N',
        scale=alt.Scale(
            domain=['East', 'West', 'Central', 'South'],
            range=['#1b9e77', '#d95f02', '#7570b3', '#666666']
        )
    )
).properties(
    title='Office Supplies Sales Trend by Region'
)

# Display in Streamlit
st.subheader("Office Supplies Sales Trend Across Regions")
st.altair_chart(region_chart, use_container_width=True)


#chart for profit
# Filter Office Supplies category

# Group by Region and calculate total Profit
region_profit = off_data.groupby('Region')['Profit'].sum().reset_index()

# Sort for better visualization
region_profit = region_profit.sort_values(by='Profit', ascending=False)

# Create a bar chart
profit_chart = alt.Chart(region_profit).mark_bar().encode(
    x=alt.X('Profit:Q'),
    y=alt.Y('Region:N', sort='-x'),
    color=alt.Color('Profit:Q', scale=alt.Scale(scheme='greens'))
).properties(
    title='Profit by Region - Office Supplies Category'
)

# Display in Streamlit
st.subheader(" Office Supplies Profit by Region")
#st.altair_chart(profit_chart, use_container_width=True)




##---------------------------------------------------------------------- growth rate in profit

# Calculate profit by region
region_profit = off_data.groupby(['Year', 'Region'])['Profit'].sum().reset_index()

# Calculate mean profit for 'Others' (excluding 'East')
others_profit = region_profit[region_profit['Region'] != 'East'].groupby('Year')['Profit'].mean().reset_index()
others_profit.rename(columns={'Profit': 'Others'}, inplace=True)

# Calculate profit for 'East'
east_profit = region_profit[region_profit['Region'] == 'East'][['Year', 'Profit']]
east_profit.rename(columns={'Profit': 'East'}, inplace=True)

# Merge the two DataFrames
profit_comparison = pd.merge(east_profit, others_profit, on='Year')
#\\\\
                                          
#st.write(profit_comparison) 

# Melt the DataFrame for easier plotting
profit_comparison_melted = profit_comparison.melt(id_vars='Year', var_name='Region', value_name='Profit')



## Create a line chart for profit comparison_melted
profit_chart = alt.Chart(profit_comparison_melted).mark_line(point=True).encode(
    x='Year:O',
    y='Profit:Q',
    color='Region:N'
).properties(
    title='Profit Comparison: East vs Others'
)


### Calculate growth rate
# Calculate baseline value for 2014
baseline = profit_comparison_melted[profit_comparison_melted['Year'] == 2014].set_index('Region')['Profit']

# Calculate difference from baseline for each year
profit_comparison_melted['Diff'] = profit_comparison_melted.apply(
    lambda row: row['Profit'] - baseline[row['Region']] if row['Region'] in baseline else None,
    axis=1
)

#st.write(profit_comparison_melted)
## chart for growth rate


colors = ['#80C11E', 'grey']

x_domain = [profit_comparison_melted['Year'].min(), last_year + 1]

chart_rigion = alt.Chart(profit_comparison_melted).mark_line().encode(
    x = alt.X('Year:Q',
              title=None, 
              scale=alt.Scale(domain=x_domain),
              axis=alt.Axis(format='d',tickMinStep=10)), #A
    y = alt.Y('Diff:Q', 
              title='Difference of profit from 2014',
              axis=alt.Axis(format='.2s')), #B
    color = alt.Color('Region:N', 
                      scale=alt.Scale(range=colors), #C
                      legend=None), #D
    opacity = alt.condition(alt.datum['Rigon'] == 'East', alt.value(1), alt.value(0.5) )#E
 ).properties(
    title={
        "text": "Profit in the East over the last 3 years",
        "anchor": "start"  
    },
    width=600,
    height=250
 )


mask = profit_comparison_melted['Year'] == 2017
last_year = profit_comparison_melted['Year'].max()
na = profit_comparison_melted[mask]['Diff'].values[0] #A 
oth = profit_comparison_melted[mask]['Diff'].values[1]
df_text = pd.DataFrame({'text' : ['Rest of Riogns(mean)','East'],
 'x' : [last_year , last_year ], #B
 'y' : [oth,na]}) #C
text_re = alt.Chart(df_text).mark_text(fontSize=15, align='left', dx=5).encode(
 x = 'x',
 y = 'y',
 text = 'text',
 color = alt.condition(alt.datum.text == 'East', alt.value('#80C11E'), alt.value('grey') #D
 ))

## adding annotation gp
# Create annotation DataFrame

offset = 60 #A

 
df_vline = pd.DataFrame({'y' : [oth + offset, na - offset], 
                         'x' : [last_year,last_year]})
 
line = alt.Chart(df_vline).mark_line(color='black').encode(
    y = 'y',
    x = 'x'
 ) #D
diff = na - oth #B
df_ann = pd.DataFrame({'text' : [f'{diff:.0f}'],
       'x' : [last_year - 0.3], #C
       'y' : [na + (oth-na)/2]}) #E
 
ann = alt.Chart(df_ann).mark_text(fontSize=20, align='left').encode(
    x = 'x',
    y = 'y',
    text = 'text'
 )


## Create context DataFrame

df_context = pd.DataFrame({'text' : [' why this gap?',
                            '1. Is it balanced across sub-categories?', 
                            '2. more discounts', 
                            '3. segment of the market'],
                           'y': [0,1,2,3]})
 
context = alt.Chart(df_context).mark_text(fontSize=13, align='left', dy=10, dx=-0).encode(
    x = alt.value(-30),
    y = alt.Y('y:O', axis=alt.Axis(title=None, ticks=False, domain=False, labels=False)),
    text = 'text',
    stroke = alt.condition(alt.datum.y == 0, alt.value('#80C11E'), alt.value('grey')),
    strokeWidth = alt.condition(alt.datum.y == 0, alt.value(1), alt.value(0)
 )
)


total_re = (context | (chart_rigion + text_re + line + ann))

total_re = total_re.configure_axis(
    grid=False,  # Ensure grid lines are visible
    domain=True,  # Ensure axis lines are visible
    domainColor='black',  # Set axis line color to black
    domainWidth=1,  # Set axis line width
    tickColor='black',  # Set tick color to black
    tickWidth=1,  # Set tick width
    labelFontSize=12,  # Adjust label font size for better visibility
    titleFontSize=14 
).configure_title(
    fontSize=16,
    color='#80C11E',
    offset=10, 
    anchor='start'

).configure_view(
    strokeWidth=1
)

st.altair_chart(total_re)
#----------------------------------------------------------------------------------------


# Display unique sub-categories in Office Supplies data
#st.subheader("Sub-categories in Office Supplies")
#st.write(off_data['Sub-Category'].unique())
# Display unique sub-categories in Technology data
#st.subheader("Sub-categories in Technology")
#tech_subcategories = tech_df['Sub-Category'].unique()
#st.write(tech_subcategories)

#--------------------------------------------------------------------------------------- gpt insight
# Generate insight with GPT
if st.button("Show Insight "):
    with st.sidebar:
        st.subheader("GPT Insight on Profit Comparison")
        prompt = f"""Here is a dataset comparing profit in the East region with the average profit of other regions over the years:
        

        {profit_comparison_melted.round(2).to_string(index=False)}

    Please provide a short analysis as an  Invitor to Actionin **3-4 lines maximum**. highlight only the most important trend or insight. mention  other analysis that could be done after. provide a clear and concise insight for
 Target Audience of businessmen  ."""

        response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

        st.info(response.choices[0].message.content)

#--------------------------------------------------------------------------------------- sub-category analysis
sub_cat_data = off_data.groupby(['Year', 'Sub-Category' ])['Profit'].sum().reset_index()
# Filter data for first and last year

# Select first and last year for sub-category analysis
first_year_sub = sub_cat_data['Year'].min()
last_year_sub = sub_cat_data['Year'].max()
# Calculate the percentage of growth in each sub-category between the first and last year
first_year_data = sub_cat_data[sub_cat_data['Year'] == first_year_sub]
last_year_data = sub_cat_data[sub_cat_data['Year'] == last_year_sub]

# Merge first and last year data on Sub-Category
merged_data = pd.merge(first_year_data, last_year_data, on='Sub-Category', suffixes=('_first', '_last'))
# Calculate percentage growth
merged_data['Growth (%)'] = ((merged_data['Profit_last'] - merged_data['Profit_first']) / merged_data['Profit_first']) * 100
merged_data['diff'] = (merged_data['Profit_last'] - merged_data['Profit_first'])
# Display the result



sub_chart = alt.Chart(merged_data).mark_bar().encode(
    x=alt.X('Sub-Category:N', title='Sub-Category',sort='-y'),
    y=alt.Y('Growth (%):Q', title='Growth (%)'),
    color=alt.Color('Growth (%):Q', scale=alt.Scale(scheme='blues')),
    tooltip=['Sub-Category:N', 'Growth (%):Q']
).properties(
    title='Percentage Growth in Profit by Sub-Category'
) 

line = alt.Chart(pd.DataFrame({'y': [75]})).mark_rule(color='red').encode(y='y:Q')
sub_chart = sub_chart + line
sub_chart.configure_axis(
    labelAngle=0  # Ensure labels are horizontal
).configure_title(
    fontSize=16,
    color='#80C11E',
    offset=10, 
    anchor='start'
) 
st.altair_chart(sub_chart, use_container_width=True)
sub_chart_last = alt.Chart(merged_data).mark_bar().encode(
    x=alt.X('Sub-Category:N', title='Sub-Category',sort='-y'),
    y=alt.Y('Profit_last:Q', title='Last Profit'),
    #color=alt.Color('Growth (%):Q', scale=alt.Scale(scheme='blues')),
    tooltip=['Sub-Category:N', 'dlastProfit:Q']
).properties(
    title='Last Year Profit by Sub-Category'
).configure_axis(
    labelAngle=0
)   
st.altair_chart(sub_chart_last, use_container_width=True)