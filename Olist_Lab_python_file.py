# %% 
# importing libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from wordcloud import WordCloud, STOPWORDS

# %%
#merging data

def read_csv_date(path,date_cols):
    """
        This function loads the file to read it and converts the dates colomns to dates
        """
    return pd.read_csv(path, parse_dates=date_cols, date_format='mixed')
# files dics to read and convert date colomns
file_config = {'deals':('olist_closed_deals_dataset.csv',['won_date']),
               'marketing':('olist_marketing_qualified_leads_dataset.csv',['first_contact_date']),
               'reviews':('olist_order_reviews_dataset.csv',['review_creation_date','review_answer_timestamp']),
               'orders':('olist_orders_dataset.csv',['order_purchase_timestamp','order_approved_at','order_delivered_carrier_date','order_delivered_customer_date','order_estimated_delivery_date']),
               'orderitem':('olist_order_items_dataset.csv',['shipping_limit_date']),}


dfs = {name: read_csv_date(path,date_cols) for name, (path,date_cols) in file_config.items()}
# renaming colomns and merging the files
merged_emp_df = dfs['orderitem'].merge(dfs['orders'], on='order_id', how='left')
merged_emp_df2 = merged_emp_df.merge(dfs['reviews'], on='order_id', how='left')
merged_emp_df3 = dfs['marketing'].merge(dfs['deals'], on='mql_id')
df = merged_emp_df2.merge(merged_emp_df3, how='left', on='seller_id')
df
# %%
df.describe()
# %%
df.info()
# %%
df.duplicated()
# %%
df.shape
# %%
df.isnull().sum()
# %%
df.columns
# %%
df = df.drop(columns= ['has_company','has_gtin','average_stock','declared_monthly_revenue','declared_product_catalog_size','review_comment_title'])
df.columns
# %%
# Visualizations
# %%
def clean_chart(chart,labels=True,fmt='%.2f' , rota =0):
    ''' This function removes the borders, grid, and y-axis of the charts and labels the colomns on the top
     '''
    if labels:
        for container in chart.containers:
            chart.bar_label(container, fmt=fmt, padding = 3, rotation = rota)
    chart.get_yaxis().set_visible(False)
    chart.grid(False)
    chart.spines[['left','right','top']].set_visible(False)
    chart.margins(y=0.12)
    title = chart.get_title().lower().replace(' ', '_')
    chart.figure.savefig(f'{title}.png', transparent=True, dpi=300, bbox_inches='tight')
    return chart
# %%
#Are resellers the most successful sellers?
#score_type= df.groupby('business_type')['review_score'].value_counts()
order_num_type= df.groupby('business_type')['order_id'].count().sort_values(ascending=False)
chart = order_num_type.plot(kind = 'bar', stacked=False, rot=0, width = 0.90, title='Most Successful Seller Based on Order Counts', xlabel = '');
#chart = plt.bar(x=order_num_type.index, height=order_num_type.values);
clean_chart(chart,labels=True,fmt='%.0f' , rota =0);
# %%
#How do sellers compare across business types?
plt.figure(figsize =(17,12))
chart = sns.countplot(data=df, x='business_segment', width = 1, hue='business_type', palette='Set2')
plt.xlabel('')
plt.title('Bussiness Type Count per Segment')
plt.xticks(rotation=90);
clean_chart(chart,labels=True,fmt='%.0f' , rota =0);
# %%
#Are resellers more expensive than manufacturers? yes
price_filter= df.groupby('business_type')['price'].mean()
chart = price_filter.plot(kind = 'bar', stacked=False, rot=0, width = 0.90, title='Most Expensive Bussiness Type', xlabel = '');
clean_chart(chart,labels=True,fmt='%.1f' , rota =0);
# %%
# Who are the red-flag (underperforming) sellers? FS
# avarage score and number of reviews per seller
seller_perform = df.groupby('seller_id')['review_score'].agg(['mean','count'])
# consider sellers with at least 10 review only, take the 10 lowest only 
Red_flags = seller_perform[seller_perform['count']>=10].nsmallest(10,'mean')
# plot the mean only
chart = Red_flags['mean'].plot(kind = 'barh', stacked=False, rot=0, width = 0.90, title='Top 10 underperforming sellers', xlabel = '');

# %%
#Who are the most successful sellers? the opposit of above . FS
# avarage score and number of reviews per seller
seller_perform = df.groupby('seller_id')['review_score'].agg(['mean','count'])
# consider sellers with at least 1000 review , take the 10 lowest only 
green_flags = seller_perform[seller_perform['count']>=1000].nlargest(10,'mean')
# plot the mean only
chart = green_flags['mean'].plot(kind = 'barh', stacked=False, rot=0, width = 0.90, title='Top 10 Successful sellers', xlabel = '');
# %%
#Do manufacturers ship orders faster to carriers than resellers, and how does this affect seller ratings? FS
df['delivery_days'] = (df['order_delivered_customer_date']- df['order_purchase_timestamp']).dt.days
order_delivery= df.groupby('business_type')['delivery_days'].mean()
chart = order_delivery.plot(kind = 'bar', stacked=False, rot=0, width = 0.90, title='Average Shipping Days per Bussines Type', xlabel = '');
clean_chart(chart,labels=True,fmt='%.1f' , rota =0);
# %%
#and how does this affect seller ratings
seller_rating= df.groupby('business_type')['review_score'].mean()
chart = seller_rating.plot(kind = 'bar', stacked=False, rot=0, width = 0.90, title='Average Rating per Bussines Type', xlabel = '');
clean_chart(chart,labels=True,fmt='%.1f' , rota =0);
# %%

nltk.download('stopwords')
from nltk.corpus import stopwords as nltk_stopwords

worst_id = Red_flags.index
# filter to get complaints only assuming they get 1,2 scores only
complaints = df[df['seller_id'].isin(worst_id)&(df['review_score']<=2)]

# word cloud
text= " ".join(complaints['review_comment_message'].dropna().astype(str).str.lower())
stopwords = set(STOPWORDS)
stopwords.update(nltk_stopwords.words('portuguese'))
stopwords.update(['producto','productos','produtos','produto','product', 'nao', 'pra', 'q', 'vc', 'tá', 'ta', 'pq', 'tb', 'tbm','comprei', 'compra', 'comprar', 'recebi', 'recebemos', 'receber',
    'pedido', 'pedi', 'loja', 'dia', 'dias', 'peço', 'favor', 'estou', 'vou',
    'ainda', 'então', 'ja', 'já', 'pois', 'sendo', 'outro', 'outros', 'mesmo' ])

wc = WordCloud(
    width = 1200, height=600, background_color= "white",
    stopwords=stopwords, colormap="viridis").generate(text)

plt.figure(figsize=(12,6))
plt.imshow(wc,interpolation='bilinear')
plt.axis('off')
plt.show()


# %%
# drop any null
d = df.dropna(subset=["business_segment", "business_type", "price"])
d = d[d["business_type"].isin(["reseller", "manufacturer"])]

# mean for eeach type of segments
avg = d.groupby(["business_segment", "business_type"])["price"].mean().reset_index()

# pivot
pivot = avg.pivot(index="business_segment", columns="business_type", values="price").reset_index()

# price gap
pivot["price_gap"] = (pivot["reseller"] - pivot["manufacturer"]) / pivot["manufacturer"] * 100

# bar chart price gap
order = pivot.dropna(subset=["price_gap"]).sort_values("price_gap", ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(data=order, x="business_segment", y="price_gap")
plt.axhline(0, color="red", linestyle="--")
plt.ylabel("Price gap (%)")
plt.xticks(rotation=60, ha="right")
plt.tight_layout()
plt.show()

#Q:How do resellers perform compared to manufacturers in terms of sales, revenue, and customer satisfaction? 
#reseller sell more with revenue of 598779.08 and mean score 4.2 , sales 3922
performance_overall= df.groupby('business_type').agg(
    sales=('order_id','nunique'),
    revenue=('price','sum'),
    satisfaction=('review_score','mean')
)
performance_overall
#does any manufacturer outperform a reseller? no
performance= df.groupby(['business_type','seller_id']).agg(
    sales=('order_id','nunique'),
    revenue=('price','sum'),
    satisfaction=('review_score','mean')
)
#plot1 (Total Sales)
performance_overall['sales'].plot(kind='bar',title='Total Sales');
#plot2(Revenue)
performance_overall['revenue'].plot(kind='bar',title='Total Revenue');
#plot3
performance_overall['satisfaction'].plot(kind='bar',title='Mean Satisfaction Rate');

#Do high-performing sellers specialize in narrow catalogs (niche) or wide catalogs (generalist)??? 
catalog= df.groupby('seller_id').agg(
    unique_products=('product_id','nunique'),
    sales=('order_id','nunique'),
    revenue=('price','sum'),
    satisfaction=('review_score','mean')
)
catalog[['unique_products','sales','revenue','satisfaction']].corr() #can make heatmap
#wider catalog(198-399) products
catalog.sort_values('unique_products',ascending=False).head(10)
#narrow cat
catalog.sort_values('unique_products',ascending=True).head(10)

#redfalg ids
seller_per= df.groupby('seller_id').agg(
    sales=('order_id','nunique'),
    revenue=('price','sum'),
    satisfaction=('review_score','mean')
)
this_filter=(seller_per['satisfaction'] <=2) & (seller_per['revenue']<50) & (seller_per['sales']<15)
redfalg_ids=seller_per[this_filter].index
redfalg_ids


#Lead rate in olist
total_lead=dfs['marketing']['mql_id'].nunique()
leads_won= dfs['deals']['mql_id'].nunique()
rate=(leads_won/total_lead)*100 #only 10% became sellers
print(total_lead,leads_won,rate)

marketingdf= dfs['marketing'].merge(
    dfs['deals'], on='mql_id',how='left')
#which channel leads won more?? 
marketingdf['converted']=marketingdf['won_date'].notna()
marketingdf['converted'].value_counts()
won_by_origin=marketingdf.groupby('origin').agg(
    total_leads=('mql_id','count'),
    won_leads=('converted','sum'),

)
won_by_origin['rate']=(won_by_origin['won_leads']/won_by_origin['total_leads'])*100
won_by_origin=won_by_origin.sort_values(by='rate',ascending=False).round(2)
won_by_origin

#plot: leads origins by rate
won_by_origin['rate'].plot(kind='bar')

plt.xlabel("Channels")
plt.ylabel("Rate")
plt.title("Lead origins by Rate");
