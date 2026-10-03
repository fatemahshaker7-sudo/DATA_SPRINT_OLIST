# Olist Sellers Evaluation Analysis​

## Problem Statement

Between 2017 and 2018 in Brazil, Olist faced inconsistencies in sellers performance across sales, revenue, and customer satisfaction, while lacking a clear understanding of what distinguishes successful sellers. This may negatively affect customer experience and business performance, creating a need to better understand its seller base and identify opportunities to increase profitability.​

## Executive Summary

**Process.** We merged Seven Olist datasets (order items, orders, reviews, marketing qualified leads,products,category,  and closed deals) into one seller-level table keyed on `order_id`, `seller_id`, and `mql_id`. Date columns were parsed, columns that were mostly empty or not used in the analysis were dropped (`has_company`, `has_gtin`, `average_stock`, `declared_monthly_revenue`, `declared_product_catalog_size`, `review_comment_title`), and several features were engineered (delivery time, delivery-time buckets, translated review text, and a "good seller" flag). Analysis was exploratory and descriptive: group-by aggregations, correlations, price-gap comparisons, funnel conversion rates, and a word cloud of translated complaints. No predictive models were built.

**Key findings.**
- **Resellers outperform manufacturers.** Resellers account for the most orders and revenue (about 3,922 sales and roughly 598.8K in revenue in the reseller group, with a mean review score of about 4.2), and no manufacturer was found to outperform a reseller on sales, revenue, and satisfaction together.
- **Resellers are more expensive.** Resellers' average and median prices are higher than manufacturers' overall, with the size of the price gap varying considerably by business segment and product category.
- **Delivery time drives ratings.** Average review score falls as delivery time increases, for both business types.
- **The funnel is narrow.** Only about 10% of marketing qualified leads became sellers, and conversion differs by lead origin channel.
- **Sales-rep (SR) effectiveness varies.** Using a simple definition of a "good" seller (more than 10 sales and an average review of at least 3), the share of good sellers differs noticeably across sales reps.

**Conclusions and recommendations (summary).** Olist should prioritise recruiting and supporting resellers, focus acquisition spend on the highest-converting lead channels, monitor sellers who combine low review scores with low sales/revenue as early red flags, and track delivery time as a leading indicator of seller quality. Sales-rep performance in selecting sellers should be reviewed and best practices shared from the top-performing reps.

## File Directory / Table of Contents

```
.
├── README.md
├── Data/
│   ├── Original/
│   │   ├── olist_closed_deals_dataset.csv
│   │   ├── olist_marketing_qualified_leads_dataset.csv
│   │   ├── olist_order_reviews_dataset.csv
│   │   ├── olist_orders_dataset.csv
│   │   ├── olist_order_items_dataset.csv
│   │   ├── olist_products_dataset.csv
│   │   └── product_category_name_translation.csv
│   └── Cleaned/
│       └── merged_seller_orders.csv        # merged + cleaned table 
├── Code/
│    └── Olist_Lab_python_file.py            # load raw CSVs, parse dates, EDA, visualizations and analyse 
├── Presentation/
│   ├── Olist_Seller_Analysis.pdf
│   └── images/                             # charts exported by clean_chart()
```

**Libraries:** `pandas`, `seaborn`, `matplotlib`, `nltk`, `wordcloud`, `translatepy`

```bash
pip install pandas seaborn matplotlib nltk wordcloud translatepy
```

## Data and Data Dictionary

**Source.** The data comes from two public Kaggle datasets published by Olist: the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (orders, items, reviews, products) and the [Marketing Funnel by Olist](https://www.kaggle.com/datasets/olistbr/marketing-funnel-olist) (marketing qualified leads and closed deals). Join logic: order items → orders (`order_id`) → reviews (`order_id`) → seller/lead info via `seller_id`; MQLs → closed deals (`mql_id`).

### Final merged dataset (`df`)

| Feature | Type | Description |
|---|---|---|
| `order_id` | string | Unique order identifier |
| `order_item_id` | int | Sequential item number within an order |
| `product_id` | string | Product identifier |
| `seller_id` | string | Seller identifier |
| `shipping_limit_date` | datetime | Seller's deadline to hand the item to the carrier |
| `price` | float | Item price |
| `freight_value` | float | Freight cost for the item |
| `customer_id` | string | Customer identifier for the order |
| `order_status` | string | Order status (e.g., delivered, shipped, canceled) |
| `order_purchase_timestamp` | datetime | When the order was placed |
| `order_approved_at` | datetime | When payment was approved |
| `order_delivered_carrier_date` | datetime | When the order was handed to the carrier |
| `order_delivered_customer_date` | datetime | When the customer received the order |
| `order_estimated_delivery_date` | datetime | Estimated delivery date shown to the customer |
| `review_id` | string | Review identifier |
| `review_score` | int (1-5) | Customer satisfaction score |
| `review_comment_message` | string | Review text (Portuguese) |
| `review_creation_date` | datetime | When the review survey was sent |
| `review_answer_timestamp` | datetime | When the review was answered |
| `mql_id` | string | Marketing qualified lead identifier |
| `first_contact_date` | datetime | Date of the lead's first contact |
| `landing_page_id` | string | Landing page the lead came from |
| `origin` | string | Marketing channel of the lead (e.g., organic search, paid search, social) |
| `sdr_id` | string | Sales development rep identifier |
| `sr_id` | string | Sales rep identifier |
| `won_date` | datetime | Date the deal was closed |
| `business_segment` | string | Seller's business segment |
| `lead_type` | string | Type of lead |
| `lead_behaviour_profile` | string | Behavioural profile of the lead |
| `business_type` | string | Seller type: manufacturer, reseller, or other |

**Dropped columns:** `has_company`, `has_gtin`, `average_stock`, `declared_monthly_revenue`, `declared_product_catalog_size`, `review_comment_title`.

### Engineered features

| Feature | Description |
|---|---|
| `delivery_days` | Days between `order_purchase_timestamp` and `order_delivered_customer_date` |
| `delivery_bucket` | `delivery_days` binned into 0-7, 8-14, 15-21, 22-30, and 30+ |
| `review_comment_message_english` | Machine-translated (translatepy) English version of the review text, for 1-2 star reviews of the 10 worst sellers |
| `product_category_name_english` | English product category, merged from the products and translation files |
| `converted` | Boolean: whether an MQL has a `won_date` (became a seller) |
| `rate` | Lead conversion rate (%) per `origin` |
| `price_gap` / `gap_price` | % difference between reseller and manufacturer price (mean or median) per segment or category |
| `sales`, `revenue`, `satisfaction` | Per-seller (or per-type) distinct orders, summed price, and mean review score |
| `unique_products` | Number of distinct products per seller (catalog breadth) |
| `good` / `good_pct` / `not_good_pct` | Seller flagged good if more than 10 sales and mean review of at least 3; share per sales rep |

## Key Visualizations

Charts are exported to `Presentation/images/` by `clean_chart()` (transparent PNG, 300 dpi). Suggested highlights:

- `most_successful_seller_based_on_order_counts.png` – order counts by business type
- `top_10_underperforming_sellers.png` and `top_10_successful_sellers.png` – red-flag and green-flag sellers
- `average_rating_by_delivery_time.png` – rating vs. delivery time by business type
- Price gap (reseller vs. manufacturer) by business segment and product category
- Lead conversion rate by origin channel
- Good vs. not-good seller share by sales representative

## Conclusions and Recommendations

**Conclusions**
1. Resellers lead on order volume, revenue, and satisfaction, and no manufacturer outperforms a reseller overall.
2. Resellers charge more than manufacturers, but the gap depends heavily on segment and category.
3. Slower delivery is associated with lower review scores, and complaints from the worst sellers (word cloud of translated 1-2 star reviews) center on delivery and product issues.
4. Roughly 10% of MQLs convert to sellers, and channels differ in conversion rate, seller count, revenue, and average review.
5. The share of good sellers varies by sales representative.

**Recommendations**
- Prioritise reseller recruitment while monitoring manufacturers for niche opportunities.
- Shift marketing effort toward the highest-converting channels and review the lowest-converting ones.
- Build an automated red-flag rule (e.g., satisfaction of 2 or less with low revenue and low sales) and review flagged sellers early.
- Set delivery-time expectations and coach sellers whose orders fall into the slowest buckets.
- Use the sales-rep comparison to share best practices and refine seller selection criteria.

## Areas for Further Research / Study

- Build predictive model to predict seller success from lead attributes before onboarding.
- Build predictive model to predict long term seller performance based on bussines segment, sales, revenue, scores, comments and suggest improvement plan.
- Examine causes of delivery delays (carrier handoff vs. transit) and their relation to seller location.

## Sources

- Olist. *Brazilian E-Commerce Public Dataset by Olist.* Kaggle. https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
- Olist. *Marketing Funnel by Olist.* Kaggle. https://www.kaggle.com/datasets/olistbr/marketing-funnel-olist
- pandas, seaborn, matplotlib, NLTK, WordCloud, and translatepy documentation.
