---

NOTE
🟩 Version 1 (simple baseline)

In v1, you did:

- manual feature selection
- picked only the “important” features (e.g., median_income, etc.)
- used standard linear regression from sklearn
- kept it very simple and educational

🟦 Version 2 (full ML pipeline)

In Version 2, you are building a professional-grade ML pipeline, so the strategy changes:

⭐ WE USE ALL FEATURES + ALL ENGINEERED FEATURES

(unless a feature is clearly useless or duplicates another — but we haven’t removed any yet)

When you engineer features correctly and standardize them, linear regression benefits from more features, not fewer.

If a feature is useless, L1/L2 regularization will shrink its weight automatically.

So manual feature selection is no longer needed.

🟦 This is why we needed EDA notebook

- to justify logs (because skew)
- to justify ratios (because scale imbalance)
- to justify polynomial features (because nonlinearity)
- to justify categorical encoding
