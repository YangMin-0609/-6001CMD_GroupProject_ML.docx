# Processed credit card default dataset

[`credit_default_processed.csv`](credit_default_processed.csv) contains 30,000 records with 23 cleaned predictors, the binary target `DEFAULT_NEXT_MONTH`, and a `SPLIT` column. The latter identifies the 24,000 training and 6,000 test rows and should be excluded from model inputs. Customer IDs are excluded.

The export uses the same column names, fixed education-code recoding (`5` and `6` to `0`), and 80/20 stratified split (`random_state=42`) as [the notebook](../credit_default_full.ipynb). The CSV is saved at the cleaned-input stage. The model pipelines fit RobustScaler and one-hot encoding within each training fold and then refit on the training partition for final testing.

Source and attribution: Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository. [Dataset and license](https://doi.org/10.24432/C55S3H). The original data are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
