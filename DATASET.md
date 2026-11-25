# Dataset provenance

Original listing: [Logistics and supply chain dataset](https://www.kaggle.com/datasets/datasetengineer/logistics-and-supply-chain-dataset), published by **DatasetEngineer**. Kaggle metadata lists **CC0: Public Domain**, last updated 2024-10-20. The listing and public download were checked on 2026-11-25. The repository's original acquisition date is unknown.

The downloaded archive contains `dynamic_supply_chain_logistics_dataset.csv`. Its SHA-256 is identical to the repository file:

```text
3b90b86fb658823687e461f6444dc10bbde325ea1c37733da92b0dad7071b10e
```

Metadata endpoints used for verification:

- https://www.kaggle.com/api/v1/datasets/view/datasetengineer/logistics-and-supply-chain-dataset
- https://www.kaggle.com/api/v1/datasets/download/datasetengineer/logistics-and-supply-chain-dataset

The publisher describes Southern California logistics observations and identifies GPS, IoT, warehouse systems and external providers as sources. These claims have not been independently authenticated. The description's end date is January 2024; the actual CSV extends to 2024-08-29. No generation code or named collecting organization is supplied in the listing. Real versus synthetic origin remains unverified.

The publisher defines ETA variation using actual arrival and fulfillment status using whether fulfillment was on time. Both are excluded as outcome information. The operational timing of the remaining measurements still needs verification before prospective use. The source describes some status inputs as binary, but the CSV contains continuous values; do not silently recode these without clarification.

Independent inspection of all 32,065 local rows found that `risk_classification` is reproduced exactly by `disruption_likelihood_score` thresholds of 0.3 and 0.7. This is an empirical relationship, not verified label-generation code. The score and any features derived from it are excluded. Other target-like scores and actual delivery deviation are also excluded. This project predicts the supplied risk labels, not independently measured delay events.
