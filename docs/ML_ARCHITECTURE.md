# MOL O2C Lead Time Analytics – ML & Advanced Analytics: architecture preparation

Status update: the **Machine Learning & Advanced Analytics tab is live with two subpages – EDA and Correlation Analysis** (computed by the Python backend, reusing the dashboard filters plus a T1/T2/RS selector). The last subpage (Model Evaluation & Comparison) is visible as "planned" and wait for the business confirmations in section 5. The Overview and Median Lead Time pages are unchanged (their validation table still shows 8/8 matches); the only change in their code path is one branch at the start of `show()` for the new tab.

## 1. Inspection of the current application
| Layer | Today | Consequence for ML |
|---|---|---|
| Data prep | `build_data.py` reads sheet 1 of the Excel and writes `data/orders.json` (columnar, rounded, 2 helper columns dropped) | JSON is a frontend format; ML needs the full, typed, versioned dataset |
| Business logic | Duplicated inside `index.html` (step mapping, filters, region/allocation/category derivation, 48 h flag) | A second copy in Python would drift -> one shared schema is needed |
| Frontend | Single static HTML file, filters in a JS object `F` | Filters must become an API contract; the ML tab must reuse them without a rewrite |
| Serving | `python -m http.server` | A separate backend process is needed for Python ML (CORS enabled for :5500) |

Verified data facts that shape the ML design (from the actual workbook, 14,601 rows):
* `Sales Document` is unique per row (one row = one order **item**, e.g. `401625043#20`); 12,040 distinct base documents -> items of one order are correlated -> **grouped** validation.
* All duration columns are stored in **days**; the processed dataset adds `h_*` hours columns (x24).
* `>48 SO-OC` / `>48 PO-OC` equal 1 when the time is **<= 48 h** (name is inverted). Validated against `h_po_to_oc`.
* `Lead time relevancy` is empty (the file contains only in-scope orders). Z5 removal is missing for 21.9 % of rows (treated as missing, never 0).
* Negative durations exist: Scheduled->Z5 27, Z5->Confirmation 56, Confirmation->Loading 351, SO->OC 7. Flagged, never silently removed (`exclude_negative_durations` switch, as in the dashboard).
* Targets are strongly skewed (median PO->OC 80.1 h vs sum of step medians ~27 h) -> models must use robust metrics and consider log/quantile targets.

## 2. What was prepared
```
pipeline/schema.py      single source of truth: events, 16 lead-time measures, filters, column availability, prediction points
pipeline/processed.py   Excel -> data/processed/orders_processed.parquet + schema.json (data dictionary + quality report + dataset id)
build_data.py           unchanged output, plus calls the pipeline (skips gracefully without pyarrow)
backend/app/core        dataset loader (parquet, cached), FilterSpec (dashboard filters as API contract)
backend/app/api         implemented: schema, filter options, summary (dashboard parity), prediction points, leakage-guarded feature lists
                        planned (HTTP 501 with contract): 8 ML endpoints
backend/app/ml/*        empty, separated layers: features, stats, models (BaseLeadTimeModel + registry), evaluation, prediction, explain
backend/tests           contract tests: backend reproduces dashboard numbers (14,601 / 11,929 / T1 4,825; 80.1 h; 40 % / 38 % / 31 %)
```
Original Excel column names are preserved. Derived columns: `d_*` (dimensions: `d_region_code`, `d_allocation_group`, `d_customer_category`, `d_early`, `d_within_48h_po_oc`, `d_sales_document_base`) and `h_*` (hours of each measure).

## 3. Data model
`data/processed/orders_processed.parquet` – one row per order item, all original columns + `d_*` + `h_*`.
`data/processed/schema.json` – dataset id (md5 of the Excel), build time, per-column dtype/missing/**availability event**/role, per-measure source column/start/end event/valid & negative counts/median, filter registry, quality report. Every trained model must store this `dataset.id` (reproducibility).

Measures (id -> Excel column): 7 sequential (`po_to_so_sap` ... `oc_to_loading`), 7 aggregated (`po_to_oc_agg`, `erp_to_z5`, `erp_to_sm_agg`, `sm_to_ss_agg`, `erp_to_oc`, `erp_to_price_match`, `po_to_loading`), 2 derived (`so_to_oc`, `po_to_oc`). Sequential and aggregated are never mixed.

## 4. Filter contract (dashboard <-> API)
| Dashboard key (`F`) | API id | Processed column |
|---|---|---|
| reg | region | d_region_code |
| Company | company | Company |
| Sales Document type | doc_type | Sales Document type |
| ag | allocation_group | d_allocation_group |
| Source of Allocation | source_of_allocation | Source of Allocation |
| PNGroup | pn_group | PNGroup |
| Sales Rep | sales_rep | Sales Rep |
| Material group 5 | material_group_5 | Material group 5 |
| Distribution Channel | distribution_channel | Distribution Channel |
| early | early | d_early |
| Month of Allocation | month_of_allocation | Month of Allocation |
| (initial category) | customer_category | d_customer_category |
Semantics: AND across filters; a filter accepts a list (dashboard sends one value). Same request body for every endpoint -> the ML tab simply forwards the existing filter state.

## 5. Leakage prevention (implemented as data, not prose)
Process events in order: purchase_order -> sales_order_sap -> erp_order -> scheduling_match -> scheduled_status -> z5_removal -> order_confirmation -> real_loading.
A **prediction point** = an event. A column is a legal feature only if its availability event is at or before the point; a measure is a legal target only if it ends after the point. `GET /api/v1/ml/features/{point}` returns allowed features, valid targets, exclusions with reasons, and a `pending_business_review` list.
Example, `at_purchase_order`: Region, Company, Sales Rep, PO date (and its derived calendar features) allowed; Loaded volume, Z5 removal, Order Confirmation Date, Loading Date, all durations after PO, the >48 flags: excluded.
Always excluded: identifiers, free text, Z5 change-log fields, outcome flags, high-cardinality `Sold-to party`/`Name` (until a grouping/encoding strategy is chosen).

**Needs business confirmation (currently excluded, 'review'):** `Allocation Date`, `Month of Allocation`, `korai / nem korai rendelés` (is "early" known at PO time or derived afterwards?), `Sale Plan`, `diff between sale plan and order`, `QualityLevel`, `Price match date`, `Past due reason`. Confirm their timing, then change one line in `pipeline/schema.py`.

## 6. Validation policy (one policy for all models)
* Chronological split on `Purchase Order Date` (train past -> test future) as the headline result; grouped K-fold on `d_sales_document_base` as cross-validation. Never random row splits.
* Same split, same filters, same target transform for every model -> fair comparison.
* Baselines always reported: historical median (global and per-region).
* Metrics: R2, adj. R2 (linear), MAE, RMSE on unseen data; residuals; n per analysis; warning below 300 observations (`MIN_ROWS_FOR_MODELLING`).
* Insights are generated from computed numbers; wording separates association / predictive importance / causation.

## 7. API structure (`/api/v1`)
| Method + path | State | Purpose |
|---|---|---|
| GET /health | live | liveness |
| GET /meta/schema | live | data dictionary, measures, events, quality |
| GET /meta/filters | live | option lists for all filters |
| POST /data/summary | live | n rows/documents, median PO->OC, <=48 h rate, modelling warning |
| GET /ml/prediction-points | live | prediction points |
| GET /ml/features/{point} | live | leakage-guarded features/targets |
| POST /ml/eda | **live** | distributions, box/violin data, outliers, missing/duplicates, group comparisons, generated insights (EN/HU) |
| POST /ml/correlation | **live** | Pearson/Spearman matrix with p-values, eta + Kruskal-Wallis for nominal variables, pairwise scatter, generated insights |
| POST /ml/regression/linear | **live** | OLS y~x, equation, slope CI, test/train metrics, residuals, grouped CV |
| POST /ml/regression/multiple | **live** | OLS with leakage-guarded features, coefficients, VIF, test metrics, grouped CV |
| GET /ml/candidates/{point} | **live** | candidate explanatory variables with allowed/blocked + reason |
| POST /ml/regression/random-forest | **live** | RF (n_trees/max_depth/min_samples_leaf), grouped CV, permutation + impurity importance, comparison with linear regression |
| POST /ml/interpretation | **live** | permutation + impurity importance, SHAP, per-category contributions, numeric dependence, single-order explanation, variable coverage, generated insights |
| POST /ml/prediction | **live** | prediction + conformal-style interval + comparison with historical medians + contributions + confidence/limitations; returns the input form definition |
| POST /ml/evaluation/compare | planned | comparison table |
Long-running training (RF, SHAP) -> later `POST ... -> job_id`, `GET /ml/jobs/{id}`; results cached by (dataset.id, filters, params).

## 8. ML backend architecture
`data prep (pipeline/) -> features (leakage + encoding) -> stats | models -> evaluation (shared splits) -> explain -> prediction -> API (JSON for charts)`. Layers do not import each other upward; the API layer only calls services. New algorithms: subclass `BaseLeadTimeModel`, decorate with `@register("name")`; evaluation/comparison pick them up from `MODEL_REGISTRY`. Planned libs: pandas, numpy, scikit-learn, statsmodels, shap, FastAPI (`backend/requirements-backend.txt`, commented).
Model artifacts (joblib + JSON card): dataset id, filters, prediction point, features, params, split definition, metrics, training time.

## 9. Frontend integration plan (not built)
1. Add a third nav button; ML sub-navigation (8 pages) lives in its own script/module.
2. Reuse the existing filter bar: expose `F` as `{api_id: [values]}` via the table in section 4 and send it as the request body.
3. Thin API client with base URL `http://localhost:8000`; if the backend is down, show an "ML backend offline" state – the two existing pages never depend on it.
4. Charts for histograms/box/violin/heatmap/SHAP: add a charting library when the tab is built; consider moving the monolithic `index.html` to Vite + React + TypeScript at that point (the API contract makes this independent).

## 10. Open items / risks
* Business confirmation of the 'review' columns (section 5) – decides which predictors are legitimate.
* `Loaded volume` is an actual (post-loading) value: not usable for pre-loading prediction; a planned/ordered volume field would be needed.
* Potential selection effect: only orders that reached loading in Jan–Jun are in the file (right-censoring of very slow orders).
* Early orders are structurally different (0 % within 48 h); model them separately or as an explicit feature once its timing is confirmed.
* Production deployment (auth, hosting, data refresh) is out of scope for this phase.

## 11. Roadmap
1. Confirm review columns -> 2. EDA + correlation endpoints (done) -> 3. linear/multiple regression + shared evaluation (done) -> 4. random forest + importance/SHAP -> 5. prediction page + insight generator -> 6. frontend tab.

## 12. ML tab – how it works (live part)
* Request body (EDA / correlation): `{ "filters": FilterSpec, "measure": "<measure id>", "lang": "en|hu", ... }`; the dashboard filter state is mapped to API ids by `FMAP` in `index.html` (section 4).
* Nominal variables are **not** turned into ordinal numbers. Their association with the target is the correlation ratio eta (share of variance explained by the category groups) with a Kruskal-Wallis test; numeric pairs get Pearson and Spearman with p-values.
* Correlations between a total (PO->OC) and its own steps are flagged as structural, not as findings.
* Every analysis shows its observation count; selections below 300 rows show a warning, below 10 rows are refused (HTTP 422).
* Insights are text templates filled only with computed numbers. They always end with the correlation-is-not-causation note.
* `Early / Non-early` currently shows the strongest association with PO->OC (eta ~0.66). Its timing relative to the order is unconfirmed, so it is excluded from future prediction features until confirmed (section 5); the UI says so.

## 13. Regression modules (live part)
* **Validation (shared, `evaluation/splits.py`):** chronological hold-out on `Purchase Order Date` (last 20 % of orders = test), all items of an order on the same side; grouped 5-fold CV on the training period. Metrics on the test period: R2, adj. R2, MAE, RMSE, always next to the naive baseline "train median".
* **Leakage guard (`features/catalog.py`):** only variables known at the chosen prediction point are selectable; blocked ones are listed with the reason. Variables with unconfirmed timing ("review") are off by default and marked when switched on.
* **Encoding (`features/encoding.py`):** fitted on training rows only; nominal -> one-hot with the most frequent category as reference, rare levels -> "Other", missing -> "Missing".
* **Collinearity:** nominal variables that fully determine each other (Sales Rep -> Region, Source of Allocation -> Allocation group / T1/T2/RS) are detected automatically; the one earlier in the catalog is kept, the other is excluded with the reason. VIF is reported per coefficient and per variable.
* **Inference:** statsmodels OLS with HC3 robust standard errors (lead times are strongly heteroscedastic). Optional ln(1+y) target: effects become percentages, metrics are reported back in hours.
* **Current finding (at_sales_order, default variables):** test R2 ~0.19, MAE ~78 h vs 79 h for the naive median -> the information known early explains little of PO->Order-confirmation time. Region (R02/R05 vs R03) and PNGroup (T1 vs T2/S) are the clearest associations; the unconfirmed "early" flag would explain much more but cannot be used until its timing is confirmed.
* Simple regression is a descriptive tool: a leakage warning is shown if the explanatory variable is only known after the target has ended (e.g. Loaded volume) and a structural warning if it is a component of the target.

## 14. Random forest (live part)
* Same pipeline as the linear models: identical chronological hold-out, leakage guard, encoding and variable selection (nested variables removed for both models), so the comparison table (naive median / multiple linear / random forest) is like-for-like.
* Parameters in the UI: number of trees (10–500), maximum depth (0 = unlimited), minimum orders per leaf; `max_features` is fixed at 50 %. Defaults (150 trees, depth 12, leaf 20) are deliberately conservative against overfitting; the page reports the overfit gap (train R2 − test R2) and warns above 0.15.
* Importance: permutation importance on the **test period** (MAE increase when a variable's columns are shuffled together) next to impurity importance; both aggregated per original variable. Implemented as a result of the model, interpreted in business language only through the generated insights.
* Current finding (at_sales_order): forest MAE ~75.8 h vs linear 78.2 h vs naive median 79.1 h, but test R2 0.16 (linear 0.19). Inside the training period the forest reaches R2 ~0.39 (CV) and ~0.45 (train), on the later test period only 0.16 -> the relationships drift over time; Region is by far the most influential variable.
* SHAP and the dedicated interpretation page are still planned (`/ml/interpretation`).

## 15. Feature importance & model interpretation (live part)
* Model: the same leakage-guarded random forest as in section 14 (same split and variables). Everything is computed on the **unseen test period** where possible.
* Three importance measures per original variable (dummies of one variable are shuffled/summed together): permutation importance (MAE increase, 5 repeats, with spread), SHAP mean |contribution| (exact TreeSHAP on a 400-row test sample) and impurity importance. Rank agreement between permutation and SHAP is reported.
* Contribution by category: average SHAP contribution of each category (e.g. Region R02 +80 h vs the average prediction); numeric variables get a SHAP-vs-value plot.
* Order explanation: enter a Sales Document; the page shows average prediction, prediction, actual value and the largest contributions; it states whether the order was in the training or test period.
* Variable coverage table: every candidate variable (incl. Loaded volume, Source of Allocation, Material group 5, early flag) with "used" or "excluded" + reason (leakage guard, constant, perfectly collinear).
* Insights are generated from the numbers (top variables, share of the top variable, permutation/SHAP agreement, strongest category effects, direction of numeric effects, number of excluded variables, weak-model caveat). They always state that importance is predictive, not causal.
* SHAP is optional: if the `shap` package is missing the page still shows permutation and impurity importance and says so.
* Finding that matters for the business review (section 5): with the unconfirmed "review" variables switched on, **Early / Non-early** (+31 h) and **ERP order >> Price match** (+12 h) become the dominant variables - exactly the ones whose timing must be confirmed before they may be used for prediction.

## 16. Lead time prediction (live part)
* **Prediction point** is explicit (purchase order arrives / sales order created in SAP / ERP order created / after scheduling match). The form offers only variables known at that point (same leakage guard as everywhere); a target that already ended at that point is refused (HTTP 422).
* **Split:** train | calibration | test in chronological order, all items of an order on the same side (about 60/20/20).
* **Models:** random forest (default) or multiple linear regression. The delivered predictor is the one that was calibrated and tested, so the reported accuracy applies to it; refit regularly with new data.
* **Uncertainty:** interval = prediction + signed quantiles of the calibration residuals (split-conformal style, asymmetric because lead times are skewed; lower bound clipped at 0). The page reports the interval's **measured coverage on the test period** next to the nominal level - currently ~86 % vs 80 % at the PO point - and warns when coverage falls below nominal - 10 pp (drift).
* **Historical comparison:** median / middle half of similar past orders (all selected categorical inputs, relaxed until >= 30 orders) and of all orders. For targets ending at the order confirmation the share confirmed within 48 h among similar orders is the headline figure; the model-based probability is shown as a rough secondary estimate (it assumes equal uncertainty for every order).
* **Factors:** SHAP contributions for the random forest (fallback and for linear: effect of setting each variable to its typical value). Warnings for categories rarely/never seen and for numeric inputs outside the 1st-99th percentile; the prediction is an average-type estimate and therefore above the median of similar orders (right-skewed data) - the page says so.
* **Confidence grade** (low / moderate / high) comes from the test R2 (<0.2 / <0.5 / otherwise). Current honest verdict at the PO point: **low** (test R2 ~0.18, MAE ~74 h vs 79 h for the median); the page repeats the limitations and that what-if changes of inputs compare order types, they are not causal effects.

## 17. Model evaluation & comparison (live part)
* **One protocol for every model:** same variables (leakage-guarded), same chronological hold-out (last 20 % of orders = test), same grouped 5-fold CV on the training period. Numbers are identical to those on the regression and forest pages (checked by an automated test).
* **Models** (registered in `models/adapters.py`, so a new algorithm only has to register itself): naive median, median by region, linear regression on the best single variable (chosen by CV on the training period only), multiple linear regression, random forest, gradient boosting (squared-error loss) and gradient boosting with absolute-error loss.
* **Why two boosters:** mean-type models (squared error) are structurally disadvantaged under MAE on right-skewed data, while a median baseline is MAE-optimal. The absolute-error booster is the fair opponent for the median baselines; the page also explains the MAE/RMSE difference.
* **Evidence beyond a single split:** monthly rolling-origin validation (train on all earlier months, test on the month), paired per-order bootstrap CI and Wilcoxon test vs the baseline and vs the best model, error distribution (box plot of actual - predicted) and MAE per region.
* **Current finding (at_sales_order, PO -> Order confirmation):** gradient boosting with absolute-error loss is best on the test period (MAE 69.4 h, 12 % below the naive median; saving 9.3 h per order, 95 % CI 6.8-11.6 h) and in cross-validation. Median by region (73.5 h) is a strong simple benchmark; mean-type models are better by RMSE (multiple linear 114.5 h). The winner changes between months (random forest in March, boosting later) and the tree models overfit (train R2 0.3-0.5 vs test ~0.1-0.16) - the relationships drift over time. The best R2 stays low (<= 0.19): most of the variation is not explained by information known early.
* **Recommendation logic:** lowest test MAE, confirmed or not by cross-validation (the page says which), with the caveats above.

## 18. Export tab (PDF report)
* **What it does:** a fourth main tab, *Export*, builds a PDF report from the current state of the dashboard. **Overview is always included**; *Median Lead Time* and *Machine Learning & Advanced Analytics* are optional (any combination). For Machine Learning the individual subpages can be ticked (default: EDA, Correlation, Multiple linear, Random forest, Interpretation, Evaluation; Linear regression and Prediction depend on user-chosen inputs and are off by default).
* **Filters are part of the report:** the first page lists every active filter, the Negative-durations switch, the number of orders in the selection and (with ML) the T1/T2/RS selector; every ML page repeats its own settings (target measure, prediction point, model parameters, order explained, ...). The language (HU/EN) of the dashboard is used.
* **How it is built:** the report re-uses the dashboard's own page renderers (so numbers are identical to the screen), strips the interactive controls, translates the text and wraps everything in a print layout (A4, light theme, page numbers, tables break across pages with intact rows, headings kept with their content). The ML pages are fetched from the backend one after another with the same endpoints as the ML tab.
* **Preview:** the finished report is shown in an embedded preview; it becomes "stale" (with a warning) as soon as a filter or option changes.
* **PDF download (two paths):** (1) *Download PDF* posts the report HTML to `POST /api/v1/export/pdf`; the backend prints it with the **Chrome / Edge / Chromium already installed** on the machine (headless `--print-to-pdf`, no extra Python package; set `O2C_CHROME` to point to a specific browser; `GET /export/status` tells whether it is found). (2) If no browser is found or the backend is offline the dashboard automatically opens the browser's print dialog - choose *Save as PDF*.
* **Safety:** scripts and inline handlers are removed from the report before rendering; CORS is limited to localhost.
* **Tested:** the report was built with filters in Hungarian (overview + median + six ML pages = 22 A4 pages) and printed with a real Chromium engine; pages were inspected as images. The backend plumbing is covered by automated tests with a stand-in browser. A found bug on the way: the rolling validation on the comparison page failed for filtered selections whose months had different category levels (fixed, with a regression test).
