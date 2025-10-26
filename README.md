# Building Uber-Style ETAs from Scratch: NYC Taxi Rides, Real Maps, and a Pinch of ML

> Predicting "how long until it arrives?" looks simple on your phone. Under the hood, it’s maps + math + a model that learns traffic patterns you can’t see.

> ⚡ **Bonus:** This project is designed to help you *crack your Machine Learning System Design interviews* by demonstrating how to approach real-world ML pipelines end-to-end.

---

## TL;DR

* **Goal:** Predict trip duration from `(start_lat, start_lng, end_lat, end_lng, datetime)`.
* **Data:** NYC (2015) + SF (2012) taxi trips; cleaned weird coords, ocean points, zero/6h+ rides.
* **Features:** Great-circle & Manhattan distances, Google Maps metrics, time signals (hour/day/month/weekend), location clusters (DBSCAN), rain.
* **Models tried:** Linear, Ridge, Lasso, SVR, Random Forest, XGBoost, small NN.
* **Winner:** **XGBoost** — ~**5 minutes** average error (RMSE ≈ 306s), **R² ≈ 0.75**.
* **Why it works:** Combines road reality (gmaps), temporal rhythm (rush hour), geography (airports/city centers), and weather.

---

## Why ETAs Are Hard (and Why ML Helps)

Two rides of the same distance can take **8 minutes today** and **20 minutes tomorrow**. Reasons:

* **Traffic rhythm:** Rush hour vs 2am.
* **Road constraints:** One-ways, turns, bridges.
* **Local context:** Airports, CBD, stadiums, events.
* **Weather:** Rain slows everything down.
* **Noise:** GPS glitches, bad meters, logging errors.

A single formula can’t keep up. **Machine learning** learns patterns across millions of trips and generalizes to the next one.

---

## Dataset: Simple Columns, Messy Reality

* **Columns:** `start_lat, start_lng, end_lat, end_lng, datetime, duration (sec)`
* **Hidden complexity:** Two cities mixed (NYC & SF), holidays, airports, ocean points, zero-second trips, 6-hour marathons.

> The test set is the same **but without `duration`** — perfect for evaluating predictions.

---

## Data Processing & EDA (a.k.a. “Clean the chaos”)

### Quick sanity checks

* Drop nulls / dedupe
* Ensure types (`datetime` → pandas datetime)
* Histograms for longs/lats and duration to spot **bi-modal cities** and **long tails**

### Outlier handling with IQR (and some detective work)

Outliers for `start_lng` revealed “points in Kyrgyzstan.” Why? **Sign errors** — positive longitudes that should be negative.

```python
# Detect start_lng outliers and fix sign
lb, ub = iqr_bounds(train_df['start_lng'])
start_lng_outliers = train_df[(train_df['start_lng'] < lb) | (train_df['start_lng'] > ub)]
train_df.loc[start_lng_outliers.index, 'start_lng'] = -train_df.loc[start_lng_outliers.index, 'start_lng']
```

**End longitudes in the ocean?** Drop clearly invalid destinations (e.g., a spike around -60 to -50 longitudes with no landfall).

### Duration cleanup (zeros, long tails, and minutes)

* **Zero durations** (non-identical start/end) → remove.
* **> 2 hours** — business rule: remove because these are noisy/out-of-scope for “typical” rides.
* Convert **seconds → minutes** for readability.

#### “Business rule” vs IQR — what’s the difference?

* **IQR rule:** purely statistical; flags points far from the median spread (Q1–Q3).
* **Business rule:** domain-driven; e.g., “typical taxi trips are under 2 hours.” Enforces **use-case boundaries**.

You can (and often should) **use both**: IQR to catch obvious garbage, business rules to keep the target use-case sharp.

---

## Feature Engineering (turn columns into signals)

### 1) Distances: Straight-line, Manhattan-on-a-sphere, and Real roads

* **Haversine (great-circle) distance** — shortest path over Earth’s surface.
* **Manhattan distance** — mimics city-grid routes.
* **Google Maps metrics** — road-based distance & typical travel time.

### 2) Time signals (the heartbeat of a city)

* **Hour:** rush hour vs quiet time
* **Day of week:** weekdays vs weekends
* **Month:** seasonal patterns
* **Weekend flag:** binary indicator

### 3) Location clusters (airports, CBDs, suburbs)

DBSCAN finds hotspots like airports and downtowns, providing spatial context for the model.

### 4) Weather (rain slows everything)

Merge daily precipitation data; rain days generally lead to longer trip times.

---

## Modeling: Baselines → Boosting

**Split:** 80% train / 20% valid.
**Scaling:** MinMax for linear & NN (trees don’t need it).

**Models tested:** Linear, Ridge, Lasso, SVR, Random Forest, XGBoost, Neural Net.

### **Winner: XGBoost**

| Metric |             Score |
| :----- | ----------------: |
| RMSE   | 306 sec (≈ 5 min) |
| R²     |             0.747 |
| MAPE   |             42.6% |

**Why it worked:**

* Captures non-linear traffic patterns.
* Handles outliers gracefully.
* Provides interpretability through feature importance.

**Top Features:**

* Google Maps duration & distance.
* Hour of day (rush hours!).
* Euclidean/Manhattan distance.
* Start coordinates.
* Weather and clusters.

---

## Reality Check (Manual spot tests)

| Route                         | Predicted | Google ETA |
| :---------------------------- | --------: | ---------: |
| Times Square → Central Park   |    12 min |  11–15 min |
| JFK → Manhattan               |    52 min |  45–65 min |
| Brooklyn Bridge → Wall Street |     8 min |   7–12 min |

For historical data (2015) with no live traffic, that’s remarkably close!

---

## What Ride-Hailing Apps Likely Do (Conceptually)

1. **Routing engine:** road distance + base duration.
2. **Historical ML:** learns traffic rhythms.
3. **Live signals:** real-time traffic, surge, incidents.
4. **Personalization:** driver behavior, zone rules.
5. **Calibration:** keep ETAs *slightly optimistic* to improve UX.

Your model mirrors this real-world architecture—great system design discussion for interviews!

---

## Error Analysis & Guardrails

* **Short trips:** MAPE inflation; prefer RMSE.
* **Sparse regions:** use cluster labels.
* **API quirks:** keep fallbacks (Haversine, Manhattan).
* **Long trips:** separate model or cap scope.

---

## From Notebook to Production

* **Input:** `(start_lat, start_lng, end_lat, end_lng, datetime)`
* **Transform:** feature pipeline (distances, time, clusters, weather)
* **Predict:** XGBoost → seconds → minutes
* **Serve:** FastAPI endpoint or Flask app
* **Monitor:** track prediction error vs reality
* **Retrain:** monthly with new trips

> 🎯 *Interview Tip:* In ML design interviews, emphasize **data cleaning**, **feature pipeline**, **model monitoring**, and **retraining strategy**.

---

## Next Steps

* **Add live traffic data** via API integrations.
* **Event calendar** awareness (holidays, parades).
* **Weather nowcasts** for better precision.
* **Confidence intervals** for ETA ranges.
* **Multi-city transfer learning**.

---

## Closing Thoughts

From *“are they faking the ETA?”* to *“I built my own version!”* — this project showcases real-world ML design thinking. You clean, engineer, and model messy city data to deliver meaningful predictions.

Mastering projects like this is how you **crack ML design interviews** — by showing you understand not just *models*, but *systems*.