# VeriCare

**The right hospital, with the evidence to trust it.**

VeriCare finds which Indian hospitals really offer a life-saving service (emergency C-section, NICU, dialysis, trauma care, ICU, MRI and more) near a place or PIN code, and shows the evidence behind every answer. It also maps where care is missing, so planners know where to verify a facility and where to build one.

Built on Databricks for the CMU Datathon (Fall 2026, healthcare track).

| | |
|---|---|
| Demo video (2:30) | *add the YouTube / Drive link here* |
| Live app (Databricks Apps) | https://carefinder-7474654110161979.aws.databricksapps.com (needs a Databricks workspace login) |
| Code | this repository: `notebooks/` (data pipeline) and `app/` (Streamlit app) |

---

## The problem

India's facility listings are scraped from the web: directories, hospital pages, map listings. They are incomplete, out of date and sometimes wrong. A record can say "24x7 emergency" when the hospital closed years ago, sit 200 km from where it really is, or merge two hospitals into one row.

Picture a health worker in rural Pune with a mother who needs an emergency C-section. She doesn't need a list of "hospitals nearby". She needs to know **which hospital can actually do it**, how sure we are, and a number to call. **A confident wrong answer can cost a life.**

## Our thinking

We set ourselves four rules early on, and every design decision follows from them.

1. **Evidence, not keywords.** A hospital only counts as offering a service if its own text supports it: a procedure, a piece of equipment, a capability, a review. Every result quotes that text, so the user can judge it too.
2. **Never hide uncertainty, never fake certainty.** Every number is labelled **confirmed** (from the data or the hospital's own website) or **predicted** (from our model), and the ⓘ icon explains where it came from. Predictions come as ranges, and are shown only when the model is confident.
3. **Check the data against the real world.** We verified listings with the Google Maps API: corrected locations, flagged closed hospitals and kept every original value next to the change.
4. **A missing record is not missing care.** For planners we separate a *verification gap* (a facility claims the service but we can't confirm it: verify first) from a *care gap* (nothing in reach even claims it: build here). Treating the two the same would send money to the wrong places.

## What VeriCare does

**Find care** (for patients and health workers)
- Pick a service and a place: a city, district or 6-digit PIN code.
- Facilities in reach are ranked by **evidence (50%) + size (33%) + closeness (17%)**, multiplied by a confidence factor (strong 1.0, medium 0.85, weak 0.6). A predicted bed count counts ×0.8, so a confirmed number always beats a guess.
- Each result shows quotes from the facility's own text, distance, beds and doctors (confirmed or predicted, with the ⓘ explanation), phone, website and email, on a numbered map.

**Ask VeriCare** (chat)
- Questions in plain words, such as "NICU near 800001" or "Puneke paas MRI scan kaha hoga".
- Answers come only from our verified data, always with a contact table (phone, website, email).
- Fast path: our own service and place word lists, including Hinglish, answer without any language model. A language model (Databricks model serving or Gemini) handles everything else and writes the reply.

**12 Indian languages**
- English, हिन्दी, বাংলা, मराठी, తెలుగు, தமிழ், ગુજરાતી, ಕನ್ನಡ, മലയാളം, ਪੰਜਾਬੀ, ଓଡ଼ିଆ, اردو.
- The interface switches instantly (built-in translations), and chat replies follow the chosen language.

**Care gaps & policy** (for planners)
- For every service: how many people have verified care in reach, care that is only claimed, or nothing claimed at all.
- Districts are ranked by people beyond reach × local need (NFHS-5), and policy recommendations are generated from those numbers.

**India insights**
- Findings from the data, each with a chart and a map. Examples: the C-section paradox (too few and too many), the North-South divide, metro concentration, a state health scorecard.

## How it works

```
Raw listings (~10,000 facilities) + India Post PIN directory + NFHS-5 + Census
        │
 01 Clean ─────────► one trustworthy row per facility, location decided by vote
        │
 03 Google verify ─► location corrected / confirmed, closed hospitals flagged   (optional, needs API key)
        │
 02 Evidence ──────► per facility × service: score, quotes, confidence, rank
        │
 05 Website crawl ─► confirmed bed / doctor counts from hospitals' own sites
 04 Size model ────► predicted beds (under 50 / 50+) and doctors (80% range)
        │
 06 Care gaps ─────► PIN area × service: nearest verified facility, who is beyond reach
 07 Insights ──────► findings + state scorecard
        │
     Delta tables ──► Streamlit app on Databricks Apps (SQL warehouse + model serving)
```

### 01 · Cleaning (`notebooks/01_clean_data.py`)
- Repairs rows whose values slid into the wrong columns.
- Normalises state names and phone numbers (and flags numbers shared by many facilities).
- Drops websites that aren't real addresses, placeholder or impossible bed and doctor values, and duplicates.
- **Location voting:** up to six independent sources vote on where a facility really is: map coordinates, postcode, city field, address locality, "located in …" text, and the city in directory URLs. One wrong field can't move a hospital to another state.
- A merge-risk score flags records that seem to mix several facilities, and a final flag decides which records are safe for search.

### 02 · Evidence scoring (`notebooks/02_evidence.py`)
- For each service, curated patterns search procedures, equipment, capabilities, descriptions, Google reviews and specialties.
- Each field type carries a weight (procedure and equipment 3, capability, description and review 2, and so on).
- The result is a score with a breakdown, the exact quotes, warnings, and a **strong / some / weak** confidence label.

### 03 · Google Maps verification (`notebooks/03_google_verify.py`)
- Each facility is searched with the Places API, and a result is only accepted on a careful match: similar name plus nearby, same postcode, same city or same phone.
- **4,748 listings checked:**

  | Outcome | Count |
  |---|---|
  | Strong matches | 4,256 |
  | Weak matches | 231 |
  | Not found | 261 |
  | Locations corrected | 990 |
  | Closed hospitals flagged (excluded from search) | 53 |
- Originals are kept in `*_orig` columns, so every change can be traced.

### 04 · Predicting hospital size (`notebooks/04_size_estimate.py`)
Most listings never state beds or doctors, but size matters for an emergency.

- **Targets:** real counts, from the dataset field, then a number stated in the description, then the hospital's own website (05).
- **10 features:**

  | District and state context | The facility itself |
  |---|---|
  | District population (Census 2011) | Services offered (ICU, NICU, MRI, cath lab, dialysis …) |
  | State income per person | Breadth: number of specialties, procedures and equipment listed |
  | District wealth index (NFHS-5) | Ownership (public / private) and facility type |
  | District women's literacy (NFHS-5) | The other count (beds help predict doctors, and back) |
  | Distance to the state capital and nearest major city | Description text: 40 NLP topics, with numbers and bed / doctor phrases removed (no leakage) |
- **Models:** naive baseline, Ridge, random forest, extra trees, gradient boosting and an ensemble compete under **5-fold cross-validation**, and the best is chosen automatically. Beds use a classifier (under 50 / 50+): a text logistic regression stacked into gradient boosting. Doctors use quantile models plus conformal calibration for an honest 80% range. Everything is logged to MLflow.

**Results** (on facilities the model never saw):

| | Ours | Naive guess |
|---|---|---|
| Beds: correct class, on the 80% most confident (the ones we show) | **84.5%** | |
| Beds: correct class, all facilities | **78%** | 69% |
| Beds: balanced accuracy | **76.6%** | |
| Doctors: real count inside our predicted range | **80%** | |
| Doctors: right size band | **61%** | 33% |

**What the model actually leans on.** For beds, almost all of the signal comes from the description text. For doctors, it's the other count (45.5%), the text (21.8%) and the facility type (19.1%). District context, including women's literacy, carries little weight for an individual facility's size. Its value is at the district level, in Care gaps and Insights.

### 05 · Website crawler (`notebooks/05_website_crawler.py`)
- Reads "200-bedded", "50+ doctors" and similar phrases from hospitals' own sites.
- Polite crawling: respects robots.txt, one request per site at a time, a clear bot name.
- A chain guard ignores group totals shared by several branches (Apollo, Fortis …).
- Every value keeps the sentence and URL it came from, so the app can cite it.

### 06 · Care gaps (`notebooks/06_care_gaps.py`)
- Demand points: about 19,000 PIN-code areas, weighted by district population.
- For each service, the distance to the nearest **strong-evidence** facility, against a documented travel standard: 50 km for emergency C-section, NICU, trauma, ICU, dialysis and blood bank; 100 km for CT, MRI, cath lab and burn care.
- Priority = people beyond reach × (1 + need), with need taken from NFHS-5 district indicators.
- **Example:** for emergency C-section, about **46 crore people** live where no facility in reach even claims the service in this dataset.

### 07 · India insights (`notebooks/07_india_insights.py`)
Nine findings, each built from the data with its chart and source:
1. The C-section paradox
2. The North-South divide
3. Metro concentration
4. Insured but nowhere to go
5. The diabetes double burden
6. Aspirational districts
7. Public care is invisible in web data
8. A state health scorecard
9. The emergency distance gap

**Example: women's literacy as a signal of how well a health system reaches people** (NFHS-5, 706 districts, least vs most literate quarter of districts):

| | Least literate | Most literate |
|---|---|---|
| Mothers with 4+ antenatal check-ups | 50% | 71% |
| Births in a hospital | 84.5% | 92% |

## Numbers at a glance

| | |
|---|---|
| Facility records checked | 10,088 |
| Facilities with service evidence | 6,591 |
| Strong-evidence claims | 4,201 |
| Matched on Google Maps | 4,487 |
| PIN-code areas analysed | ~19,000 |
| Districts with NFHS-5 data | 706 |

## Tech stack

- **Databricks:** notebooks, Delta tables, SQL warehouse, MLflow, model serving (Llama 3.3 70B), Databricks Apps.
- **Python:** pandas, scikit-learn, LightGBM, NumPy.
- **App:** Streamlit, Plotly (clickable charts), Folium / OpenStreetMap.
- **APIs:** Google Maps Places API (New); optional Gemini 2.5 Flash for chat.

## Run it yourself

The app reads its tables from a Databricks SQL warehouse, so it runs on Databricks.

1. **Data.** In a Databricks workspace (Free Edition works), load the datathon's `facilities` table, the India Post PIN directory and the NFHS-5 table.
2. **Pipeline.** Run the notebooks in order: `01` → `03` (optional, needs your own Google Maps API key) → `02` → `05` (optional) → `04` → `06` → `07`. Each notebook's header lists its inputs and outputs.
3. **App.**
   - Create a Databricks App from the `app/` folder (`app.yaml` is the entry point).
   - Give it access to a SQL warehouse and, for chat, a model serving endpoint (`LLM_ENDPOINT` in `app.yaml`).
   - Optionally set `GEMINI_API_KEY` as an app secret.
4. **Never commit API keys.** Set them as secrets or environment variables.

```
app/
  app.py            the Streamlit app (Find care, Ask VeriCare, Care gaps & policy, India insights)
  translations.py   interface text for 10 more Indian languages (Hindi lives in app.py)
  app.yaml          Databricks Apps config
  requirements.txt
notebooks/          the data pipeline, 01 to 07
experiments/        local model experiments and their logs (feature and model selection)
data/               small exports and reference tables (NFHS-5, cleaning report)
```

## Limitations and honesty

- **Coverage.** About 10,000 web-sourced facilities is not all of India. Public PHCs and CHCs are underrepresented, so some "gaps" are data gaps, not care gaps. We say so in the app.
- **Evidence is text-based.** A facility can offer a service it never writes about, or claim one it no longer offers. Google verification and the confidence labels reduce this, but don't remove it.
- **Predictions are predictions.** Sizes are shown only for the most confident 80%, always labelled as predicted, with a range for doctors.
- **Ratings.** We don't use star ratings to rank hospitals. Reviews only serve as evidence text.
- **Not medical advice.** In an emergency, call 108. Always call the facility before travelling.
- **Translations** were machine-assisted and should be reviewed by native speakers before wider use.

## Team

*Add team member names and roles here.*

## Acknowledgements

- CMU Datathon organisers, for the facilities dataset and the NFHS-5 table
- India Post, for the PIN-code directory
- National Family Health Survey (NFHS-5), Census of India 2011
- OpenStreetMap contributors
