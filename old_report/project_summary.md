### Industry 4.0 Misinformation Analytics

#### 1. Problem identification

In this project, we frame online misinformation moderation as a **Quality 4.0 optimisation problem** inside a cyber-physical "social factory". Social media posts are treated as products moving through a production line, harmful misinformation is the defect, and human fact-checkers are the constrained inspection resource.

The operational goal is to decide, for each post *i*, whether to send it to manual review. For each post we estimate a risk score
$p_i = P(\text{misinformation} \mid \text{features})$ and have an estimate of its expected engagement (reach). The optimisation objective is to **maximise harmful engagement intercepted**:

> Maximise $\sum_i x_i \cdot (p_i \times \text{engagement}_i)$

where the **decision variable** is

* $x_i = 1$: post *i* is routed to fact-checkers
* $x_i = 0$: post *i* is left to the automated pipeline

subject to realistic Industry 4.0 style **constraints**:

* **Global review capacity**
  $\sum_i x_i \le K$,
  where $K$ is the maximum number of posts that can be manually reviewed per day.

* **Platform-level quotas (optional)**
  $\sum_{i \in \text{platform } j} x_i \le K_j$,
  so that no single platform (e.g., Twitter) consumes the entire review capacity.

* **Reviewer time budget (optional)**
  $\sum_i x_i \cdot \text{token\_count}_i \le T$,
  where `token_count` approximates reading/inspection time and $T$ is the total reviewer time available per cycle.

This mirrors **supplier screening in Quality 4.0**: suppliers $\approx$ authors, lots $\approx$ posts, defect probability $\approx$ misinformation risk, and the reviewer workforce $\approx$ constrained inspection cell. The key Industry 4.0 decision is not only "is this post risky?" but **"given limited human capacity, which risky posts should be seen first so that we intercept the maximum harmful engagement?"**



#### 2. Relevant data

To support this optimisation problem, we constructed a dataset of **500 posts** from **Twitter, Facebook, Telegram, and Reddit** over 12 months, containing **31 raw variables + 4 engineered features** (posting hour, human-readable label, log-scaled followers/engagement). The target indicates misinformation vs legitimate content.

Features reflect multiple Industry 4.0 information streams:

1. **Platform, geography, and time**: Platform type, country (US, India, Germany, Brazil, UK), temporal patterns (month, weekday, hour) for risk patterns and capacity rules.

2. **Author traits and reach**: Follower count (log-scaled), verification status, engagement metrics (reactions, comments, shares) as **harm proxy** in the objective.

3. **Content and linguistic signals**: Text length, token count, readability, sentiment/toxicity scores, URL/mention/hashtag counts.

4. **Reliability and AI provenance**: Source domain reliability, synthetic text detection, **generator model signature**, embedding similarity to factual content, fact-check counts/verdicts.

Key findings:
* **Misinformation prevalence**: ~54% (naïve baselines inadequate)
* **Twitter** shows highest misinformation rates; platform–country hotspots emerge (Twitter–US, Telegram–India)
* Clear **temporal patterns** (May/September spikes, Friday/Monday peaks) support adaptive staffing
* Misinformation posts are **longer, more negative, linked to lower-reliability domains**; engagement distributions similar across classes

The dataset provides exactly what optimisation requires: misinformation probability $p_i$, engagement-based harm measure, and contextual variables for realistic Industry 4.0 constraints.



#### 3. Approach to solve the problem

Our solution follows a **three-stage pipeline**: descriptive analytics → predictive modelling → prescriptive optimisation.

##### a) Exploratory analytics

We began with detailed **exploratory data analysis (EDA)** to understand where risk concentrates and to justify feature choices:

* Profiled class balance, platform-wise misinformation rates, and platform–country heatmaps.
* Analysed monthly, weekday, and hourly trends to identify **temporal “campaign-like” spikes**.
* Examined follower and engagement distributions (heavy-tailed reach, similar engagement for both classes).
* Compared content-level features (length, tokens, readability, sentiment, toxicity, URLs/mentions/hashtags) between misinformation and legitimate posts.

This EDA established that risk is **multi-factorial and diffuse**, and it highlighted the specific segments (platforms, time windows, author types) for scenario analysis in the optimisation stage.

##### b) Predictive modelling

Next, we built a **machine-learning pipeline** to estimate $p_i = P(\text{misinformation} \mid \text{features})$ for each post:

* **Text representation**

  * TF–IDF bigrams (1–2-grams), limited to 500 features
  * TruncatedSVD (50 components) to obtain dense “topic” dimensions from the sparse TF–IDF matrix

* **Feature fusion**

  * Concatenated SVD text topics with scaled numeric metadata (followers, engagement, length, sentiment, toxicity, reliability, synthetic score, similarity, token_count)
  * Added one-hot encoded categorical variables (platform, country, month, weekday, hour, fact-check verdict, generator signature)

* **Models**

  * Class-balanced **logistic regression** (saga solver) as a linear baseline
  * Class-weighted **Random Forest** with 400 trees as a non-linear ensemble

Using a **70/30 stratified train–test split**, the Random Forest delivers:

* **Recall on misinformation** $\approx$ 83.8% – it correctly flags more than 5 out of 6 harmful posts.
* **Precision** $\approx$ 54% – about half of flagged posts are truly harmful.

ROC-AUC is low due to mis-calibrated probabilities, but the **ranking and class predictions** are strong enough for triage. Feature-importance analysis shows that no single feature dominates; instead, **token_count, sentiment, toxicity, engagement, domain reliability, and text topic components** collectively drive decisions. This matches the EDA conclusion that the signal is spread across many weak indicators.

##### c) Prescriptive optimisation and Industry 4.0 link

Finally, we turn model outputs into **operational decisions** via a risk-weighted prioritisation policy. For each post:

* Compute expected harm
  $h_i = p_i \times \text{engagement}_i$
* Sort posts in descending order of $h_i$.
* For any review capacity $K$, send the top $K$ posts to reviewers.

We evaluate this policy using a **coverage curve** that compares random review and risk-weighted review for different capacities (10, 20, 50, 100, 150 posts). The results show that:

* Reviewing only the **riskiest 30% of posts captures ~56% of total harmful engagement**, almost **2×** better than random sampling at the same capacity.
* Even at 10–20% capacity, the prioritisation consistently delivers around **1.8–1.9× more harmful engagement intercepted per reviewed post** compared to random review.

From an **Industry 4.0 perspective**, this closes the loop from analytics to action:

* Sensor-like data streams (post content and metadata) →
* Digital intelligence (predictive model estimating $p_i$) →
* Optimisation layer that **allocates scarce human "inspection" capacity** where it reduces risk the most.

This architecture can be extended into a **digital twin of the moderation process**, with adaptive thresholds based on temporal risk, continuous feedback from moderators, and fairness constraints across platforms and regions.

**Repository**: [github.com/ALikesToCode/industry-4.0-gaabs-analysis](https://github.com/ALikesToCode/industry-4.0-gaabs-analysis/settings)
