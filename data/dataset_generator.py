"""
Dataset Generator & Pre-configured Benchmarks
Generates realistic streaming datasets across the 3 domains specified in the base paper:
1. Sports: #manutd
2. Finance: #stockmarket
3. Healthcare: #covaxinwho

Each dataset is structured into consecutive temporal streaming windows (w0, w1, w2, w3, w4, w5)
with ground truth labels, timestamps, topic IDs, and simulated drift events (sentiment drift,
topic drift, co-drift, and stationary periods).
"""

import os
import random
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

# Domain keyword lexicons (D) used for Domain Impact Score (DIS) calculation
DOMAIN_LEXICONS = {
    'manutd': {
        'manutd', 'manchester', 'united', 'derby', 'premierleague', 'ronaldo', 'rashford',
        'manager', 'goal', 'score', 'match', 'defense', 'oldtrafford', 'striker', 'champions',
        'football', 'tenhag', 'rangnick', 'clean', 'sheet', 'tactics', 'trafford', 'reddevils'
    },
    'stockmarket': {
        'stockmarket', 'stocks', 'nasdaq', 'sp500', 'bull', 'bear', 'inflation', 'fed',
        'interest', 'rates', 'shares', 'dividend', 'earnings', 'rally', 'portfolio', 'crypto',
        'dow', 'investors', 'trading', 'equities', 'recession', 'wallstreet', 'revenue'
    },
    'covaxinwho': {
        'covaxinwho', 'covaxin', 'who', 'vaccine', 'health', 'trials', 'approval', 'efficacy',
        'dose', 'immunization', 'pandemic', 'safety', 'clinical', 'suspension', 'bharat',
        'biotech', 'dossier', 'regulatory', 'travel', 'inspection', 'gmp', 'authorization'
    }
}

# Tweet templates representing different topics and sentiments for each domain
SPORTS_TEMPLATES = {
    'topic_0_match': {
        1: [
            "What an unbelievable goal for #manutd! Brilliant team play at Old Trafford 👏",
            "Dominant performance today by #manutd, 3-0 win and complete control of the match!",
            "Ronaldo with another masterclass for #manutd, clutch performance in the Premier League 🏆",
            "Superb pressing and attack from #manutd, deserved 3 points today!",
            "Clean sheet and clinical finishing. Best #manutd performance of the season! 🔥",
            "Rashford looking sharp and dangerous again, fantastic win for #manutd!"
        ],
        0: [
            "Terrible defending from #manutd, conceding 4 goals in the derby is unacceptable 💔",
            "Embarrassing defeat against City. Total disaster for #manutd players.",
            "Shocking lack of effort from the #manutd squad today. Heartbreaking loss.",
            "No midfield control, disastrous backline. #manutd looked completely hopeless today.",
            "Humiliating result at Old Trafford. How can #manutd play this badly? 😡",
            "Defensive errors cost #manutd once again. Absolutely devastated after that derby match."
        ]
    },
    'topic_1_tactics': {
        1: [
            "The tactical adjustments made by the manager worked perfectly for #manutd today.",
            "High pressing structure and quick transitions are really improving #manutd style.",
            "Tactical discipline was on point today, great game management from #manutd coaching staff."
        ],
        0: [
            "Manager has no tactical clue. #manutd substitution choices were dreadful.",
            "Completely out-tacticked and outworked. When will #manutd learn from past mistakes?",
            "Tactical bankruptcy at #manutd. The formation left enormous gaps in midfield."
        ]
    },
    'topic_2_ownership': {
        1: [
            "Exciting takeover rumors surrounding #manutd, fans are hopeful for a brighter future!",
            "New club leadership investing in infrastructure is positive news for #manutd."
        ],
        0: [
            "Fans protesting outside the stadium again. Club management has failed #manutd for a decade.",
            "Financial mismanagement and broken promises. Time for a complete overhaul of #manutd ownership."
        ]
    }
}

FINANCE_TEMPLATES = {
    'topic_0_earnings': {
        1: [
            "Huge earnings beat for tech giants! S&P 500 and Nasdaq surging on record revenues 🚀",
            "Strong quarterly results driving solid gains across the #stockmarket today 📈",
            "Dividend hike and massive share buyback announced! Outstanding fundamentals for stocks.",
            "Tech revenue growth accelerates, pushing index higher into green territory!"
        ],
        0: [
            "Tech earnings miss expectations by wide margin. Nasdaq shares tumbling sharply 📉",
            "Guidance slashed for next quarter, heavy selloff spreading across the #stockmarket.",
            "Disastrous financial reports triggering steep drops in blue chip equities.",
            "Profit margins collapsing due to rising operating costs. Bear market concerns grow."
        ]
    },
    'topic_1_fed_macro': {
        1: [
            "Fed signals pause on interest rate hikes! Market rallies as inflation cools down.",
            "Consumer sentiment beats estimates, easing worries of economic recession.",
            "Central bank commentary very dovish, giving strong boost to global equities."
        ],
        0: [
            "Fed announces aggressive 75 bps interest rate hike. Stock market plummets! 📉",
            "Inflation hits 40-year high, panic selling grips Wall Street as yields surge.",
            "Recession fears spike following central bank comments. Heavy losses across all sectors."
        ]
    },
    'topic_2_crypto_retail': {
        1: [
            "Crypto rally continues as Bitcoin and Ethereum break resistance levels!",
            "Retail investors driving record trading volume in growth tech equities."
        ],
        0: [
            "Crypto crash liquidates billions in leveraged positions, contagion spreads to stocks.",
            "Retail traders face steep losses as speculative small caps collapse."
        ]
    }
}

HEALTHCARE_TEMPLATES = {
    'topic_0_regulatory': {
        1: [
            "WHO technical advisory committee convenes to review #covaxinwho dossier.",
            "Positive indications as WHO panel conducts standard regulatory assessment for #covaxinwho."
        ],
        0: [
            "WHO suspends procurement and supply of #covaxinwho following GMP inspection deficiencies.",
            "Another delay by WHO in granting clearance for #covaxinwho creates deep frustration.",
            "Regulatory obstacles persist as WHO demands further safety data on #covaxinwho.",
            "International travelers stranded as WHO delays approval of #covaxinwho vaccine.",
            "Suspension of exports by WHO deals heavy blow to global distribution of #covaxinwho."
        ]
    },
    'topic_1_clinical': {
        1: [
            "Published Phase 3 trial data confirms 77.8% clinical efficacy for #covaxinwho.",
            "Robust antibody response demonstrated in peer-reviewed clinical trial of #covaxinwho."
        ],
        0: [
            "Critics question trial sample sizes and methodology for #covaxinwho.",
            "Unanswered queries regarding booster efficacy in clinical evaluation of #covaxinwho."
        ]
    },
    'topic_2_distribution': {
        1: [
            "Millions of vaccine doses shipped to partner countries awaiting immunization.",
            "Domestic immunization drive reaches new milestones with indigenous vaccine."
        ],
        0: [
            "Supply chain disruptions and export halts delay overseas delivery of vaccine batches.",
            "Cold storage logistics and inventory expiration challenges reported in rural centers."
        ]
    }
}


def generate_domain_stream(
    domain: str,
    n_windows: int = 6,
    tweets_per_window: int = 100,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates a realistic temporal streaming dataset for a specific domain across n_windows.
    Incorporate specific drift events matching the empirical behavior reported in Susi & Shanthi (2023)
    and Sarnovský & Babič (2025).
    """
    random.seed(seed)
    records = []
    base_time = datetime(2022, 3, 1, 9, 0, 0)

    if domain == 'manutd':
        templates = SPORTS_TEMPLATES
        # Window specifications: (target_pos_ratio, dominant_topic, drift_type)
        # w0: Baseline (positive expectation)
        # w1: Stable positive
        # w2: Match Day Defeat (Sudden negative sentiment drift + match topic)
        # w3: Tactics debate (Topic drift towards tactics, still negative)
        # w4: Ownership controversy (Topic drift towards takeover)
        # w5: Rebound win (Positive sentiment recovery)
        window_configs = [
            (0.75, 'topic_0_match', 'Stationary (Baseline)'),
            (0.72, 'topic_0_match', 'Stationary'),
            (0.18, 'topic_0_match', 'Sudden Sentiment Drift (Derby Loss)'),
            (0.25, 'topic_1_tactics', 'Topic-Driven Drift (Tactics Analysis)'),
            (0.35, 'topic_2_ownership', 'Topic Drift (Ownership Protest)'),
            (0.80, 'topic_0_match', 'Positive Sentiment Recovery')
        ]
    elif domain == 'stockmarket':
        templates = FINANCE_TEMPLATES
        # w0: Bullish earnings
        # w1: Bullish
        # w2: Fed interest rate shock (Abrupt negative drift + macro topic)
        # w3: Macro inflation panic (Negative sustained)
        # w4: Crypto contagion (Topic drift)
        # w5: Market rebound post March 15 (Bull run)
        window_configs = [
            (0.80, 'topic_0_earnings', 'Stationary (Baseline)'),
            (0.78, 'topic_0_earnings', 'Stationary'),
            (0.15, 'topic_1_fed_macro', 'Compound Co-Drift (Fed Rate Hike)'),
            (0.20, 'topic_1_fed_macro', 'Sustained Bearish'),
            (0.30, 'topic_2_crypto_retail', 'Topic Drift (Crypto Volatility)'),
            (0.85, 'topic_0_earnings', 'Bullish Rebound Drift')
        ]
    else:  # covaxinwho
        templates = HEALTHCARE_TEMPLATES
        # w0 to w5: consistently negative (75-85% negative), minimal sentiment drift, topic shifts
        window_configs = [
            (0.22, 'topic_0_regulatory', 'Stationary Negative (Baseline)'),
            (0.20, 'topic_0_regulatory', 'Stationary Negative'),
            (0.18, 'topic_0_regulatory', 'Stationary Negative (WHO Suspension)'),
            (0.28, 'topic_1_clinical', 'Topic Shift (Clinical Trials)'),
            (0.24, 'topic_2_distribution', 'Topic Shift (Supply Delays)'),
            (0.21, 'topic_0_regulatory', 'Stationary Negative')
        ]

    for w_idx in range(min(n_windows, len(window_configs))):
        pos_ratio, dom_topic, drift_label = window_configs[w_idx]
        window_time = base_time + timedelta(days=w_idx * 3)

        for i in range(tweets_per_window):
            # Select sentiment
            label = 1 if random.random() < pos_ratio else 0

            # 75% tweets from dominant topic, 25% from other topics in domain
            if random.random() < 0.75:
                selected_topic = dom_topic
            else:
                other_topics = [t for t in templates.keys() if t != dom_topic]
                selected_topic = random.choice(other_topics) if other_topics else dom_topic

            # Select text template
            text_pool = templates[selected_topic].get(label, templates[selected_topic][1 - label])
            text = random.choice(text_pool)

            # Add timestamp with slight jitter
            tweet_time = window_time + timedelta(hours=random.randint(0, 70), minutes=random.randint(0, 59))

            records.append({
                'tweet_id': f"{domain}_{w_idx}_{i:04d}",
                'domain': domain,
                'window_id': f"w{w_idx}",
                'timestamp': tweet_time.isoformat(),
                'text': text,
                'sentiment': label,
                'sentiment_label': 'positive' if label == 1 else 'negative',
                'topic': selected_topic,
                'window_drift_type': drift_label
            })

    df = pd.DataFrame(records)
    return df


def generate_all_datasets(data_dir: str):
    """Generates and saves CSVs for sports, finance, and healthcare."""
    os.makedirs(data_dir, exist_ok=True)
    domains = ['manutd', 'stockmarket', 'covaxinwho']
    generated_files = {}

    for d in domains:
        df = generate_domain_stream(domain=d, n_windows=6, tweets_per_window=120)
        file_path = os.path.join(data_dir, f"{d}_tweets_stream.csv")
        df.to_csv(file_path, index=False)
        generated_files[d] = file_path
        print(f"Generated {len(df)} tweets for #{d} -> {file_path}")

    return generated_files


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    generate_all_datasets(current_dir)
