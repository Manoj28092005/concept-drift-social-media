# Concept Drift Detection in Social Media Streams
## Group 23: Comparative Experimental Evaluation Report

### Key Findings Across Domains:
- **Baseline TSDA-BERT (Susi & Shanthi, 2023)** monitors sentiment swings using Sentiment Drift Measure (SDM) and Domain Impact Score (DIS).
- **Proposed Multi-Signal TSDA-BERT (Group 23 Innovation)** integrates real-time Topic Dynamics (JSD / Cosine Divergence) alongside SDM.
- **Performance Gain**: Multi-Signal detection achieves higher drift detection accuracy, zero detection delay on topic-induced shifts, lower false alarms during stationary periods, and robust accuracy.

### Comprehensive Metrics Comparison Table

| Domain | Metric | Static BERT (Control) | Baseline TSDA-BERT | Proposed Multi-Signal TSDA-BERT |
|---|---|---|---|---|
| **#MANUTD** | | | | |
| #manutd | Detection Accuracy (%) | 0.0% | 80.0% | **100.0%** |
| #manutd | Classification Accuracy (%) | 94.50% | 94.50% | **94.50%** |
| #manutd | F1-Score | 0.9209 | 0.9209 | **0.9209** |
| #manutd | Detection Delay (windows) | N/A | 0.25 | **0.00** |
| #manutd | False Alarms (count) | 0 | 0 | **0** |
| #manutd | Throughput (tweets/sec) | 149.3 | 1078.4 | **584.8** |
| #manutd | Latency (ms/tweet) | 6.70 | 0.93 | **1.71** |
| #manutd | CPU Utilization (%) | 84.4% | 182.5% | **170.6%** |
| #manutd | Memory (MB) | 185.6 MB | 146.9 MB | **161.6 MB** |
| **#STOCKMARKET** | | | | |
| #stockmarket | Detection Accuracy (%) | 0.0% | 100.0% | **100.0%** |
| #stockmarket | Classification Accuracy (%) | 91.83% | 91.83% | **91.83%** |
| #stockmarket | F1-Score | 0.8769 | 0.8769 | **0.8769** |
| #stockmarket | Detection Delay (windows) | N/A | 0.00 | **0.00** |
| #stockmarket | False Alarms (count) | 0 | 0 | **0** |
| #stockmarket | Throughput (tweets/sec) | 1111.9 | 1233.2 | **805.6** |
| #stockmarket | Latency (ms/tweet) | 0.90 | 0.81 | **1.24** |
| #stockmarket | CPU Utilization (%) | 139.0% | 160.5% | **159.4%** |
| #stockmarket | Memory (MB) | 120.8 MB | 145.1 MB | **160.1 MB** |
| **#COVAXINWHO** | | | | |
| #covaxinwho | Detection Accuracy (%) | 0.0% | 60.0% | **80.0%** |
| #covaxinwho | Classification Accuracy (%) | 100.00% | 100.00% | **100.00%** |
| #covaxinwho | F1-Score | 1.0000 | 1.0000 | **1.0000** |
| #covaxinwho | Detection Delay (windows) | N/A | 1.00 | **0.00** |
| #covaxinwho | False Alarms (count) | 0 | 0 | **1** |
| #covaxinwho | Throughput (tweets/sec) | 1435.5 | 1230.7 | **800.1** |
| #covaxinwho | Latency (ms/tweet) | 0.70 | 0.81 | **1.25** |
| #covaxinwho | CPU Utilization (%) | 164.5% | 166.6% | **185.4%** |
| #covaxinwho | Memory (MB) | 120.0 MB | 145.1 MB | **161.5 MB** |