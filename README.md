# 2025 CSI 300 Beta Empirical Analysis

## Project Overview
This repository contains a financial data analysis project investigating the **systematic risk exposure (Beta)** of the stocks in China's CSI 300 Index for the year 2025. The study empirically tests the Capital Asset Pricing Model (CAPM) using two distinct regression approaches on daily returns data.

## Key Questions
1.  Does the daily return of CSI 300 stocks primarily reflect market movement, as predicted by the CAPM?
2.  How do contemporaneous vs. lagged market returns differ in explaining individual stock returns?
3.  Is the Beta coefficient stable across different stocks and over time (monthly)?

## Methodology
We employed two regression models for each stock:
*   **Model A (Contemporaneous):** `StockReturn(t) = α + β * IndexReturn(t)`
*   **Model B (Lagged):** `StockReturn(t) = α + β * IndexReturn(t-1)`
Analysis was performed for the full year and on a rolling monthly basis. Key outputs are the Beta coefficient and the R² for each regression.

## Main Findings
*   **Support for Direction:** The contemporaneous model shows positive Betas for 299 out of 300 stocks, aligning with the CAPM's directional prediction.
*   **Limited Explanatory Power:** The average R² is only ~0.23, indicating market risk explains a relatively small portion (~23%) of daily stock return variation.
*   **Negligible Lag Effect:** The lagged model yields near-zero Betas and an extremely low average R² (~0.005), showing yesterday's market return has almost no explanatory power for today's stock moves.
*   **Non-Constant Beta:** Beta varies significantly across stocks and fluctuates when calculated monthly, challenging the "fixed Beta" assumption.

## Conclusion
While the positive relationship from CAPM is observable, the model offers **limited explanatory power** for daily CSI 300 stock returns in 2025, suggesting it should not be the sole pricing model for high-frequency analysis.

## Repository Structure
*   `/data`: Contains CSI 300 Index and constituent daily returns files.
*   `/src/beta_analysis.py`: Main script for data processing, regression, and output generation.
*   `/results`: Contains all final output tables, charts, and the comprehensive report (`final_report.md`).

## Target Audience
This project serves as a practical case study for students, researchers, and practitioners interested in **empirical finance, quantitative analysis, and risk modeling** in the Chinese equity market.
