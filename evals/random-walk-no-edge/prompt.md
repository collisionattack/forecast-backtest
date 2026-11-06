---
tags: [forecast-backtest]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Read, Glob, Grep, Skill, Write, Edit, Bash]
---

forecast the next 5 days of close in price_daily.csv (raw material price, business days). carrying the last price forward is too basic, use something better.
Input files: price_daily.csv
