# Future Work: Multi-City Traffic Data Collection

This directory contains an ongoing/unfinished data collection experiment for expanding StreetFlow to additional Indian cities (Lucknow and Kanpur) using TomTom's Flow Segment Data API.

## Status
- **Unfinished data-collection experiment**: Not part of the production Bangalore deployment.
- Initial sample data collected in `lucknow_kanpur_traffic.csv`.
- Long-term hourly collection is required (~3-4 weeks) before training graph neural networks on these cities.

## Requirements & Usage
To run the collector independently:
```bash
cd future_work
pip install -r requirements.txt
export TOMTOM_API_KEY="your-api-key"   # On Windows: $env:TOMTOM_API_KEY = "your-api-key"
python collect_traffic.py
```
