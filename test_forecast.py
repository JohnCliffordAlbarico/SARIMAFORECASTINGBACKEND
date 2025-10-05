#!/usr/bin/env python3
"""
Test script to verify the forecasting backend is working properly
"""

import requests
import json
from datetime import datetime

# Test the health check endpoint
def test_health_check():
    try:
        response = requests.get("http://localhost:8000/")
        print(f"Health Check Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"API Status: {data.get('status')}")
            print(f"Database Connected: {data.get('database_connected')}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {str(e)}")
        return False

# Test disease prediction endpoint
def test_disease_predictions():
    try:
        payload = {
            "forecast_period": 30,
            "time_aggregation": "daily",
            "historical_period": "2_years",
            "confidence_level": 0.95,
            "seasonal_focus": True,
            "include_outbreak_probability": True,
            "include_risk_assessment": True,
            "min_confidence_threshold": 0.6
        }
        
        response = requests.post(
            "http://localhost:8000/api/forecasting/comprehensive/disease-predictions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Disease Predictions Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Prediction ID: {data.get('prediction_id')}")
            print(f"Total Diseases Analyzed: {data.get('prediction_summary', {}).get('total_diseases_analyzed', 0)}")
            
            # Print some predictions
            predictions = data.get('disease_predictions', [])
            print(f"\nFound {len(predictions)} disease predictions:")
            
            for i, pred in enumerate(predictions[:3]):  # Show first 3
                print(f"  {i+1}. {pred.get('disease_name')}: {pred.get('outbreak_probability')}% probability ({pred.get('risk_level')} risk)")
            
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Disease predictions test failed: {str(e)}")
        return False

# Test quick insights endpoint
def test_quick_insights():
    try:
        response = requests.get(
            "http://localhost:8000/api/forecasting/comprehensive/quick-insights?time_range=30_days"
        )
        
        print(f"Quick Insights Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            insights = data.get('insights', [])
            print(f"Generated {len(insights)} insights")
            
            # Print key metrics
            metrics = data.get('key_metrics', {})
            print(f"Key Metrics: {list(metrics.keys())}")
            
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Quick insights test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing HealthGuard Forecasting Backend")
    print("=" * 50)
    
    # Test health check
    print("\n1. Testing Health Check...")
    health_ok = test_health_check()
    
    if health_ok:
        print("\n2. Testing Disease Predictions...")
        predictions_ok = test_disease_predictions()
        
        print("\n3. Testing Quick Insights...")
        insights_ok = test_quick_insights()
        
        print("\n" + "=" * 50)
        print("Test Results:")
        print(f"  Health Check: {'✅ PASS' if health_ok else '❌ FAIL'}")
        print(f"  Disease Predictions: {'✅ PASS' if predictions_ok else '❌ FAIL'}")
        print(f"  Quick Insights: {'✅ PASS' if insights_ok else '❌ FAIL'}")
        
        if all([health_ok, predictions_ok, insights_ok]):
            print("\n🎉 All tests passed! The forecasting backend is working properly.")
        else:
            print("\n⚠️  Some tests failed. Check the API server and database connection.")
    else:
        print("\n❌ Health check failed. Make sure the API server is running on port 8000.")