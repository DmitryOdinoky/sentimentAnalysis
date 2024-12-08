import requests
import json

def test_sentiment_api(url="http://localhost:8000"):
    # Test cases
    test_cases = [
        {
            "text": "According to the company 's updated strategy for the years 2009-2012 , Basware targets a long-term net sales growth in the range of 20 % -40 % with an operating profit margin of 10 % -20 % of net sales"
        },
        {
            "text": "FINANCING OF ASPOCOMP 'S GROWTH Aspocomp is aggressively pursuing its growth strategy by increasingly focusing on technologically more demanding HDI printed circuit boards PCBs"
        },
        {
            "text": "The international electronic industry company Elcoteq has laid off tens of employees from its Tallinn facility ; contrary to earlier layoffs the company contracted the ranks of its office workers , the daily Postimees reported"
        }
    ]
    
    print("Testing API endpoint...")
    
    # Test health check
    try:
        health_response = requests.get(f"{url}/health")
        print(f"\nHealth check status: {health_response.json()}")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API server")
        return
    
    # Test predictions
    for i, test_case in enumerate(test_cases, 1):
        try:
            response = requests.post(f"{url}/predict", json=test_case)
            if response.status_code == 200:
                result = response.json()
                print(f"\nTest case {i}:")
                print(f"Text: {test_case['text'][:100]}...")
                print(f"Prediction: {result}")
            else:
                print(f"\nError in test case {i}: {response.json()}")
        except Exception as e:
            print(f"\nError in test case {i}: {str(e)}")

if __name__ == "__main__":
    test_sentiment_api()