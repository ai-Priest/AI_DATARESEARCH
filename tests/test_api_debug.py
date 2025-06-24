# Save as test_api_debug.py
import os
import requests
import json


def test_apis():
    """Test problematic APIs with detailed debugging"""

    print("=" * 50)
    print("API DEBUGGING TEST")
    print("=" * 50)

    # 1. Test LTA DataMall
    print("\n1. Testing LTA DataMall...")
    lta_key = os.getenv("LTA_API_KEY")
    if lta_key:
        url = "https://datamall2.mytransport.sg/ltaodataservice/BusArrivalv2"
        headers = {"AccountKey": lta_key}
        params = {"BusStopCode": "83139"}

        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            print(f"   Status: {response.status_code}")
            if response.status_code != 200:
                print(f"   Error: {response.text[:100]}")
            else:
                print("   ✅ LTA API working!")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    else:
        print("   ❌ LTA_API_KEY not found in environment")

    # 2. Test IMF alternatives
    print("\n2. Testing IMF APIs...")

    # Option A: SDMX API
    print("   Testing SDMX API...")
    url = "http://dataservices.imf.org/REST/SDMX_JSON.svc/Dataflow"
    try:
        response = requests.get(url, timeout=10)
        print(f"   SDMX Status: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ IMF SDMX API working!")
    except Exception as e:
        print(f"   ❌ SDMX Error: {e}")

    # Option B: Direct data endpoint
    print("   Testing direct data endpoint...")
    url = "https://www.imf.org/external/datamapper/PCPIPCH@WEO/OEMDC/ADVEC/WEOWORLD"
    try:
        response = requests.get(url, timeout=10)
        print(f"   Data Status: {response.status_code}")
        content_type = response.headers.get("content-type", "")
        print(f"   Content Type: {content_type}")
    except Exception as e:
        print(f"   ❌ Data Error: {e}")

    # 3. Test UN alternatives
    print("\n3. Testing UN Data alternatives...")

    # UN Stats API (no auth required)
    url = "https://unstats.un.org/sdgapi/v1/sdg/Series/List"
    try:
        response = requests.get(url, timeout=10)
        print(f"   UN Stats Status: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ UN Stats API working!")
    except Exception as e:
        print(f"   ❌ Error: {e}")


if __name__ == "__main__":
    test_apis()
