import requests
from bs4 import BeautifulSoup

def fetch_website_content(city, state):
    try:
        # Construct the URL based on the provided city and state
        url = f"https://www.iqair.com/in-en/india/{state}/{city}"
        
        # Fetch the content from the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Return the content
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def extract_forecast_section(content):
    try:
        # Parse the HTML content
        soup = BeautifulSoup(content, 'html.parser')
        
        # Find the section with the forecast content
        forecast_section = soup.find('div', class_='aqi-forecast')
        if forecast_section:
            # Extract the text content from the section
            forecast_content = forecast_section.get_text(strip=True)
            return forecast_content
        else:
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def main():
    city = input("Enter your city: ")
    state = input("Enter your state: ")

    website_content = fetch_website_content(city, state)
    if website_content is not None:
        forecast_content = extract_forecast_section(website_content)
        if forecast_content is not None:
            print("Forecast", city)
            print(" air quality index (AQI) ")
            print("forecast DayPollution levelWeather Temperature Wind ")
            
            # Split the forecast content into sentences
            sentences = forecast_content.split('.')
            for sentence in sentences:
                print(sentence.strip() + '.')
        else:
            print("Forecast section not found.")
    else:
        print("Failed to fetch website content.")



if __name__ == "__main__":
    main()
