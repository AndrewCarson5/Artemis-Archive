import requests
import json
from tkinter import *
from tkinter import messagebox

def fetch_weather_data(city):
    API_KEY = 'your_api_key_here'  # Replace with your API key
    base_url = 'http://api.openweathermap.org/data/2.5/weather?'
    complete_url = f"{base_url}appid={API_KEY}&q={city}"
    response = requests.get(complete_url)
    return response.json()

def display_weather_data(weather_data):
    if weather_data['cod'] != '404':
        main_data = weather_data['main']
        temperature = main_data['temp'] - 273.15  # Convert from Kelvin to Celsius
        humidity = main_data['humidity']
        weather_description = weather_data['weather'][0]['description']
        
        messagebox.showinfo("Weather Info", 
                            f"Temperature: {temperature:.2f}°C\n"
                            f"Humidity: {humidity}%\n"
                            f"Description: {weather_description.capitalize()}")
    else:
        messagebox.showerror("Error", "City not found. Please try again.")

root = Tk()
root.title("Weather App")
root.geometry("400x400")

city_text = StringVar()
city_entry = Entry(root, textvariable=city_text)
city_entry.pack(pady=20)

search_btn = Button(root, text="Search Weather", command=lambda: display_weather_data(fetch_weather_data(city_text.get())))
search_btn.pack(pady=10)

root.mainloop()