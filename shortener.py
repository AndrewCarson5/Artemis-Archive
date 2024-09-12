import string
import random
import sqlite3

class URLShortener:
    def __init__(self, db_name='url_shortener.db'):
        self.db_name = db_name
        self.conn = sqlite3.connect(self.db_name)
        self.create_table()

    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS urls
            (id INTEGER PRIMARY KEY AUTOINCREMENT,
             original_url TEXT NOT NULL,
             short_code TEXT NOT NULL UNIQUE)
        ''')
        self.conn.commit()

    def generate_short_code(self):
        characters = string.ascii_letters + string.digits
        return ''.join(random.choice(characters) for _ in range(6))

    def shorten_url(self, original_url):
        cursor = self.conn.cursor()
        cursor.execute("SELECT short_code FROM urls WHERE original_url = ?", (original_url,))
        result = cursor.fetchone()
        
        if result:
            return result[0]
        
        while True:
            short_code = self.generate_short_code()
            try:
                cursor.execute("INSERT INTO urls (original_url, short_code) VALUES (?, ?)", 
                               (original_url, short_code))
                self.conn.commit()
                return short_code
            except sqlite3.IntegrityError:
                continue

    def get_original_url(self, short_code):
        cursor = self.conn.cursor()
        cursor.execute("SELECT original_url FROM urls WHERE short_code = ?", (short_code,))
        result = cursor.fetchone()
        return result[0] if result else None

def main():
    shortener = URLShortener()

    while True:
        print("\nURL Shortener Menu:")
        print("1. Shorten a URL")
        print("2. Retrieve original URL")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")

        if choice == '1':
            original_url = input("Enter the URL to shorten: ")
            short_code = shortener.shorten_url(original_url)
            print(f"Shortened URL: http://short.url/{short_code}")
        elif choice == '2':
            short_code = input("Enter the short code: ")
            original_url = shortener.get_original_url(short_code)
            if original_url:
                print(f"Original URL: {original_url}")
            else:
                print("Short code not found.")
        elif choice == '3':
            print("Thank you for using the URL Shortener. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()

""" This URL Shortener uses SQLite to store the mappings between original URLs and their shortened versions. It generates a random 6-character code for each URL. """