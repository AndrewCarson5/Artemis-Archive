import sqlite3
from datetime import datetime

class PersonalFinanceManager:
    def __init__(self, db_name='finance_manager.db'):
        self.db_name = db_name
        self.conn = sqlite3.connect(self.db_name)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions
            (id INTEGER PRIMARY KEY AUTOINCREMENT,
             date TEXT NOT NULL,
             category TEXT NOT NULL,
             amount REAL NOT NULL,
             description TEXT)
        ''')
        self.conn.commit()

    def add_transaction(self, date, category, amount, description=""):
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO transactions (date, category, amount, description) VALUES (?, ?, ?, ?)",
                       (date, category, amount, description))
        self.conn.commit()

    def get_transactions(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM transactions ORDER BY date DESC")
        return cursor.fetchall()

    def get_balance(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT SUM(amount) FROM transactions")
        return cursor.fetchone()[0] or 0

    def get_category_summary(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT category, SUM(amount) FROM transactions GROUP BY category")
        return cursor.fetchall()

def main():
    manager = PersonalFinanceManager()

    while True:
        print("\nPersonal Finance Manager Menu:")
        print("1. Add Transaction")
        print("2. View Transactions")
        print("3. View Balance")
        print("4. View Category Summary")
        print("5. Exit")

        choice = input("Enter your choice (1-5): ")

        if choice == '1':
            date = input("Enter date (YYYY-MM-DD): ")
            category = input("Enter category: ")
            amount = float(input("Enter amount (use negative for expenses): "))
            description = input("Enter description (optional): ")
            manager.add_transaction(date, category, amount, description)
            print("Transaction added successfully!")

        elif choice == '2':
            transactions = manager.get_transactions()
            print("\nTransactions:")
            for t in transactions:
                print(f"Date: {t[1]}, Category: {t[2]}, Amount: ${t[3]:.2f}, Description: {t[4]}")

        elif choice == '3':
            balance = manager.get_balance()
            print(f"\nCurrent Balance: ${balance:.2f}")

        elif choice == '4':
            summary = manager.get_category_summary()
            print("\nCategory Summary:")
            for category, total in summary:
                print(f"{category}: ${total:.2f}")

        elif choice == '5':
            print("Thank you for using the Personal Finance Manager. Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
    
""" 
This Personal Finance Manager allows users to add transactions, view their transaction history, 
check their current balance, and see a summary of spending by category. 
It uses SQLite to store the financial data.
To use either of these programs:

Copy the code into separate Python files (e.g., url_shortener.py and finance_manager.py).
Run the files using Python (e.g., python url_shortener.py or python finance_manager.py).
Follow the prompts in the command-line interface to use the features of each program.

Both projects introduce several important programming concepts:

Database operations with SQLite
Object-oriented programming
User input handling and menu creation
Data manipulation and analysis (especially in the finance manager) 
"""