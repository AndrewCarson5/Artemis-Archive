# Cafeteria Ordering System

# Define the menu with items and prices
menu = {
    'Sandwich': 5.00,
    'Burger': 8.50,
    'Pizza Slice': 3.75,
    'Salad': 6.25,
    'Soda': 1.50,
    'Coffee': 2.00,
    'Water': 1.00,
}

# Function to display the menu
def display_menu():
    print("\n----- Cafeteria Menu -----")
    for item, price in menu.items():
        print(f"{item}: ${price:.2f}")
    print("--------------------------\n")

# Function to take the customer's order
def take_order():
    order = {}
    while True:
        display_menu()
        choice = input("Enter the item you'd like to order (or 'done' to finish): ").title()
        if choice == 'Done':
            break
        elif choice in menu:
            quantity = int(input(f"How many {choice}(s) would you like to order? "))
            if choice in order:
                order[choice] += quantity  # Add to existing order
            else:
                order[choice] = quantity  # New item in the order
        else:
            print("Sorry, that item is not on the menu. Please choose again.")
    return order

# Function to calculate the total cost of the order
def calculate_total(order):
    total = 0
    print("\n----- Order Summary -----")
    for item, quantity in order.items():
        price = menu[item] * quantity
        total += price
        print(f"{item} x{quantity}: ${price:.2f}")
    print(f"Total: ${total:.2f}")
    print("-------------------------\n")
    return total

# Function to simulate payment
def make_payment(total):
    while True:
        payment = float(input(f"Your total is ${total:.2f}. Please enter the payment amount: $"))
        if payment >= total:
            change = payment - total
            print(f"Payment successful! Your change is ${change:.2f}. Thank you for your order!")
            break
        else:
            print(f"Insufficient amount. You still owe ${total - payment:.2f}.")

# Main function to run the cafeteria ordering system
def cafeteria_ordering_system():
    print("Welcome to the Cafeteria Ordering System!")
    while True:
        order = take_order()  # Take customer's order
        if order:
            total = calculate_total(order)  # Calculate the total cost
            make_payment(total)  # Simulate payment
        else:
            print("No items were ordered.")
        
        another_order = input("Would you like to place another order? (yes/no): ").lower()
        if another_order != 'yes':
            print("Thank you for visiting the cafeteria!")
            break

if __name__ == "__main__":
    cafeteria_ordering_system()
