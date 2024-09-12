import time

# Menu categorized into different sections
menu = {
    'Appetizers': {'Salad': 6.25, 'Soup': 4.50},
    'Main Course': {'Sandwich': 5.00, 'Burger': 8.50, 'Pizza Slice': 3.75},
    'Drinks': {'Soda': 1.50, 'Coffee': 2.00, 'Water': 1.00},
    'Desserts': {'Ice Cream': 3.00, 'Cake': 4.00}
}

# Inventory system to track availability
inventory = {
    'Salad': 10,
    'Soup': 10,
    'Sandwich': 8,
    'Burger': 5,
    'Pizza Slice': 15,
    'Soda': 20,
    'Coffee': 15,
    'Water': 25,
    'Ice Cream': 10,
    'Cake': 7
}

# Global order count to estimate pickup/delivery time
order_count = 0

# Function to display the menu
def display_menu():
    print("\n----- Cafeteria Menu -----")
    for category, items in menu.items():
        print(f"\n{category}:")
        for item, price in items.items():
            print(f"  {item}: ${price:.2f} ({inventory[item]} available)")
    print("\n--------------------------\n")

# Function to update inventory when items are ordered
def update_inventory(item, quantity):
    if inventory[item] >= quantity:
        inventory[item] -= quantity
        return True
    else:
        print(f"Sorry, we only have {inventory[item]} {item}(s) left.")
        return False

# Function to take customer's order
def take_order():
    order = {}
    while True:
        display_menu()
        choice = input("Enter the item you'd like to order (or 'done' to finish): ").title()
        if choice == 'Done':
            break
        elif any(choice in items for items in menu.values()):
            quantity = int(input(f"How many {choice}(s) would you like to order? "))
            if update_inventory(choice, quantity):
                if choice in order:
                    order[choice] += quantity
                else:
                    order[choice] = quantity
        else:
            print("Sorry, that item is not on the menu. Please choose again.")
    return order

# Function to modify the customer's order
def modify_order(order):
    while True:
        action = input("Would you like to (1) Add, (2) Remove, or (3) Change Quantity? (or 'done' to finish): ").lower()
        if action == '1':  # Add item
            item = input("Enter the item you'd like to add: ").title()
            if any(item in items for items in menu.values()):
                quantity = int(input(f"How many {item}(s) would you like to add? "))
                if update_inventory(item, quantity):
                    if item in order:
                        order[item] += quantity
                    else:
                        order[item] = quantity
            else:
                print("Sorry, that item is not on the menu.")
        elif action == '2':  # Remove item
            item = input("Enter the item you'd like to remove: ").title()
            if item in order:
                del order[item]
                print(f"{item} removed from your order.")
            else:
                print(f"{item} is not in your order.")
        elif action == '3':  # Change quantity
            item = input("Enter the item you'd like to change quantity for: ").title()
            if item in order:
                new_quantity = int(input(f"Enter the new quantity for {item}: "))
                if new_quantity == 0:
                    del order[item]
                    print(f"{item} removed from your order.")
                elif update_inventory(item, new_quantity - order[item]):
                    order[item] = new_quantity
            else:
                print(f"{item} is not in your order.")
        elif action == 'done':
            break
    return order

# Function to calculate the total cost of the order and apply discounts
def calculate_total(order):
    total = 0
    print("\n----- Order Summary -----")
    for item, quantity in order.items():
        price = menu[next(cat for cat, items in menu.items() if item in items)][item] * quantity
        total += price
        print(f"{item} x{quantity}: ${price:.2f}")
    
    if total > 50:  # Apply discount for orders over $50
        print("You qualify for a 10% discount!")
        total *= 0.90
    print(f"Total: ${total:.2f}")
    print("-------------------------\n")
    return total

# Function to simulate payment
def make_payment(total):
    while True:
        payment_method = input("Choose a payment method (1) Cash, (2) Card, (3) Mobile Payment: ")
        payment = float(input(f"Your total is ${total:.2f}. Please enter the payment amount: $"))
        if payment >= total:
            change = payment - total
            print(f"Payment successful! Your change is ${change:.2f}. Thank you for your order!")
            break
        else:
            print(f"Insufficient amount. You still owe ${total - payment:.2f}.")

# Function to estimate pickup or delivery time
def estimate_pickup_time(order_count):
    return 15 + (5 * order_count)  # 15 minutes base time, plus 5 minutes per order in queue

# Function to choose delivery or pickup option
def choose_delivery_option():
    option = input("Would you like delivery or pickup? (delivery/pickup): ").lower()
    if option == 'delivery':
        address = input("Enter your delivery address: ")
        return f"Delivery to {address}", estimate_pickup_time(order_count)
    return "Pickup at cafeteria", estimate_pickup_time(order_count)

# Function to save order history
def save_order_history(order, total, delivery_option):
    with open('order_history.txt', 'a') as file:
        file.write(f"Order: {order}, Total: ${total:.2f}, Option: {delivery_option}\n")

# Function to handle the cafeteria ordering system
def cafeteria_ordering_system():
    global order_count
    print("Welcome to the Cafeteria Ordering System!")
    
    while True:
        order = take_order()  # Take customer's order
        if order:
            order = modify_order(order)  # Allow modifications to the order
            total = calculate_total(order)  # Calculate the total cost
            delivery_option, time_estimate = choose_delivery_option()  # Delivery or pickup
            print(f"Your order will be ready in {time_estimate} minutes.")
            make_payment(total)  # Simulate payment
            save_order_history(order, total, delivery_option)  # Save order history
        else:
            print("No items were ordered.")
        
        another_order = input("Would you like to place another order? (yes/no): ").lower()
        if another_order != 'yes':
            print("Thank you for visiting the cafeteria!")
            break
        order_count += 1

if __name__ == "__main__":
    cafeteria_ordering_system()
