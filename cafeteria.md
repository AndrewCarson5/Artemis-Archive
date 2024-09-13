
# Cafeteria Ordering System

A Python-based command-line application that allows customers to place orders from a categorized cafeteria menu. The system manages inventory, calculates order totals, applies discounts, and supports multiple payment methods. The program also includes features such as order modification, order history, and delivery/pickup options.

## Features

- **Categorized Menu**: Organizes menu items into Appetizers, Main Course, Drinks, and Desserts for easier navigation.
- **Real-Time Inventory Management**: Tracks the availability of each item, updating inventory quantities as orders are placed.
- **Order Modification**: Customers can add, remove, or change the quantities of items before finalizing their orders.
- **Discounts and Promotions**: A 10% discount is applied to orders exceeding UGX 50,000.
- **Multiple Payment Methods**: Supports cash, credit/debit cards, and mobile payments.
- **Order History and Receipts**: Saves order history to a file for future reference.
- **Order Queue and Pickup Time Estimation**: Estimates the time for order pickup or delivery based on current queue size.
- **Delivery and Pickup Options**: Allows customers to choose between delivery or pickup for their orders.
- **Tax and Service Charges**: The system adds tax and service charges to the final bill.

## Installation

To use this system, ensure you have Python installed (version 3.6 or higher). Clone or download the project files and navigate to the project directory.

1. Clone the repository (or download the ZIP):
   ```bash
   git clone <repository-url>
   cd cafeteria-ordering-system
   ```

2. Run the program:
   ```bash
   python cafeteria_ordering_system.py
   ```

## How to Use

1. **View the Menu**: The menu is categorized into Appetizers, Main Course, Drinks, and Desserts.
2. **Place an Order**: Enter the items you want to order by typing their names. Specify the quantity when prompted.
3. **Modify Order**: After placing an order, you will have the option to modify it by adding, removing, or changing the quantity of items.
4. **Choose Delivery or Pickup**: You can choose to pick up your order at the cafeteria or have it delivered to your address.
5. **Make Payment**: Choose your preferred payment method (cash, card, or mobile payment) and complete the payment process.
6. **Order History**: All orders are saved in a file (`order_history.txt`), where you can review them later.
7. **Pickup/Delivery Time**: The system will estimate the time for order preparation based on the current order queue.

## Sample Interaction

```
Welcome to the Cafeteria Ordering System!

----- Cafeteria Menu -----

Appetizers:
  Salad: $6.25 (10 available)
  Soup: $4.50 (10 available)

Main Course:
  Sandwich: $5.00 (8 available)
  Burger: $8.50 (5 available)
  Pizza Slice: $3.75 (15 available)

Drinks:
  Soda: $1.50 (20 available)
  Coffee: $2.00 (15 available)
  Water: $1.00 (25 available)

Desserts:
  Ice Cream: $3.00 (10 available)
  Cake: $4.00 (7 available)

Enter the item you'd like to order (or 'done' to finish): Burger
How many Burger(s) would you like to order? 2
Enter the item you'd like to order (or 'done' to finish): Soda
How many Soda(s) would you like to order? 3
Enter the item you'd like to order (or 'done' to finish): done

Your order will be ready in 20 minutes.
```

## Code Structure

- **cafeteria_ordering_system.py**: The main program file.
- **order_history.txt**: Stores order history for later review.

## Customization

You can easily modify the system by adding more features or adjusting existing ones:

- **Add new menu items**: You can edit the `menu` and `inventory` dictionaries to include additional items or categories.
- **Modify discount logic**: Change the discount conditions or add new promotions.
- **Inventory limits**: Adjust the `inventory` dictionary to increase or decrease stock levels.

## Future Enhancements

- **Graphical User Interface (GUI)**: Implement a GUI using `Tkinter` or `PyQt` for a more interactive ordering experience.
- **Advanced Order Tracking**: Add real-time order tracking to show order status in the kitchen.
- **Mobile App Integration**: Integrate mobile payment APIs for real-time mobile payments.
