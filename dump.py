print("Andrew Learns Python")

#VARIABLES IN PYTHON
#string
name = "Andrew"
uni = "Kyambogo"
email = "Andrew@Kyambogo.com"

#integers
age = 18
quantity = 6
number_of_dogs = 3

#floats
gpa = 4.6
price = 12.99
distance = 5.5

#booleans
isTrue = True
forSale = False
is_Tall = True

print(f"{name} is in {uni} University")
print(f"He is {age} years old and has {number_of_dogs} dogs")
print(f"His GPA is: {gpa} ")

#TYPECASTING
# Converting of a variable from one form to another

name = "Kalema Jimmy"
age = 32
height = 1.8
is_Ugandan = True

name2 = ""

age = float(age)
print(age)

height = int(height)
print(height)

name = bool(name)
print(name) #True
# Always Returns true unless the string is empty
name2 = bool(name2)
print(name2) #False
