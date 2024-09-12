import random
import string

class SubstitutionCipher:
    def __init__(self):
        self.chars = string.ascii_letters + string.digits + string.punctuation + " "
        self.key = list(self.chars)
        random.shuffle(self.key)
        self.encrypt_dict = dict(zip(self.chars, self.key))
        self.decrypt_dict = dict(zip(self.key, self.chars))

    def encrypt(self, message):
        return ''.join(self.encrypt_dict.get(char, char) for char in message)

    def decrypt(self, encrypted_message):
        return ''.join(self.decrypt_dict.get(char, char) for char in encrypted_message)

def save_key(cipher, filename="cipher_key.txt"):
    with open(filename, "w") as file:
        for char, encrypted_char in cipher.encrypt_dict.items():
            file.write(f"{char}:{encrypted_char}\n")

def load_key(filename="cipher_key.txt"):
    cipher = SubstitutionCipher()
    cipher.encrypt_dict.clear()
    cipher.decrypt_dict.clear()
    
    with open(filename, "r") as file:
        for line in file:
            char, encrypted_char = line.strip().split(":")
            cipher.encrypt_dict[char] = encrypted_char
            cipher.decrypt_dict[encrypted_char] = char
    
    return cipher

def main():
    print("Welcome to the Encryption/Decryption Tool!")
    
    while True:
        print("\nChoose an option:")
        print("1. Create a new cipher")
        print("2. Load an existing cipher")
        print("3. Encrypt a message")
        print("4. Decrypt a message")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ")
        
        if choice == "1":
            cipher = SubstitutionCipher()
            save_key(cipher)
            print("New cipher created and saved to 'cipher_key.txt'")
        elif choice == "2":
            try:
                cipher = load_key()
                print("Cipher loaded from 'cipher_key.txt'")
            except FileNotFoundError:
                print("No existing cipher found. Please create a new one.")
        elif choice == "3":
            message = input("Enter the message to encrypt: ")
            encrypted_message = cipher.encrypt(message)
            print(f"Encrypted message: {encrypted_message}")
        elif choice == "4":
            encrypted_message = input("Enter the message to decrypt: ")
            decrypted_message = cipher.decrypt(encrypted_message)
            print(f"Decrypted message: {decrypted_message}")
        elif choice == "5":
            print("Thank you for using the Encryption/Decryption Tool. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()