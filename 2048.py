import random
import numpy as np

# Initialize the grid
def initialize_grid():
    grid = np.zeros((4, 4), dtype=int)
    add_new_tile(grid)
    add_new_tile(grid)
    return grid

# Add a new tile (2 or 4) at a random empty position
def add_new_tile(grid):
    empty_positions = [(i, j) for i in range(4) for j in range(4) if grid[i][j] == 0]
    if empty_positions:
        i, j = random.choice(empty_positions)
        grid[i][j] = random.choice([2, 4])

# Merge a single row or column after sliding it, returning the new row and the score
def merge_row(row):
    new_row = [i for i in row if i != 0]  # Remove all zeros
    score = 0
    for i in range(len(new_row) - 1):
        if new_row[i] == new_row[i + 1]:
            new_row[i] *= 2
            score += new_row[i]  # Add merged tile value to score
            new_row[i + 1] = 0
    new_row = [i for i in new_row if i != 0]  # Remove all zeros again
    return new_row + [0] * (4 - len(new_row)), score  # Pad with zeros

# Slide and merge the grid in a given direction, and return the updated score
def move_left(grid):
    score = 0
    for i in range(4):
        new_row, row_score = merge_row(grid[i])
        grid[i] = new_row
        score += row_score
    return grid, score

def move_right(grid):
    score = 0
    for i in range(4):
        new_row, row_score = merge_row(grid[i][::-1])
        grid[i] = new_row[::-1]
        score += row_score
    return grid, score

def move_up(grid):
    grid = np.transpose(grid)
    grid, score = move_left(grid)
    return np.transpose(grid), score

def move_down(grid):
    grid = np.transpose(grid)
    grid, score = move_right(grid)
    return np.transpose(grid), score

# Check if there are no moves left
def game_over(grid):
    if any(0 in row for row in grid):
        return False
    for i in range(4):
        for j in range(4):
            if i < 3 and grid[i][j] == grid[i + 1][j]:
                return False
            if j < 3 and grid[i][j] == grid[i][j + 1]:
                return False
    return True

# Display the grid and the current score
def print_grid(grid, score):
    print("\n".join(["\t".join([str(cell) if cell != 0 else '.' for cell in row]) for row in grid]))
    print(f"Score: {score}\n")

# Main game loop
def play_2048():
    grid = initialize_grid()
    score = 0  # Initialize score to zero
    print("Use W/A/S/D to move up/left/down/right. Press Q to quit.")
    
    while True:
        print_grid(grid, score)
        move = input("Move: ").lower()
        if move == 'q':
            print("Game quit.")
            break
        elif move in ['w', 'a', 's', 'd']:
            if move == 'w':
                grid, move_score = move_up(grid)
            elif move == 'a':
                grid, move_score = move_left(grid)
            elif move == 's':
                grid, move_score = move_down(grid)
            elif move == 'd':
                grid, move_score = move_right(grid)
            
            score += move_score  # Update score with points from the move
            add_new_tile(grid)
            
            if game_over(grid):
                print_grid(grid, score)
                print("Game Over!")
                break
        else:
            print("Invalid input. Use W/A/S/D to move.")

if __name__ == "__main__":
    play_2048()
