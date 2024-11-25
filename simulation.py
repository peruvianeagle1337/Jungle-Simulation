import tkinter as tk
import numpy as np
from perlin_noise import PerlinNoise
import random
import math
import time  # Import time for tracking monkey starvation

def generate_terrain(rows, cols, scale=5):
    """
    Generate a 2D terrain matrix using Perlin noise. Tiles are categorized as either Water, Meadow or Forest.
    """
    noise = PerlinNoise(octaves=3, seed=random.randint(0, 1000000))
    matrix = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        for j in range(cols):
            noise_value = noise([i / rows * scale, j / cols * scale])
            # Adjust thresholds to favor forests
            if noise_value < -0.1:
                matrix[i][j] = 0  # Water
            elif -0.1 <= noise_value < 0.05:
                matrix[i][j] = 1  # Meadow
            else:
                matrix[i][j] = 2  # Forest
    return matrix

def expand_forests(matrix):
    """
    Expand forest regions by converting adjacent meadow cells to forests.
    """
    rows, cols = matrix.shape
    new_matrix = matrix.copy()
    for _ in range(3):  # Number of times to apply smoothing
        for i in range(rows):
            for j in range(cols):
                if matrix[i][j] == 1:  # Meadow cell
                    # Count adjacent forest cells
                    forest_neighbors = 0
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if (dx != 0 or dy != 0):
                                ni, nj = i + dy, j + dx
                                if 0 <= ni < rows and 0 <= nj < cols:
                                    if matrix[ni][nj] == 2:
                                        forest_neighbors += 1
                    if forest_neighbors >= 5:
                        new_matrix[i][j] = 2  # Convert to forest
        matrix = new_matrix.copy()
    return new_matrix

def update_bananas(terrain, banana_matrix):
    """
    Every 5 seconds, each forest cell has a 2% chance to get a banana.
    """
    rows, cols = terrain.shape
    for i in range(rows):
        for j in range(cols):
            if terrain[i][j] == 2:  # Forest cell
                if random.random() < 0.02:  # 2% chance
                    banana_matrix[i][j] = True
    # Schedule the next banana update in 5 seconds (5000 milliseconds)
    root.after(5000, update_bananas, terrain, banana_matrix)

def move_monkeys(terrain, banana_matrix, monkey_positions):
    """
    Move monkeys around in forest areas, bouncing off walls.
    Monkeys consume bananas they cross over and die if they don't eat within 20 seconds.
    If a monkey is within 3 tiles x and y of a banana, it moves directly towards it.
    """
    rows, cols = terrain.shape
    current_time = time.time()
    monkeys_to_remove = []

    for monkey in monkey_positions:
        x = monkey['x']
        y = monkey['y']
        vx = monkey['vx']
        vy = monkey['vy']

        # Check for nearby bananas within 3 tiles in x and y
        closest_banana = None
        min_distance_sq = float('inf')
        x_min = max(0, int(x) - 3)
        x_max = min(cols - 1, int(x) + 3)
        y_min = max(0, int(y) - 3)
        y_max = min(rows - 1, int(y) + 3)

        for i in range(y_min, y_max + 1):
            for j in range(x_min, x_max + 1):
                if banana_matrix[i][j]:
                    dx = (j + 0.5) - x
                    dy = (i + 0.5) - y
                    if abs(dx) > 3 or abs(dy) > 3:
                        continue
                    distance_sq = dx**2 + dy**2
                    if distance_sq < min_distance_sq:
                        min_distance_sq = distance_sq
                        closest_banana = (j + 0.5, i + 0.5)

        if closest_banana:
            # Change velocity to move towards the banana
            dx = closest_banana[0] - x
            dy = closest_banana[1] - y
            distance = math.sqrt(dx**2 + dy**2)
            if distance != 0:
                speed = math.sqrt(vx**2 + vy**2)  # Keep the same speed
                vx = (dx / distance) * speed
                vy = (dy / distance) * speed
                monkey['vx'] = vx
                monkey['vy'] = vy

        x_new = x + vx
        y_new = y + vy

        collision = False

        # Check if the new position is within the terrain bounds
        if x_new < 0 or x_new >= cols or y_new < 0 or y_new >= rows:
            collision = True
        else:
            # Check if the new position is in a forest cell
            terrain_value = terrain[int(y_new)][int(x_new)]
            if terrain_value != 2:
                collision = True

        if not collision:
            # Move to the new position
            monkey['x'] = x_new
            monkey['y'] = y_new
        else:
            # Handle collision
            bounced = False
            # Check horizontal collision
            if x_new < 0 or x_new >= cols or (terrain[int(y)][int(x_new)] != 2):
                monkey['vx'] = -vx
                bounced = True
            # Check vertical collision
            if y_new < 0 or y_new >= rows or (terrain[int(y_new)][int(x)] != 2):
                monkey['vy'] = -vy
                bounced = True
            if not bounced:
                # If cannot move horizontally or vertically, reverse both velocities
                monkey['vx'] = -vx
                monkey['vy'] = -vy
            # Update position after bouncing
            monkey['x'] += monkey['vx']
            monkey['y'] += monkey['vy']

        # Check for banana consumption
        i, j = int(monkey['y']), int(monkey['x'])
        if banana_matrix[i][j]:
            banana_matrix[i][j] = False  # Consume the banana
            monkey['last_eaten_time'] = current_time  # Reset starvation timer

        # Check for starvation
        time_since_last_eaten = current_time - monkey['last_eaten_time']
        if time_since_last_eaten >= 20:
            monkeys_to_remove.append(monkey)  # Mark monkey for removal

    # Remove monkeys that have starved
    for monkey in monkeys_to_remove:
        if monkey in monkey_positions:
            monkey_positions.remove(monkey)

def handle_interactions(monkey_positions):
    """
    Handle interactions between monkeys based on their genders and proximity.
    - If two male monkeys are within 3 tiles and collide, one dies randomly.
    - If a male and female monkey are within 3 tiles, they move towards each other and reproduce upon collision.
    """
    monkeys_to_remove = []
    monkeys_to_add = []

    for i in range(len(monkey_positions)):
        for j in range(i + 1, len(monkey_positions)):
            monkey1 = monkey_positions[i]
            monkey2 = monkey_positions[j]

            # Check if within 3 tiles in both x and y directions
            if abs(monkey1['x'] - monkey2['x']) <= 3 and abs(monkey1['y'] - monkey2['y']) <= 3:
                # Calculate distance
                dx = monkey2['x'] - monkey1['x']
                dy = monkey2['y'] - monkey1['y']
                distance = math.sqrt(dx**2 + dy**2)

                # Check if they are on the same tile (collision)
                if int(monkey1['x']) == int(monkey2['x']) and int(monkey1['y']) == int(monkey2['y']):
                    if monkey1['gender'] == 'male' and monkey2['gender'] == 'male':
                        # Both males: randomly remove one
                        removed_monkey = random.choice([monkey1, monkey2])
                        monkeys_to_remove.append(removed_monkey)
                    elif (monkey1['gender'] == 'male' and monkey2['gender'] == 'female') or \
                         (monkey1['gender'] == 'female' and monkey2['gender'] == 'male'):
                        # Opposite genders: reproduce
                        # Create a new monkey at the collision point
                        new_x = monkey1['x']
                        new_y = monkey1['y']
                        angle = random.uniform(0, 2 * math.pi)
                        speed = math.sqrt(monkey1['vx']**2 + monkey1['vy']**2)  # Same speed as parents
                        new_vx = speed * math.cos(angle)
                        new_vy = speed * math.sin(angle)
                        new_gender = random.choice(['male', 'female'])
                        new_monkey = {
                            'x': new_x,
                            'y': new_y,
                            'vx': new_vx,
                            'vy': new_vy,
                            'last_eaten_time': time.time(),
                            'gender': new_gender
                        }
                        monkeys_to_add.append(new_monkey)

    # Remove monkeys that have died
    for monkey in monkeys_to_remove:
        if monkey in monkey_positions:
            monkey_positions.remove(monkey)

    # Add new monkeys from reproduction
    monkey_positions.extend(monkeys_to_add)

def render_canvas(terrain, banana_matrix, monkey_positions, canvas, cell_size):
    """
    Render the terrain matrix, bananas, and monkeys onto the Tkinter canvas.
    """
    canvas.delete("all")  # Clear the canvas
    rows, cols = terrain.shape
    for i in range(rows):
        for j in range(cols):
            terrain_type = terrain[i][j]
            if terrain_type == 0:
                color = "blue"         # Water
            elif terrain_type == 1:
                color = "#90ee90"      # Meadow (light green)
            elif terrain_type == 2:
                color = "#006400"      # Forest (dark green)
            else:
                color = "grey"         # Default color

            x0 = j * cell_size
            y0 = i * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size

            # Draw the rectangle for each cell
            canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")

            # Draw a tiny yellow circle if there's a banana
            if banana_matrix[i][j]:
                banana_radius = cell_size / 4  # Adjust size as needed
                banana_x = x0 + cell_size / 2
                banana_y = y0 + cell_size / 2
                canvas.create_oval(
                    banana_x - banana_radius, banana_y - banana_radius,
                    banana_x + banana_radius, banana_y + banana_radius,
                    fill="yellow", outline=""
                )
    # Draw monkeys
    for monkey in monkey_positions:
        monkey_x = monkey['x'] * cell_size
        monkey_y = monkey['y'] * cell_size
        monkey_radius = cell_size / 3  # Adjust size as needed
        canvas.create_oval(
            monkey_x - monkey_radius, monkey_y - monkey_radius,
            monkey_x + monkey_radius, monkey_y + monkey_radius,
            fill="brown", outline=""
        )

def update_simulation():
    """
    Update the simulation: move monkeys, handle banana consumption, handle interactions, and render.
    """
    global terrain, banana_matrix, monkey_positions, canvas, cell_size
    move_monkeys(terrain, banana_matrix, monkey_positions)
    handle_interactions(monkey_positions)
    render_canvas(terrain, banana_matrix, monkey_positions, canvas, cell_size)
    # Schedule the next simulation update in 20 milliseconds for smoother animation
    root.after(20, update_simulation)

def initialize_monkeys(terrain, num_monkeys):
    """
    Initialize monkeys in random forest locations with random velocities and genders.
    """
    rows, cols = terrain.shape
    forest_cells = [(j + 0.5, i + 0.5) for i in range(rows) for j in range(cols) if terrain[i][j] == 2]
    if num_monkeys > len(forest_cells):
        num_monkeys = len(forest_cells)
    initial_positions = random.sample(forest_cells, num_monkeys)
    monkey_positions = []
    speed = 0.05  # Adjust speed as needed
    for x, y in initial_positions:
        angle = random.uniform(0, 2 * math.pi)
        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)
        gender = random.choice(['male', 'female'])
        monkey = {
            'x': x,
            'y': y,
            'vx': vx,
            'vy': vy,
            'last_eaten_time': time.time(),  # Initialize last eaten time
            'gender': gender
        }
        monkey_positions.append(monkey)
    return monkey_positions

def main():
    global root, canvas, cell_size, terrain, banana_matrix, monkey_positions  # Declare as global to use in update functions
    # Terrain dimensions and cell size
    rows = 64
    cols = 64
    cell_size = 10

    # Generate the terrain matrix
    terrain = generate_terrain(rows, cols)

    # Expand forests to create larger groupings
    terrain = expand_forests(terrain)

    # Initialize banana matrix (same size as terrain), all False initially
    banana_matrix = np.zeros((rows, cols), dtype=bool)

    # Initialize monkeys in forest areas
    num_monkeys = 5  # Adjust the number of monkeys as needed
    monkey_positions = initialize_monkeys(terrain, num_monkeys)

    # Initialize Tkinter
    root = tk.Tk()
    root.title("Procedurally Generated Terrain with Monkeys and Bananas")

    # Create the canvas
    canvas_width = cols * cell_size
    canvas_height = rows * cell_size
    canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
    canvas.pack()

    # Render the initial terrain onto the canvas
    render_canvas(terrain, banana_matrix, monkey_positions, canvas, cell_size)

    # Start the simulation update loop (monkeys move every 20 milliseconds)
    root.after(20, update_simulation)

    # Start the banana update loop (bananas regrow every 5 seconds)
    root.after(5000, update_bananas, terrain, banana_matrix)

    # Start the Tkinter event loop
    root.mainloop()

if __name__ == "__main__":
    main()
