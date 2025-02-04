
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import colors
import random
from collections import deque

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义动物基类
class Animal:
    def __init__(self, x, y, energy):
        self.x = x
        self.y = y
        self.energy = energy
        self.gender = random.choice(["M", "F"])
        self.has_reproduced = False
        self.path = [(x, y)]
        self.visual_x = x # For smoother visual movement
        self.visual_y = y
        self.target_x = x
        self.target_y = y

    def get_type(self):
        raise NotImplementedError

    def update_visual_position(self, frame_progress, speed_factor=0.2): # Control speed_factor for smoother movement (lower is slower, smoother)
        self.visual_x += (self.target_x - self.visual_x) * speed_factor * frame_progress
        self.visual_y += (self.target_y - self.visual_y) * speed_factor * frame_progress
        if abs(self.visual_x - self.target_x) < 0.01 and abs(self.visual_y - self.target_y) < 0.01: # to prevent overshooting
            self.visual_x = self.target_x
            self.visual_y = self.target_y

# 草食动物
class Herbivore(Animal):
    def get_type(self):
        return 'herbivore'

# 肉食动物
class Carnivore(Animal):
    def get_type(self):
        return 'carnivore'

# 生态系统仿真类
class Ecosystem:
    def __init__(self, grid_size=50,
                 init_plants=200,
                 init_herbivores=50,
                 init_carnivores=20,
                 plant_gain=5,
                 herbivore_move_cost=0.5,
                 herbivore_reproduce_threshold=40,
                 carnivore_move_cost=0.5,
                 carnivore_gain=10,
                 carnivore_reproduce_threshold=50,
                 plant_growth_prob=0.02,
                 herbivore_vision_range=7): # Herbivore vision range for cluster search

        self.grid_size = grid_size
        self.plant_grid = np.zeros((grid_size, grid_size), dtype=bool)
        self.animal_grid = [[None for _ in range(grid_size)] for _ in range(grid_size)]
        self.animals = []
        self.plant_gain = plant_gain
        self.herbivore_move_cost = herbivore_move_cost
        self.herbivore_reproduce_threshold = herbivore_reproduce_threshold
        self.carnivore_move_cost = carnivore_move_cost
        self.carnivore_gain = carnivore_gain
        self.carnivore_reproduce_threshold = carnivore_reproduce_threshold
        self.plant_growth_prob = plant_growth_prob
        self.herbivore_vision_range = herbivore_vision_range

        # 记录繁殖事件
        self.reproduction_events = []

        empty_cells = [(i, j) for i in range(grid_size) for j in range(grid_size)]
        init_plants_positions = random.sample(empty_cells, min(init_plants, len(empty_cells)))
        for (i, j) in init_plants_positions:
            self.plant_grid[i, j] = True

        available = [pos for pos in empty_cells if not self.plant_grid[pos[0], pos[1]]]
        init_herbivores_positions = random.sample(available, min(init_herbivores, len(available)))
        for (i, j) in init_herbivores_positions:
            herb = Herbivore(i, j, energy=20)
            self.animals.append(herb)
            self.animal_grid[i][j] = herb

        available = [(i, j) for i, j in empty_cells if self.animal_grid[i][j] is None]
        init_carnivores_positions = random.sample(available, min(init_carnivores, len(available)))
        for (i, j) in init_carnivores_positions:
            carn = Carnivore(i, j, energy=20)
            self.animals.append(carn)
            self.animal_grid[i][j] = carn

    def get_neighbors(self, x, y):
        neighbors = []
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                neighbors.append((nx, ny))
        return neighbors

    def bfs_find_path(self, start):
        # 针对 carnivore 使用 BFS 搜索最近的草食动物，返回完整路径
        visited = set()
        queue = deque()
        queue.append((start, [start]))
        visited.add(start)
        while queue:
            current, path = queue.popleft()
            if current != start:
                cx, cy = current
                occupant = self.animal_grid[cx][cy]
                if occupant is not None and occupant.get_type() == 'herbivore':
                    return path
            for neighbor in self.get_neighbors(*current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def find_plant_clusters(self, herbivore):
        clusters = []
        cluster_positions = set()
        for i in range(max(0, herbivore.x - self.herbivore_vision_range), min(self.grid_size, herbivore.x + self.herbivore_vision_range + 1)):
            for j in range(max(0, herbivore.y - self.herbivore_vision_range), min(self.grid_size, herbivore.y + self.herbivore_vision_range + 1)):
                if self.plant_grid[i, j] and (i, j) not in cluster_positions:
                    cluster = []
                    q = deque([(i, j)])
                    cluster_positions.add((i, j))
                    cluster.append((i,j))
                    while q:
                        cx, cy = q.popleft()
                        for nx, ny in self.get_neighbors(cx, cy):
                            if self.plant_grid[nx, ny] and (nx, ny) not in cluster_positions:
                                cluster_positions.add((nx, ny))
                                cluster.append((nx, ny))
                                q.append((nx, ny))
                    if cluster:
                        clusters.append(cluster)
        return clusters

    def plan_move(self, animal):
        current_pos = (animal.x, animal.y)
        if animal.get_type() == 'herbivore':
            clusters = self.find_plant_clusters(animal)
            if clusters:
                # Find closest cluster
                closest_cluster = min(clusters, key=lambda cluster: min(np.sqrt((animal.x - px)**2 + (animal.y - py)**2) for px, py in cluster)) # corrected line - used animal.x, animal.y
                target_plant_pos = random.choice(closest_cluster) # Go towards a random plant in the closest cluster
                path = [current_pos]
                cx, cy = current_pos
                target_x, target_y = target_plant_pos
                while (cx, cy) != target_plant_pos:
                    possible_moves = self.get_neighbors(cx, cy) + [(cx, cy)] # Include staying in place if stuck
                    best_move = None
                    min_distance = float('inf')
                    for nx, ny in possible_moves:
                        distance = np.sqrt((nx - target_x)**2 + (ny - target_y)**2)
                        if distance < min_distance and self.animal_grid[nx][ny] is None: # Ensure target cell is empty
                            min_distance = distance
                            best_move = (nx, ny)
                    if best_move:
                        path.append(best_move)
                        cx, cy = best_move
                    else: # Stuck, break to avoid infinite loop
                        break
                if len(path) > 1:
                    return path
                else:
                    neighbors = self.get_neighbors(animal.x, animal.y)
                    plant_neighbors = [ (nx, ny) for nx, ny in neighbors if self.plant_grid[nx, ny] ]
                    if plant_neighbors:
                        next_pos = random.choice(plant_neighbors)
                    else:
                        next_pos = random.choice(neighbors) if neighbors else (animal.x, animal.y)
                    return [current_pos, next_pos]

            else: # No clusters found, fallback to original behavior (neighbors or random)
                neighbors = self.get_neighbors(animal.x, animal.y)
                plant_neighbors = [ (nx, ny) for nx, ny in neighbors if self.plant_grid[nx, ny] ]
                if plant_neighbors:
                    next_pos = random.choice(plant_neighbors)
                else:
                    next_pos = random.choice(neighbors) if neighbors else (animal.x, animal.y)
                return [current_pos, next_pos]

        elif animal.get_type() == 'carnivore':
            path = self.bfs_find_path((animal.x, animal.y))
            if path is not None and len(path) > 1:
                return path
            else:
                neighbors = self.get_neighbors(animal.x, animal.y)
                next_pos = random.choice(neighbors) if neighbors else (animal.x, animal.y)
                return [current_pos, next_pos]
        else:
            return [current_pos]


    def reproduction_phase(self):
        # 性繁殖：只依赖能量条件，无需年龄判断
        for animal in self.animals:
            animal.has_reproduced = False

        animals_copy = self.animals[:]
        for animal in animals_copy:
            threshold = self.herbivore_reproduce_threshold if animal.get_type()=='herbivore' else self.carnivore_reproduce_threshold
            if animal.energy >= threshold and not animal.has_reproduced:
                neighbors = self.get_neighbors(animal.x, animal.y)
                mate_found = None
                for nx, ny in neighbors:
                    mate = self.animal_grid[nx][ny]
                    if mate is not None and mate.get_type() == animal.get_type() and mate.gender != animal.gender and not mate.has_reproduced:
                        mate_found = mate
                        break
                if mate_found:
                    child_energy = (animal.energy + mate_found.energy) // 4
                    animal.energy -= child_energy // 2
                    mate_found.energy -= child_energy // 2
                    animal.has_reproduced = True
                    mate_found.has_reproduced = True
                    candidate_positions = self.get_neighbors(animal.x, animal.y) + self.get_neighbors(mate_found.x, mate_found.y)
                    random.shuffle(candidate_positions)
                    offspring_pos = None
                    for pos in candidate_positions:
                        px, py = pos
                        if self.animal_grid[px][py] is None:
                            offspring_pos = pos
                            break
                    if offspring_pos is None:
                        offspring_pos = (animal.x, animal.y)
                    child = (Herbivore if animal.get_type()=='herbivore' else Carnivore)(offspring_pos[0], offspring_pos[1], energy=child_energy)
                    self.animals.append(child)
                    self.animal_grid[offspring_pos[0]][offspring_pos[1]] = child
                    self.reproduction_events.append({'pos': offspring_pos, 'duration': 10})
                else:
                    if animal.energy >= threshold * 1.5 and random.random() < 0.05:
                        child_energy = animal.energy // 4
                        animal.energy -= child_energy
                        animal.has_reproduced = True
                        offspring_pos = random.choice(self.get_neighbors(animal.x, animal.y))
                        child = (Herbivore if animal.get_type()=='herbivore' else Carnivore)(offspring_pos[0], offspring_pos[1], energy=child_energy)
                        self.animals.append(child)
                        self.animal_grid[offspring_pos[0]][offspring_pos[1]] = child
                        self.reproduction_events.append({'pos': offspring_pos, 'duration': 10})

    def step(self):
        # 以能量为依据，动物能量耗尽时死亡
        for animal in self.animals:
            animal.has_reproduced = False
        random.shuffle(self.animals)
        new_animals = []
        self.animal_grid = [[None for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        for animal in self.animals:
            if animal.energy <= 0:
                continue
            planned_path = self.plan_move(animal)
            if len(planned_path) > 1:
                nx, ny = planned_path[1]
            else:
                nx, ny = planned_path[0] # stay in place if only one position in path

            animal.target_x = nx # set target position for smooth move
            animal.target_y = ny

            if animal.get_type() == 'herbivore':
                if self.animal_grid[nx][ny] is None:
                    if self.plant_grid[nx, ny]:
                        animal.energy += self.plant_gain
                        self.plant_grid[nx, ny] = False
                    animal.x, animal.y = nx, ny
                    animal.path.append((nx, ny))
                animal.energy -= self.herbivore_move_cost
            elif animal.get_type() == 'carnivore':
                target_animal = self.animal_grid[nx][ny]
                if target_animal is not None and target_animal.get_type() == 'herbivore':
                    animal.energy += self.carnivore_gain
                    target_animal.energy = 0
                    animal.x, animal.y = nx, ny
                    animal.path.append((nx, ny))
                elif self.animal_grid[nx][ny] is None:
                    animal.x, animal.y = nx, ny
                    animal.path.append((nx, ny))
                animal.energy -= self.carnivore_move_cost
            if animal.energy > 0:
                new_animals.append(animal)
                self.animal_grid[animal.x][animal.y] = animal
        self.animals = new_animals
        self.grow_plants()
        self.reproduction_phase()
        # 仅依赖能量，动物能量为0则死亡（无需年龄判断）
        self.animals = [a for a in self.animals if a.energy > 0]
        for event in self.reproduction_events:
            event['duration'] -= 1
        self.reproduction_events = [event for event in self.reproduction_events if event['duration'] > 0]

    def grow_plants(self):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.animal_grid[i][j] is None and not self.plant_grid[i, j]:
                    if random.random() < self.plant_growth_prob:
                        self.plant_grid[i, j] = True

    def get_state(self):
        state = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.plant_grid[i, j]:
                    state[i, j] = 1
                if self.animal_grid[i][j] is not None:
                    if self.animal_grid[i][j].get_type() == 'herbivore':
                        state[i, j] = 2
                    elif self.animal_grid[i][j].get_type() == 'carnivore':
                        state[i, j] = 3
        return state

    # 统计数据方法
    def get_statistics(self):
        plants = np.sum(self.plant_grid)
        herbivores = [a for a in self.animals if a.get_type()=='herbivore']
        carnivores = [a for a in self.animals if a.get_type()=='carnivore']
        num_herb = len(herbivores)
        num_carn = len(carnivores)
        avg_energy_herb = sum(a.energy for a in herbivores) / num_herb if num_herb > 0 else 0
        avg_energy_carn = sum(a.energy for a in carnivores) / num_carn if num_carn > 0 else 0
        return {
            'Plants': plants,
            'Herbivores': num_herb,
            'Carnivores': num_carn,
            'Herbivore Avg Energy': avg_energy_herb,
            'Carnivore Avg Energy': avg_energy_carn
        }

# 主程序：使用 matplotlib 动态展示仿真过程、规划路径和统计面板
if __name__ == '__main__':
    sim = Ecosystem(grid_size=50, init_plants=200, init_herbivores=50, init_carnivores=20)

    # 创建左右两个子图：左侧显示仿真图像，右侧显示统计信息
    fig, (ax_sim, ax_stats) = plt.subplots(1, 2, figsize=(12, 6))
    cmap = colors.ListedColormap(['white', 'green', 'blue', 'red'])
    im = ax_sim.imshow(sim.get_state(), cmap=cmap, vmin=0, vmax=3)
    ax_sim.set_title("生态系统仿真")
    ax_sim.axis('off')
    ax_stats.axis('off')
    stats_text = ax_stats.text(0.05, 0.95, "", transform=ax_stats.transAxes, fontsize=12,
                               verticalalignment='top')

    planned_lines = []
    repro_markers = []
    animal_markers = [] # To update animal marker positions smoothly
    scatter_animals_herb = None
    scatter_animals_carn = None


    num_animation_frames_per_step = 10 # increase for smoother movement, decrease for faster animation (but less smooth)
    def update(frame):
        global scatter_animals_herb, scatter_animals_carn

        frame_progress = (frame % num_animation_frames_per_step) / num_animation_frames_per_step # for smooth animation
        if frame % num_animation_frames_per_step == 0: # do simulation step every num_animation_frames_per_step frames
            sim.step()

        for animal in sim.animals:
            animal.update_visual_position(frame_progress) # update visual position each frame

        im.set_data(sim.get_state())
        ax_sim.set_title(f"生态系统仿真 - 第 {frame // num_animation_frames_per_step} 步 (帧: {frame})")

        # Clear previous markers and lines
        if scatter_animals_herb:
            scatter_animals_herb.remove()
        if scatter_animals_carn:
            scatter_animals_carn.remove()
        for line in planned_lines:
            line.remove()
        planned_lines.clear()
        for marker in repro_markers:
            marker.remove()
        repro_markers.clear()

        # Plot animals as scatter plots for smooth animation
        herbivore_positions = [(animal.visual_y, animal.visual_x) for animal in sim.animals if animal.get_type() == 'herbivore']
        carnivore_positions = [(animal.visual_y, animal.visual_x) for animal in sim.animals if animal.get_type() == 'carnivore']

        if herbivore_positions:
            scatter_animals_herb = ax_sim.scatter([p[1] for p in herbivore_positions], [p[0] for p in herbivore_positions], color='blue', marker='o', s=50)
        else:
            scatter_animals_herb = None
        if carnivore_positions:
            scatter_animals_carn = ax_sim.scatter([p[1] for p in carnivore_positions], [p[0] for p in carnivore_positions], color='red', marker='^', s=50)
        else:
            scatter_animals_carn = None


        # Draw planned paths (still as lines but positions updated smoothly will look nicer)
        for animal in sim.animals:
            planned = sim.plan_move(animal)
            xs = [coord[1] for coord in planned]
            ys = [coord[0] for coord in planned]
            color = 'cyan' if animal.get_type()=='herbivore' else 'magenta'
            line, = ax_sim.plot(xs, ys, linestyle='--', color=color, linewidth=1, marker='o', markersize=2) # Reduced markersize for paths
            planned_lines.append(line)

        # Reproduction markers
        for event in sim.reproduction_events:
            ex, ey = event['pos']
            marker = ax_sim.scatter(ey, ex, s=150, facecolors='none', edgecolors='red', linewidths=2)
            repro_markers.append(marker)

        # Update stats panel
        stats = sim.get_statistics()
        stats_str = (f"步数: {frame // num_animation_frames_per_step}\n"
                     f"植物数: {stats['Plants']}\n"
                     f"草食动物: {stats['Herbivores']}\n"
                     f"肉食动物: {stats['Carnivores']}\n"
                     f"草食动物平均能量: {stats['Herbivore Avg Energy']:.1f}\n"
                     f"肉食动物平均能量: {stats['Carnivore Avg Energy']:.1f}")
        stats_text.set_text(stats_str)

        artists = [im] + planned_lines + repro_markers + [stats_text]
        if scatter_animals_herb:
            artists.append(scatter_animals_herb)
        if scatter_animals_carn:
            artists.append(scatter_animals_carn)

        return artists


    ani = FuncAnimation(fig, update, frames=200 * num_animation_frames_per_step, interval=50, blit=True) # faster interval
    plt.show()