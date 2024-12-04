#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math
from queue import PriorityQueue
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped
from create_plan_msgs.srv import CreatePlan
from nav2_simple_commander.robot_navigator import BasicNavigator

class AStarNode:
    _node_count = 0  # Class variable to generate unique IDs
    
    def __init__(self, position, g_cost=float('inf'), h_cost=0):
        self.position = position  # (x, y)
        self.g_cost = g_cost     # cost from start to current
        self.h_cost = h_cost     # heuristic cost to goal
        self.parent = None
        self.default_cost_thresholds = [50, 70, 90]  # Store default values
        self.cost_thresholds = self.default_cost_thresholds.copy()
        # Assign unique ID for tie-breaking
        self.id = AStarNode._node_count
        AStarNode._node_count += 1
        
    def f_cost(self):
        return self.g_cost + self.h_cost
        
    def __lt__(self, other):
        # First compare f_costs
        if self.f_cost() != other.f_cost():
            return self.f_cost() < other.f_cost()
        # If f_costs are equal, break ties using h_cost
        if self.h_cost != other.h_cost:
            return self.h_cost < other.h_cost
        # If still tied, use unique ID
        return self.id < other.id

class PathPlannerNode(Node):
    def __init__(self):
        super().__init__("path_planner_node")
        self.basic_navigator = BasicNavigator()
        self.srv = self.create_service(CreatePlan, 'create_plan', self.create_plan_cb)
        self.resolution = 0.05  # meters per cell
        self.max_search_time = 5.0  # seconds
        self.cost_thresholds = [50, 70, 90]  # Progressive cost thresholds
        
        # Path caching and reuse parameters
        self.last_path = None
        self.last_path_time = None
        self.path_reuse_timeout = 2.0  # seconds
        self.path_similarity_threshold = 0.5  # meters
        
        # Stuck detection parameters
        self.position_history = []
        self.history_length = 10
        self.stuck_threshold = 0.1
        self.stuck_time_threshold = 2.0
        self.last_position_time = None

    def create_plan_cb(self, request, response):
        try:
            current_time = self.get_clock().now()
            current_pos = (request.start.pose.position.x, request.start.pose.position.y)
            goal_pos = (request.goal.pose.position.x, request.goal.pose.position.y)

            # Check if we can reuse the last path
            if self.can_reuse_path(current_pos, goal_pos, current_time):
                self.get_logger().info("Reusing previous path")
                return self.last_path

            # Check if stuck before calculating new path
            if self.is_stuck(current_pos):
                self.get_logger().warn("Robot appears to be stuck, finding alternative path")
                self.cost_thresholds = [t + 20 for t in self.cost_thresholds]
            
            # Update position history
            self.update_position_history(current_pos)
            
            # Calculate new path
            costmap = self.basic_navigator.getGlobalCostmap()
            for cost_threshold in self.cost_thresholds:
                path_coords = self.try_find_path(request, cost_threshold, current_time)
                if path_coords:
                    response.path = self.create_path_msg(path_coords, 
                                                       request.goal.header.frame_id,
                                                       current_time.to_msg(),
                                                       costmap.metadata.origin.position.x,
                                                       costmap.metadata.origin.position.y)
                    # Cache the new path
                    self.last_path = response.path
                    self.last_path_time = current_time
                    # Reset cost thresholds after successful path finding
                    self.cost_thresholds = self.default_cost_thresholds.copy()
                    return response
            
            self.get_logger().error("Failed to find path with all cost thresholds")
            return response
        except Exception as e:
            self.get_logger().error(f"Error in create_plan_cb: {str(e)}")
            return response

    def can_reuse_path(self, current_pos, goal_pos, current_time):
        if not self.last_path or not self.last_path_time:
            return False
            
        # Check if enough time has passed since last path calculation
        time_since_last_path = (current_time - self.last_path_time).nanoseconds / 1e9
        if time_since_last_path < self.path_reuse_timeout:
            # Check if current position is close enough to the path
            if self.is_position_on_path(current_pos, self.last_path):
                # Check if goal hasn't changed significantly
                last_goal = self.last_path.poses[-1].pose.position
                last_goal_pos = (last_goal.x, last_goal.y)
                if self.distance(goal_pos, last_goal_pos) < self.path_similarity_threshold:
                    return True
        return False

    def is_position_on_path(self, position, path):
        for pose in path.poses:
            path_pos = (pose.pose.position.x, pose.pose.position.y)
            if self.distance(position, path_pos) < self.path_similarity_threshold:
                return True
        return False

    def distance(self, pos1, pos2):
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return math.sqrt(dx*dx + dy*dy)
            

    def update_position_history(self, current_pos):
        current_time = self.get_clock().now()
        
        # Clear history if too old
        if self.position_history and \
           (current_time - self.position_history[0][1]).nanoseconds / 1e9 > self.stuck_time_threshold * 2:
            self.position_history.clear()
            
        self.position_history.append((current_pos, current_time))
        if len(self.position_history) > self.history_length:
            self.position_history.pop(0)

    def is_stuck(self, current_pos):
        if len(self.position_history) < self.history_length:
            return False
            
        # Check if position hasn't changed significantly
        oldest_pos, oldest_time = self.position_history[0]
        current_time = self.get_clock().now()
        
        # Calculate distance moved
        dx = current_pos[0] - oldest_pos[0]
        dy = current_pos[1] - oldest_pos[1]
        distance_moved = math.sqrt(dx*dx + dy*dy)
        
        # Calculate time elapsed
        time_elapsed = (current_time - oldest_time).nanoseconds / 1e9
        
        # Consider stuck if minimal movement over threshold time
        return distance_moved < self.stuck_threshold and time_elapsed > self.stuck_time_threshold

    def try_find_path(self, request, cost_threshold, start_time):
        costmap = self.basic_navigator.getGlobalCostmap()
        origin_x = costmap.metadata.origin.position.x
        origin_y = costmap.metadata.origin.position.y
        
        start_point = (
            int((request.start.pose.position.x - origin_x) / self.resolution),
            int((request.start.pose.position.y - origin_y) / self.resolution)
        )
        goal_point = (
            int((request.goal.pose.position.x - origin_x) / self.resolution),
            int((request.goal.pose.position.y - origin_y) / self.resolution)
        )
        
        return self.astar(start_point, goal_point, costmap, cost_threshold, start_time)

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance

    def get_neighbors(self, current, costmap, cost_threshold):
        neighbors = []
        directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,1), (1,-1), (-1,-1)]
        
        width = costmap.metadata.size_x
        height = costmap.metadata.size_y
        
        for dx, dy in directions:
            new_x = current[0] + dx
            new_y = current[1] + dy
            
            if (0 <= new_x < width and 0 <= new_y < height):
                index = int(new_y * width + new_x)
                
                # Allow higher cost cells but with penalty
                if index < len(costmap.data) and costmap.data[index] < cost_threshold:
                    # Add penalty for higher cost cells
                    cost_penalty = 1.0 + (costmap.data[index] / 100.0)
                    neighbors.append((new_x, new_y, cost_penalty))
        return neighbors

    def astar(self, start, goal, costmap, cost_threshold, start_time):
        width = costmap.metadata.size_x
        height = costmap.metadata.size_y
        
        if not self.is_valid_position(start, goal, width, height):
            return []
            
        open_set = PriorityQueue()
        start_node = AStarNode(start, 0, self.heuristic(start, goal))
        open_set.put((start_node.f_cost(), start_node))
        
        closed_set = {}
        came_from = {}
        g_scores = {start: 0}
        
        while not open_set.empty():
            # Check timeout
            if (self.get_clock().now() - start_time).nanoseconds / 1e9 > self.max_search_time:
                self.get_logger().warn("Path finding timed out")
                return []
                
            current = open_set.get()[1]
            
            if current.position == goal:
                return self.reconstruct_path(came_from, current)
                
            closed_set[current.position] = current
            
            # Unpack all three values from get_neighbors
            for new_x, new_y, cost_penalty in self.get_neighbors(current.position, costmap, cost_threshold):
                neighbor_pos = (new_x, new_y)
                
                if neighbor_pos in closed_set:
                    continue
                    
                dx = abs(new_x - current.position[0])
                dy = abs(new_y - current.position[1])
                step_cost = (1.4 if dx + dy == 2 else 1.0) * cost_penalty
                
                tentative_g = current.g_cost + step_cost
                
                if neighbor_pos not in g_scores or tentative_g < g_scores[neighbor_pos]:
                    g_scores[neighbor_pos] = tentative_g
                    neighbor = AStarNode(neighbor_pos)
                    neighbor.g_cost = tentative_g
                    neighbor.h_cost = self.heuristic(neighbor_pos, goal)
                    neighbor.parent = current
                    
                    open_set.put((neighbor.f_cost(), neighbor))
                    came_from[neighbor_pos] = current.position
        
        return []

    def is_valid_position(self, start, goal, width, height):
        tolerance = 2
        return not (
            start[0] < -tolerance or start[0] >= width + tolerance or
            start[1] < -tolerance or start[1] >= height + tolerance or
            goal[0] < -tolerance or goal[0] >= width + tolerance or
            goal[1] < -tolerance or goal[1] >= height + tolerance
        )

    def reconstruct_path(self, came_from, current):
        path = []
        while current.position in came_from:
            path.append(current.position)
            current.position = came_from[current.position]
        path.append(current.position)
        return path[::-1]

    def create_path_msg(self, path_coords, frame_id, stamp, origin_x, origin_y):
        path_msg = Path()
        path_msg.header.frame_id = frame_id
        path_msg.header.stamp = stamp
        
        for x, y in path_coords:
            pose = PoseStamped()
            pose.header = path_msg.header
            # Convert back to world coordinates
            pose.pose.position.x = x * self.resolution + origin_x
            pose.pose.position.y = y * self.resolution + origin_y
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
            
        return path_msg

def main(args=None):
    rclpy.init(args=args)
    node = PathPlannerNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()