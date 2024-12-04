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
        
    def create_plan_cb(self, request, response):
        start_time = self.get_clock().now()
        
        # Try multiple cost thresholds if initial path finding fails
        for cost_threshold in self.cost_thresholds:
            path_coords = self.try_find_path(request, cost_threshold, start_time)
            if path_coords:
                self.get_logger().info(f"Found path with cost threshold: {cost_threshold}")
                response.path = self.create_path_msg(path_coords, 
                                                   request.goal.header.frame_id,
                                                   self.get_clock().now().to_msg(),
                                                   costmap.metadata.origin.position.x,
                                                   costmap.metadata.origin.position.y)
                return response
                
        self.get_logger().error("Failed to find path with all cost thresholds")
        return response

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
            
            for neighbor_pos, cost_penalty in self.get_neighbors(current.position, costmap, cost_threshold):
                if neighbor_pos in closed_set:
                    continue
                    
                dx = abs(neighbor_pos[0] - current.position[0])
                dy = abs(neighbor_pos[1] - current.position[1])
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