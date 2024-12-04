#!/usr/bin/env python3
"""
ROS2 Path Planner Node using A* Algorithm

Provides path planning services using A* implementation that accounts for 
costmap obstacles and generates optimal paths between start and goal positions.
Handles coordinate transformations between world and grid space while considering
obstacles and cost thresholds.

Author: Teemu Partanen & Juho Matilainen
Date: 4.12.2024
"""

import rclpy
from rclpy.node import Node
import math
from queue import PriorityQueue
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped
from create_plan_msgs.srv import CreatePlan
from nav2_simple_commander.robot_navigator import BasicNavigator

class AStarNode:
    """A* algorithm node with position and cost tracking for pathfinding."""
    _node_count = 0
    
    def __init__(self, position, g_cost=float('inf'), h_cost=0):
        """Initialize node with position and path costs."""
        self.position = position
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.parent = None
        self.id = AStarNode._node_count
        AStarNode._node_count += 1
        
    def f_cost(self):
        """Return total estimated cost f = g + h."""
        return self.g_cost + self.h_cost
        
    def __lt__(self, other):
        """Define node ordering for priority queue."""
        if self.f_cost() != other.f_cost():
            return self.f_cost() < other.f_cost()
        if self.h_cost != other.h_cost:
            return self.h_cost < other.h_cost
        return self.id < other.id

class PathPlannerNode(Node):
    """ROS2 path planning node implementing A* search algorithm."""
    
    def __init__(self):
        """Initialize planner with navigation and service setup."""
        super().__init__("path_planner_node")
        self.basic_navigator = BasicNavigator()
        self.srv = self.create_service(CreatePlan, 'create_plan', self.create_plan_cb)
        self.resolution = 0.05

    def create_plan_cb(self, request, response):
        """Handle path planning service requests."""
        start_pose = request.start
        goal_pose = request.goal
        time_now = self.get_clock().now().to_msg()
        costmap = self.basic_navigator.getGlobalCostmap()
        
        origin_x = costmap.metadata.origin.position.x
        origin_y = costmap.metadata.origin.position.y
        
        start_point = (
            int((start_pose.pose.position.x - origin_x) / self.resolution),
            int((start_pose.pose.position.y - origin_y) / self.resolution)
        )
        goal_point = (
            int((goal_pose.pose.position.x - origin_x) / self.resolution),
            int((goal_pose.pose.position.y - origin_y) / self.resolution)
        )
        
        self.get_logger().info(f"Costmap size: {costmap.metadata.size_x}x{costmap.metadata.size_y}")
        self.get_logger().info(f"Start point (grid): {start_point}")
        self.get_logger().info(f"Goal point (grid): {goal_point}")
        
        path_coords = self.astar(start_point, goal_point, costmap)
        
        if not path_coords:
            self.get_logger().error("Failed to find valid path")
            return response
        
        response.path = self.create_path_msg(path_coords, goal_pose.header.frame_id, 
                                           time_now, origin_x, origin_y)
        return response

    def heuristic(self, a, b):
        """Calculate Manhattan distance between points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(self, current, costmap):
        """Return traversable neighboring cells from current position."""
        neighbors = []
        directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,1), (1,-1), (-1,-1)]
        
        width = costmap.metadata.size_x
        height = costmap.metadata.size_y
        
        for dx, dy in directions:
            new_x = current[0] + dx
            new_y = current[1] + dy
            
            if (0 <= new_x < width and 0 <= new_y < height):
                index = int(new_y * width + new_x)
                if index < len(costmap.data) and costmap.data[index] < 50:
                    neighbors.append((new_x, new_y))
        return neighbors

    def astar(self, start, goal, costmap):
        """Execute A* pathfinding algorithm."""
        width = costmap.metadata.size_x
        height = costmap.metadata.size_y
        tolerance = 2
        
        if not self._validate_positions(start, goal, width, height, tolerance):
            return []
        
        start = self._clamp_coordinates(start, width, height)
        goal = self._clamp_coordinates(goal, width, height)
        
        open_set = PriorityQueue()
        start_node = AStarNode(start, 0, self.heuristic(start, goal))
        open_set.put((start_node.f_cost(), start_node))
        
        closed_set = {}
        came_from = {}
        g_scores = {start: 0}
        
        while not open_set.empty():
            current = open_set.get()[1]
            
            if current.position == goal:
                return self.reconstruct_path(came_from, current)
            
            closed_set[current.position] = current
            
            for neighbor_pos in self.get_neighbors(current.position, costmap):
                if neighbor_pos in closed_set:
                    continue
                
                dx = abs(neighbor_pos[0] - current.position[0])
                dy = abs(neighbor_pos[1] - current.position[1])
                step_cost = 1.4 if dx + dy == 2 else 1.0
                
                tentative_g = current.g_cost + step_cost
                
                if neighbor_pos not in g_scores or tentative_g < g_scores[neighbor_pos]:
                    g_scores[neighbor_pos] = tentative_g
                    neighbor = AStarNode(neighbor_pos)
                    neighbor.g_cost = tentative_g
                    neighbor.h_cost = self.heuristic(neighbor_pos, goal)
                    neighbor.parent = current
                    
                    open_set.put((neighbor.f_cost(), neighbor))
                    came_from[neighbor_pos] = current.position
        
        self.get_logger().warn("No path found")
        return []

    def _validate_positions(self, start, goal, width, height, tolerance):
        """Check if positions are within valid map bounds."""
        return not (
            start[0] < -tolerance or start[0] >= width + tolerance or
            start[1] < -tolerance or start[1] >= height + tolerance or
            goal[0] < -tolerance or goal[0] >= width + tolerance or
            goal[1] < -tolerance or goal[1] >= height + tolerance
        )

    def _clamp_coordinates(self, point, width, height):
        """Ensure coordinates are within map boundaries."""
        return (max(0, min(width-1, point[0])), 
                max(0, min(height-1, point[1])))

    def reconstruct_path(self, came_from, current):
        """Build path by following parent references from goal to start."""
        path = []
        while current.position in came_from:
            path.append(current.position)
            current.position = came_from[current.position]
        path.append(current.position)
        return path[::-1]

    def create_path_msg(self, path_coords, frame_id, stamp, origin_x, origin_y):
        """Create ROS Path message from coordinate list."""
        path_msg = Path()
        path_msg.header.frame_id = frame_id
        path_msg.header.stamp = stamp
        
        for x, y in path_coords:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x * self.resolution + origin_x
            pose.pose.position.y = y * self.resolution + origin_y
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
            
        return path_msg

def main(args=None):
    """Entry point for the path planner node."""
    rclpy.init(args=args)
    node = PathPlannerNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()