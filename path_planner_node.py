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
    def __init__(self, position, g_cost=float('inf'), h_cost=0):
        self.position = position  # (x, y)
        self.g_cost = g_cost     # cost from start to current
        self.h_cost = h_cost     # heuristic cost to goal
        self.parent = None
        
    def f_cost(self):
        return self.g_cost + self.h_cost

class PathPlannerNode(Node):
    def __init__(self):
        super().__init__("path_planner_node")
        self.basic_navigator = BasicNavigator()
        self.srv = self.create_service(CreatePlan, 'create_plan', self.create_plan_cb)
        self.resolution = 0.05  # meters per cell
        
    def create_plan_cb(self, request, response):
        start_pose = request.start
        goal_pose = request.goal
        time_now = self.get_clock().now().to_msg()
        costmap = self.basic_navigator.getGlobalCostmap()
        
        # Convert poses to grid coordinates
        start_point = (int(start_pose.pose.position.x / self.resolution),
                      int(start_pose.pose.position.y / self.resolution))
        goal_point = (int(goal_pose.pose.position.x / self.resolution),
                     int(goal_pose.pose.position.y / self.resolution))
        
        # Get path using A*
        path_coords = self.astar(start_point, goal_point, costmap)
        
        # Convert to ROS Path message
        response.path = self.create_path_msg(path_coords, goal_pose.header.frame_id, time_now)
        return response

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance

    def get_neighbors(self, current, costmap):
        neighbors = []
        directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,1), (1,-1), (-1,-1)]
        
        for dx, dy in directions:
            new_x = current[0] + dx
            new_y = current[1] + dy
            
            # Check bounds and obstacles
            if (0 <= new_x < costmap.info.width and 
                0 <= new_y < costmap.info.height and 
                costmap.data[new_y * costmap.info.width + new_x] < 50):  # If cost < 50, assume traversable
                neighbors.append((new_x, new_y))
        return neighbors

    def astar(self, start, goal, costmap):
        open_set = PriorityQueue()
        start_node = AStarNode(start, 0, self.heuristic(start, goal))
        open_set.put((start_node.f_cost(), start_node))
        
        closed_set = {}
        came_from = {}
        
        while not open_set.empty():
            current = open_set.get()[1]
            
            if current.position == goal:
                return self.reconstruct_path(came_from, current)
            
            closed_set[current.position] = current
            
            for neighbor_pos in self.get_neighbors(current.position, costmap):
                if neighbor_pos in closed_set:
                    continue
                    
                tentative_g = current.g_cost + 1
                
                neighbor = AStarNode(neighbor_pos)
                neighbor.g_cost = tentative_g
                neighbor.h_cost = self.heuristic(neighbor_pos, goal)
                neighbor.parent = current
                
                open_set.put((neighbor.f_cost(), neighbor))
                came_from[neighbor_pos] = current.position
                
        return []  # No path found

    def reconstruct_path(self, came_from, current):
        path = []
        while current.position in came_from:
            path.append(current.position)
            current.position = came_from[current.position]
        path.append(current.position)
        return path[::-1]

    def create_path_msg(self, path_coords, frame_id, stamp):
        path_msg = Path()
        path_msg.header.frame_id = frame_id
        path_msg.header.stamp = stamp
        
        for x, y in path_coords:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x * self.resolution
            pose.pose.position.y = y * self.resolution
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