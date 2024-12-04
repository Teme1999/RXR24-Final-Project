#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from create_plan_msgs.srv import CreatePlan
from nav2_simple_commander.robot_navigator import BasicNavigator

import numpy as np

class PathPlannerNode(Node):

    def __init__(self):
        super().__init__("path_planner_node")
        self.basic_navigator = BasicNavigator()  # Uncommented to get Global Costmap in create_plan_cb

        # Creating a new service "create_plan", which is called by our Nav2 C++ planner plugin
        # to receive a plan from us.
        self.srv = self.create_service(CreatePlan, 'create_plan', self.create_plan_cb)

    def create_plan_cb(self, request, response):
        # Getting all the information to plan the path
        goal_pose = request.goal
        start_pose = request.start
        time_now = self.get_clock().now().to_msg()
        global_costmap = self.basic_navigator.getGlobalCostmap()  # Get Global Costmap

        response.path = create_dijkstra_plan(start_pose, goal_pose, time_now, global_costmap)
        return response

def create_dijkstra_plan(start, goal, time_now, costmap):
    """ 
    Creates a path using Dijkstra's algorithm between start and goal points,
    considering obstacles in the global costmap.
    """
    path = Path()
    path.header.frame_id = goal.header.frame_id
    path.header.stamp = time_now

    # Convert costmap data into a numpy array
    width = costmap.metadata.size_x
    height = costmap.metadata.size_y
    resolution = costmap.metadata.resolution
    origin = costmap.metadata.origin

    # Costmap data might be a flat list; reshape it to 2D
    costmap_data = np.array(costmap.data, dtype=np.int8).reshape((height, width))

    # Map coordinates to grid indices
    start_index = world_to_map(start.pose.position.x, start.pose.position.y, origin, resolution)
    goal_index = world_to_map(goal.pose.position.x, goal.pose.position.y, origin, resolution)

    # Run Dijkstra's algorithm to find the shortest path
    came_from = dijkstra(costmap_data, start_index, goal_index)

    if not came_from:
        # No path found
        return path

    # Reconstruct path
    grid_path = reconstruct_path(came_from, start_index, goal_index)

    # Convert grid indices back to world coordinates and create poses
    for idx in grid_path:
        world_coord = map_to_world(idx[0], idx[1], origin, resolution)
        pose = PoseStamped()
        pose.pose.position.x = world_coord[0]
        pose.pose.position.y = world_coord[1]
        pose.header.stamp = time_now
        pose.header.frame_id = goal.header.frame_id
        path.poses.append(pose)

    return path

def dijkstra(costmap, start, goal):
    """
    Implements Dijkstra's algorithm on the given costmap.
    """
    import heapq

    height, width = costmap.shape
    visited = np.full((height, width), False)
    distance = np.full((height, width), np.inf)
    came_from = {}

    distance[start] = 0
    queue = [(0, start)]

    while queue:
        current_distance, current = heapq.heappop(queue)
        if visited[current]:
            continue
        visited[current] = True

        if current == goal:
            return came_from

        x, y = current

        # Explore neighbors
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < height and 0 <= ny < width:
                if visited[nx, ny]:
                    continue
                # Check for obstacles
                if costmap[nx, ny] >= 100:  # Occupied cell
                    continue
                new_distance = current_distance + costmap[nx, ny] + 1  # Add movement cost
                if new_distance < distance[nx, ny]:
                    distance[nx, ny] = new_distance
                    heapq.heappush(queue, (new_distance, (nx, ny)))
                    came_from[(nx, ny)] = current

    # No path found
    return None

def reconstruct_path(came_from, start, goal):
    """
    Reconstructs the path from start to goal using the came_from dictionary.
    """
    current = goal
    path = [current]
    while current != start:
        current = came_from.get(current)
        if current is None:
            # Path reconstruction failed
            return []
        path.append(current)
    path.reverse()
    return path

def world_to_map(x, y, origin, resolution):
    """
    Converts world coordinates to map grid indices.
    """
    mx = int((x - origin.position.x) / resolution)
    my = int((y - origin.position.y) / resolution)
    return (my, mx)  # Note: costmap uses (row, column) format

def map_to_world(i, j, origin, resolution):
    """
    Converts map grid indices back to world coordinates.
    """
    x = origin.position.x + (j + 0.5) * resolution
    y = origin.position.y + (i + 0.5) * resolution
    return (x, y)

def main(args=None):
    rclpy.init(args=args)
    path_planner_node = PathPlannerNode()

    try:
        rclpy.spin(path_planner_node)
    except KeyboardInterrupt:
        pass

    path_planner_node.destroy_node()
    rclpy.try_shutdown()

if __name__ == '__main__':
    main()