import copy

class Node:
    def __init__(self, node_id, capacity):
        self.node_id = node_id
        self.capacity = capacity  # c_j
        self.assigned_items = []  # List of item IDs
        self.assigned_sum = 0  # Sum of b_{i, j} for items assigned to this node
        self.T_j = 0  # Total sum as defined
        self.parent = None
        self.children = []

    def update_T_j(self):
        if not self.children:
            # Leaf node
            self.T_j = self.assigned_sum
        else:
            # Internal node
            max_child_T = max(child.T_j for child in self.children)
            self.T_j = self.assigned_sum + max_child_T

    def __str__(self):
        return f"Node {self.node_id}: T_j={self.T_j}, assigned_sum={self.assigned_sum}, assigned_items={self.assigned_items}"

class Item:
    def __init__(self, item_id, attribute):
        self.item_id = item_id
        self.attribute = attribute  # a_i
        self.possible_nodes = []  # List of nodes where a_i <= c_j
        self.b_values = {}  # Dictionary of b_{i, j} for possible nodes

def build_tree_structure():
    # Build the tree structure T according to the problem.
    # For simplicity, we'll create a sample tree here.
    # In a real problem, this would be provided or read from input.
    nodes = {}
    root1 = Node(1, capacity=15)
    root2 = Node(2, capacity=15)
    node3 = Node(3, capacity=10)
    node4 = Node(4, capacity=10)
    # Build relationships
    root1.children.append(node3)
    node3.parent = root1
    root2.children.append(node4)
    node4.parent = root2
    # Add nodes to the dictionary
    nodes[1] = root1
    nodes[2] = root2
    nodes[3] = node3
    nodes[4] = node4
    roots = [root1, root2]
    return nodes, roots

def assign_items_to_nodes(items, nodes, roots):
    # Precompute possible nodes for each item
    for item in items:
        item.possible_nodes = [node for node in nodes.values() if item.attribute <= node.capacity]
    # Sort items by decreasing attribute (a_i), you can also consider b_{i, j}
    items.sort(key=lambda x: -x.attribute)
    for item in items:
        min_max_T = float('inf')
        best_node = None
        # Try assigning to each possible node
        for node in item.possible_nodes:
            # Simulate the assignment
            original_assigned_sum = node.assigned_sum
            original_T_j = {}
            current_node = node
            while current_node:
                original_T_j[current_node.node_id] = current_node.T_j
                current_node = current_node.parent
            # Assign item to node
            node.assigned_items.append(item.item_id)
            node.assigned_sum += item.b_values.get(node.node_id, 0)
            # Update T_j up to root
            current_node = node
            while current_node:
                current_node.update_T_j()
                current_node = current_node.parent
            # Check the maximum T_j among roots
            current_max_T = max(root.T_j for root in roots)
            if current_max_T < min_max_T:
                min_max_T = current_max_T
                best_node = node
            # Undo the assignment
            node.assigned_items.pop()
            node.assigned_sum = original_assigned_sum
            # Restore T_j values
            for nid, tj_value in original_T_j.items():
                nodes[nid].T_j = tj_value
        # Assign the item to the best node found
        if best_node:
            best_node.assigned_items.append(item.item_id)
            best_node.assigned_sum += item.b_values.get(best_node.node_id, 0)
            # Update T_j up to root
            current_node = best_node
            while current_node:
                current_node.update_T_j()
                current_node = current_node.parent
        else:
            print(f"Item {item.item_id} cannot be assigned to any node.")
    # Print the assignments
    print("Final assignments:")
    for node in nodes.values():
        print(node)
    max_T_j = max(root.T_j for root in roots)
    print(f"\nMinimum possible maximum T_j among roots: {max_T_j}")

if __name__ == "__main__":
        # Build the tree structure
    nodes, roots = build_tree_structure()
    # Create items with their attributes (a_i) and b_{i, j} values
    items = []
    # Sample items (In practice, input data would be used)
    item1 = Item(1, attribute=5)
    item1.b_values = {1: 12, 3: 8}
    item2 = Item(2, attribute=7)
    item2.b_values = {1: 10, 3: 7, 2: 11, 4: 9}
    item3 = Item(3, attribute=9)
    item3.b_values = {2: 14, 4: 10}
    item4 = Item(4, attribute=6)
    item4.b_values = {1: 9, 3: 6}
    items.extend([item1, item2, item3, item4])
    # Assign items to nodes
    assign_items_to_nodes(items, nodes, roots)
