import networkx as nx
from matplotlib.patches import Patch
from collections import defaultdict, deque
import matplotlib.pyplot as plt


def assign_positions(G, root):
    """
    Assigns positions to nodes in a hierarchical layout.

    Parameters:
        G (nx.DiGraph): The tree graph.
        root (node): The root node of the tree.

    Returns:
        dict: A dictionary mapping nodes to positions.
    """
    pos = {}
    x = 0

    def assign_x(node, depth):
        nonlocal x
        children = list(G.successors(node))
        if not children:
            pos[node] = (x, -depth)
            x += 1
        else:
            for child in children:
                assign_x(child, depth + 1)
            child_x = [pos[child][0] for child in children]
            pos[node] = (sum(child_x) / len(child_x), -depth)

    assign_x(root, 0)
    return pos


def is_tree(G: nx.DiGraph) -> bool:
    """
    Checks if a given directed graph is a tree.

    Parameters:
        G (nx.DiGraph): The directed graph to check.

    Returns:
        bool: True if G is a tree, False otherwise.
    """
    # A directed tree must be a directed acyclic graph (DAG) with exactly one root
    if not nx.is_directed_acyclic_graph(G):
        print("Graph is not a Directed Acyclic Graph (DAG).")
        return False

    # Identify potential roots (nodes with in_degree 0)
    roots = [n for n, d in G.in_degree() if d == 0]
    if len(roots) != 1:
        print(f"Graph has {len(roots)} roots; a tree must have exactly one root.")
        return False

    # Check if all nodes are reachable from the root
    root = roots[0]
    descendants = nx.descendants(G, root)
    if len(descendants) + 1 != len(G.nodes()):
        print("Not all nodes are reachable from the root.")
        return False

    return True


def visualize_tree(tree, save_path=None, figsize=(50, 40), dpi=300):
    """
    Visualizes the tree with hierarchical layout.
    Nodes with spectrum=None are colored red, others are colored blue.
    For large trees, it can save the visualization as a high-resolution image.

    Parameters:
        tree (Tree): The tree to visualize.
        save_path (str, optional): Path to save the image. If None, displays the plot.
        figsize (tuple, optional): Size of the matplotlib figure.
        dpi (int, optional): Resolution of the saved image.
    """
    G = nx.DiGraph()
    queue = deque()
    queue.append(tree.root)
    visited = set()

    while queue:
        node = queue.popleft()
        if id(node) in visited:
            continue
        visited.add(id(node))
        G.add_node(id(node), value=node.value, spectrum=node.spectrum)
        for child in node.children.values():
            G.add_edge(id(node), id(child))
            queue.append(child)

    try:
        if not is_tree(G):
            print("Warning: The graph is not a tree. Visualization may not be accurate.")
    except Exception as e:
        print(f"Error checking if graph is a tree: {e}")

    node_colors = []
    for node in G.nodes(data=True):
        if node[1]['spectrum'] is None:
            node_colors.append('red')
        else:
            node_colors.append('blue')

    label_counts = defaultdict(int)
    labels = {}
    for node_id, attrs in G.nodes(data=True):
        label = f"{attrs['value']:.3f}"
        count = label_counts[label]
        unique_label = f"{count}.{label}" if count > 0 else label
        labels[node_id] = unique_label
        label_counts[label] += 1

    try:
        root_node = id(tree.root)
        pos = assign_positions(G, root_node)
    except ValueError as ve:
        print(f"Error in hierarchical layout: {ve}")
        print("Falling back to spring layout.")
        pos = nx.spring_layout(G, seed=42)
    except Exception as e:
        print(f"Unexpected error in hierarchical layout: {e}")
        pos = nx.spring_layout(G, seed=42)

    if save_path:
        plt.figure(figsize=figsize, dpi=dpi)
    else:
        plt.figure(figsize=figsize)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8)

    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=15, edge_color='gray')

    nx.draw_networkx_labels(G, pos, labels, font_size=6)

    legend_elements = [
        Patch(facecolor='blue', edgecolor='black', label='Spectrum Present'),
        Patch(facecolor='red', edgecolor='black', label='Spectrum Missing')
    ]
    plt.legend(handles=legend_elements, loc='best')

    plt.title("Tree Visualization with Missing Spectra Nodes Highlighted")
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Tree visualization saved to {save_path}")
    else:
        plt.show()

    plt.close()