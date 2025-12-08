#!/usr/bin/env python3
"""
Create a clustering overview diagram showing different clustering approaches.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import numpy as np

def create_clustering_overview():
    """Create a comprehensive clustering overview diagram."""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'CS2 Player Role Clustering Approaches', 
            fontsize=20, fontweight='bold', ha='center')
    
    # Main clustering types
    # Partitional Clustering (K-means)
    kmeans_box = FancyBboxPatch((0.5, 6.5), 4, 2.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor='lightblue', edgecolor='blue', linewidth=2)
    ax.add_patch(kmeans_box)
    ax.text(2.5, 7.8, 'PARTITIONAL CLUSTERING', fontsize=14, fontweight='bold', ha='center')
    ax.text(2.5, 7.4, 'K-Means Algorithm', fontsize=12, fontweight='bold', ha='center')
    
    # K-means details
    ax.text(0.7, 7.0, '• Divides data into k clusters', fontsize=10, ha='left')
    ax.text(0.7, 6.8, '• Minimizes within-cluster variance', fontsize=10, ha='left')
    ax.text(0.7, 6.6, '• Each point belongs to one cluster', fontsize=10, ha='left')
    
    # Hierarchical Clustering
    hier_box = FancyBboxPatch((5.5, 6.5), 4, 2.5, 
                             boxstyle="round,pad=0.1", 
                             facecolor='lightgreen', edgecolor='green', linewidth=2)
    ax.add_patch(hier_box)
    ax.text(7.5, 7.8, 'HIERARCHICAL CLUSTERING', fontsize=14, fontweight='bold', ha='center')
    ax.text(7.5, 7.4, 'Agglomerative Approach', fontsize=12, fontweight='bold', ha='center')
    
    # Hierarchical details
    ax.text(5.7, 7.0, '• Creates tree-like structure', fontsize=10, ha='left')
    ax.text(5.7, 6.8, '• Bottom-up merging process', fontsize=10, ha='left')
    ax.text(5.7, 6.6, '• Shows cluster relationships', fontsize=10, ha='left')
    
    # Distance Metrics Section
    dist_box = FancyBboxPatch((1, 3.5), 8, 2, 
                             boxstyle="round,pad=0.1", 
                             facecolor='lightyellow', edgecolor='orange', linewidth=2)
    ax.add_patch(dist_box)
    ax.text(5, 5.2, 'DISTANCE METRICS', fontsize=14, fontweight='bold', ha='center')
    
    # Euclidean Distance
    ax.text(2, 4.6, 'Euclidean Distance', fontsize=12, fontweight='bold', ha='center')
    ax.text(2, 4.3, '• Standard geometric distance', fontsize=10, ha='center')
    ax.text(2, 4.1, '• Used in K-means clustering', fontsize=10, ha='center')
    ax.text(2, 3.9, '• Measures straight-line distance', fontsize=10, ha='center')
    
    # Cosine Similarity
    ax.text(8, 4.6, 'Cosine Similarity', fontsize=12, fontweight='bold', ha='center')
    ax.text(8, 4.3, '• Measures angle between vectors', fontsize=10, ha='center')
    ax.text(8, 4.1, '• Used in hierarchical clustering', fontsize=10, ha='center')
    ax.text(8, 3.9, '• Magnitude-independent', fontsize=10, ha='center')
    
    # Discovery Section
    disc_box = FancyBboxPatch((1, 1), 8, 1.8, 
                             boxstyle="round,pad=0.1", 
                             facecolor='lightcoral', edgecolor='red', linewidth=2)
    ax.add_patch(disc_box)
    ax.text(5, 2.5, 'DISCOVERY THROUGH CLUSTERING', fontsize=14, fontweight='bold', ha='center')
    
    # Discovery points
    ax.text(1.2, 2.1, '• Natural player role groupings', fontsize=10, ha='left')
    ax.text(1.2, 1.8, '• Behavioral pattern identification', fontsize=10, ha='left')
    ax.text(5.5, 2.1, '• Optimal team composition', fontsize=10, ha='left')
    ax.text(5.5, 1.8, '• Player development pathways', fontsize=10, ha='left')
    
    # Arrows showing flow
    # From clustering types to distance metrics
    arrow1 = patches.FancyArrowPatch((2.5, 6.5), (2.5, 5.5),
                                   arrowstyle='->', mutation_scale=20, 
                                   color='black', linewidth=2)
    ax.add_patch(arrow1)
    
    arrow2 = patches.FancyArrowPatch((7.5, 6.5), (7.5, 5.5),
                                   arrowstyle='->', mutation_scale=20, 
                                   color='black', linewidth=2)
    ax.add_patch(arrow2)
    
    # From distance metrics to discovery
    arrow3 = patches.FancyArrowPatch((5, 3.5), (5, 2.8),
                                   arrowstyle='->', mutation_scale=20, 
                                   color='black', linewidth=2)
    ax.add_patch(arrow3)
    
    # Add some visual elements
    # Player icons
    for i, (x, y) in enumerate([(1, 8.5), (3, 8.5), (6, 8.5), (8, 8.5)]):
        circle = Circle((x, y), 0.15, facecolor='gray', edgecolor='black')
        ax.add_patch(circle)
        ax.text(x, y-0.4, f'Player {i+1}', fontsize=8, ha='center')
    
    # Cluster visualization
    # K-means clusters
    cluster1 = Circle((1.5, 5.8), 0.3, facecolor='red', alpha=0.5)
    cluster2 = Circle((3.5, 5.8), 0.3, facecolor='blue', alpha=0.5)
    ax.add_patch(cluster1)
    ax.add_patch(cluster2)
    ax.text(1.5, 5.3, 'Cluster 1', fontsize=9, ha='center')
    ax.text(3.5, 5.3, 'Cluster 2', fontsize=9, ha='center')
    
    # Hierarchical tree
    ax.plot([7, 7.5, 8], [5.8, 6.2, 5.8], 'g-', linewidth=2)
    ax.plot([7.5, 7.5], [6.2, 6.6], 'g-', linewidth=2)
    ax.plot([7.2, 7.8], [6.6, 6.6], 'g-', linewidth=2)
    ax.text(7.5, 6.8, 'Dendrogram', fontsize=9, ha='center')
    
    plt.tight_layout()
    plt.savefig('../src/views/img/clustering_overview.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print("Clustering overview diagram created successfully!")
    print("Saved as: ../src/views/img/clustering_overview.png")

if __name__ == "__main__":
    create_clustering_overview()
