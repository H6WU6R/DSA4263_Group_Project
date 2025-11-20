"""
Example Usage Script
====================
Demonstrates how to use the feature extraction and visualization modules.
"""

from feature_extraction import EmailGraphFeatureExtractor
from visualization import EmailGraphVisualizer

# ============================================================================
# METHOD 1: Using the convenience functions (Quick & Easy)
# ============================================================================

print("METHOD 1: Quick usage with convenience functions")
print("="*70 + "\n")

from feature_extraction import extract_features
from visualization import visualize_email_graph

# Extract features
features_df = extract_features("data/processed/graph_merge.csv")

# Create all visualizations (this requires access to the extractor's G and df)
# So Method 2 below is recommended for full workflow

print("\n" + "="*70 + "\n")

# ============================================================================
# METHOD 2: Using the classes (Recommended - Full Control)
# ============================================================================

print("METHOD 2: Using classes for full control")
print("="*70 + "\n")

# Step 1: Extract features
extractor = EmailGraphFeatureExtractor()
features_df = extractor.get_features("data/processed/graph_merge.csv")

# Optional: Get feature comparison
comparison = extractor.get_feature_comparison()

# Step 2: Create visualizations
visualizer = EmailGraphVisualizer(features_df, extractor.G, extractor.df)

# Create dashboard with individual panels saved
visualizer.create_dashboard(
    save_path='spam_eda_dashboard.png',
    save_individual=True,  # Save each panel separately
    individual_dir='individual_panels'  # Directory for individual panels
)

# Create flow graph
visualizer.create_flow_graph(save_path='spam_flow_graph.png')

# Or create all at once
visualizer.create_all_figures(
    dashboard_path='spam_eda_dashboard.png',
    flow_path='spam_flow_graph.png',
    save_individual=True,
    individual_dir='individual_panels'
)

print("\n" + "="*70)
print(" COMPLETE WORKFLOW FINISHED!")
print("="*70)
print("\nOutput files:")
print("  - features_df: DataFrame with all extracted features")
print("  - spam_eda_dashboard.png: 8-panel comprehensive dashboard")
print("  - spam_flow_graph.png: Communication flow visualization")
print("  - individual_panels/: Folder with 8 separate panel images")
print("    * panel1_sender_distribution.png")
print("    * panel2_out_degree.png")
print("    * panel3_reciprocity.png")
print("    * panel4_degree_scatter.png")
print("    * panel5_clustering.png")
print("    * panel6_triangles.png")
print("    * panel7_network_sample.png")
print("    * panel8_key_insights.png")

# ============================================================================
# Access the extracted data
# ============================================================================

# The features DataFrame
print("\nFeatures DataFrame columns:")
print(features_df.columns.tolist())

# The graph
print(f"\nGraph statistics:")
print(f"  Nodes: {extractor.G.number_of_nodes()}")
print(f"  Edges: {extractor.G.number_of_edges()}")

# Save features to CSV
features_df.to_csv('extracted_features.csv', index=False)
print("\n✓ Features saved to 'extracted_features.csv'")

# ============================================================================
# OPTIONAL: Create only specific visualizations
# ============================================================================

print("\n" + "="*70)
print(" CREATING SPECIFIC VISUALIZATIONS")
print("="*70 + "\n")

# Create only the dashboard (without individual panels)
visualizer.create_dashboard(
    save_path='dashboard_only.png',
    save_individual=False  # Don't save individual panels
)

# Create only the flow graph
visualizer.create_flow_graph(save_path='flow_only.png')

print("\n✓ All done!")
