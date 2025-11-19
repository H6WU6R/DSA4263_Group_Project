"""
Graph Visualization Module
===========================
Produces comprehensive visualizations for email spam detection analysis.
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class EmailGraphVisualizer:
    """
    Creates comprehensive visualizations for email spam network analysis.
    
    Available visualizations:
    1. Comprehensive Dashboard (8-panel visualization)
    2. Communication Flow Graph (Spam ↔ Non-Spam flow)
    """
    
    def __init__(self, features_df: pd.DataFrame, G: nx.DiGraph, df: pd.DataFrame):
        """
        Initialize visualizer with data.
        
        Args:
            features_df: DataFrame with extracted features
            G: NetworkX directed graph
            df: Original email dataframe
        """
        self.features_df = features_df
        self.G = G
        self.df = df
        
        # Identify spammers
        self.spammers = set(features_df[features_df['is_spammer'] == 1]['sender'].values)
        
        # Color scheme
        self.spam_color = '#FF6F61'    # Living Coral
        self.legit_color = '#5AC8A8'   # Soft Aqua
        self.accent_color = '#6C7A89'  # Modern Grey
        
    def create_dashboard(self, save_path: Optional[str] = None) -> None:
        """
        Create comprehensive 8-panel visualization dashboard.
        
        Args:
            save_path: Optional path to save the figure
        """
        print("Creating comprehensive dashboard...")
        
        fig = plt.figure(figsize=(20, 10))
        fig.patch.set_facecolor('#FAFAFA')
        fig.suptitle('Email Spam Detection: Graph-Based Analysis Dashboard', 
                     fontsize=20, fontweight='bold', y=0.98, color='#2C3E50')
        
        # Convenience subsets
        spam_df = self.features_df[self.features_df['is_spammer'] == 1]
        legit_df = self.features_df[self.features_df['is_spammer'] == 0]
        
        # Panel 1: Sender Distribution
        self._plot_sender_distribution(plt.subplot(2, 4, 1), spam_df, legit_df)
        
        # Panel 2: Out-Degree Distribution
        self._plot_out_degree_distribution(plt.subplot(2, 4, 2), spam_df, legit_df)
        
        # Panel 3: Reciprocity Comparison
        self._plot_reciprocity_violin(plt.subplot(2, 4, 3), spam_df, legit_df)
        
        # Panel 4: In-Degree vs Out-Degree
        self._plot_degree_scatter(plt.subplot(2, 4, 4), spam_df, legit_df)
        
        # Panel 5: Clustering Coefficient
        self._plot_clustering(plt.subplot(2, 4, 5), spam_df, legit_df)
        
        # Panel 6: Triangles
        self._plot_triangles(plt.subplot(2, 4, 6), spam_df, legit_df)
        
        # Panel 7: Network Visualization
        self._plot_network_sample(plt.subplot(2, 4, 7), spam_df)
        
        # Panel 8: Key Insights
        self._plot_key_insights(plt.subplot(2, 4, 8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#FAFAFA')
            print(f"Dashboard saved to {save_path}")
        
        plt.show()
        print("✓ Dashboard created successfully!\n")
    
    def _plot_sender_distribution(self, ax, spam_df, legit_df):
        """Panel 1: Pie chart of sender distribution."""
        ax.set_facecolor('white')
        
        sender_counts = [len(spam_df), len(legit_df)]
        colors = [self.spam_color, self.legit_color]
        labels = ['Spammers', 'Legitimate']
        
        wedges, texts, autotexts = ax.pie(
            sender_counts, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90, explode=(0.05, 0),
            textprops={'fontsize': 11, 'color': self.accent_color}
        )
        
        ax.set_title('Sender Distribution', fontweight='bold', fontsize=13, color=self.accent_color)
    
    def _plot_out_degree_distribution(self, ax, spam_df, legit_df):
        """Panel 2: Histogram of out-degree distribution."""
        ax.set_facecolor('white')
        
        bins = np.linspace(0, 50, 26)
        
        ax.hist(spam_df['out_degree'], bins=bins, alpha=0.70, color=self.spam_color, 
                label=f"Spam (mean={spam_df['out_degree'].mean():.1f})", edgecolor='black')
        ax.hist(legit_df['out_degree'], bins=bins, alpha=0.70, color=self.legit_color, 
                label=f"Legit (mean={legit_df['out_degree'].mean():.1f})", edgecolor='black')
        
        ax.set_xlabel('Out-Degree', fontsize=11, fontweight='bold', color=self.accent_color)
        ax.set_ylabel('Count (log scale)', fontsize=11, fontweight='bold', color=self.accent_color)
        ax.set_yscale('log')
        ax.set_title('Out-Degree Distribution (Spam vs Legit)', fontweight='bold', fontsize=12, color=self.accent_color)
        ax.grid(alpha=0.25, linestyle='--', linewidth=0.5)
        ax.legend(frameon=True)
    
    def _plot_reciprocity_violin(self, ax, spam_df, legit_df):
        """Panel 3: Violin plot of reciprocity comparison."""
        ax.set_facecolor('white')
        
        data_recip = [spam_df['reciprocity'], legit_df['reciprocity']]
        
        parts = ax.violinplot(
            data_recip, positions=[1, 2], widths=0.7,
            showmeans=True, showmedians=True
        )
        
        # Color violins
        parts['bodies'][0].set_facecolor(self.spam_color)
        parts['bodies'][0].set_edgecolor('black')
        parts['bodies'][1].set_facecolor(self.legit_color)
        parts['bodies'][1].set_edgecolor('black')
        
        # Line colors
        for partname in ['cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans']:
            if partname in parts:
                parts[partname].set_edgecolor(self.accent_color)
                parts[partname].set_linewidth(1.6)
        
        ax.set_xticks([1, 2])
        ax.set_xticklabels(
            [f"Spam\nmean={spam_df['reciprocity'].mean():.3f}",
             f"Legit\nmean={legit_df['reciprocity'].mean():.3f}"],
            color=self.accent_color, fontsize=10
        )
        ax.set_ylabel('Reciprocity', fontsize=11, fontweight='bold', color=self.accent_color)
        ax.set_title('Victims Rarely Reply to Spam', fontsize=12, fontweight='bold', color=self.accent_color)
        ax.grid(axis='y', alpha=0.25, linestyle='--')
    
    def _plot_degree_scatter(self, ax, spam_df, legit_df):
        """Panel 4: Scatter plot of in-degree vs out-degree."""
        ax.set_facecolor('white')
        
        # Sampling
        n_sample = 2000
        spam_sample = spam_df.sample(min(n_sample, len(spam_df)))
        legit_sample = legit_df.sample(min(n_sample, len(legit_df)))
        
        # Add jitter to spam points for visibility
        spam_y = spam_sample['in_degree'] + np.random.uniform(0.08, 0.35, len(spam_sample))
        legit_y = legit_sample['in_degree']
        
        # Glow effect under spam points
        ax.scatter(
            spam_sample['out_degree'], spam_y,
            s=120, alpha=0.15, color=self.spam_color,
            linewidths=0, edgecolors='none'
        )
        
        # Spam points (foreground)
        ax.scatter(
            spam_sample['out_degree'], spam_y,
            c=self.spam_color,
            alpha=0.95,
            s=65,
            linewidths=1.2, 
            edgecolors='#444444',
            label='Spam'
        )
        
        # Legit points (subtle)
        ax.scatter(
            legit_sample['out_degree'], legit_y,
            c=self.legit_color,
            alpha=0.45,
            s=35,
            linewidths=0.3,
            edgecolors='#999999',
            label='Legit'
        )
        
        # Reference diagonal
        max_val = 50
        ax.plot([0, max_val], [0, max_val],
                color=self.accent_color, linestyle='--', linewidth=1.4)
        
        ax.set_xlabel('Out-Degree', fontsize=11, fontweight='bold', color=self.accent_color)
        ax.set_ylabel('In-Degree', fontsize=11, fontweight='bold', color=self.accent_color)
        ax.set_title('One-Way Broadcast Behavior', fontsize=12, fontweight='bold', color=self.accent_color)
        
        ax.set_xlim(-1, max_val)
        ax.set_ylim(-0.5, max_val)
        
        ax.legend(frameon=True)
        ax.grid(alpha=0.25, linestyle='--')
    
    def _plot_clustering(self, ax, spam_df, legit_df):
        """Panel 5: Bar chart of clustering coefficient."""
        ax.set_facecolor('white')
        
        clust_means = [
            spam_df['clustering'].mean(),
            legit_df['clustering'].mean()
        ]
        
        bars = ax.bar(['Spam', 'Legit'], clust_means,
                      color=[self.spam_color, self.legit_color], edgecolor='black')
        
        ax.set_ylabel('Clustering Coefficient (log scale)', fontsize=11, fontweight='bold', color=self.accent_color)
        ax.set_yscale('log')
        ax.set_title('Clustering: Social Embeddedness', fontweight='bold', fontsize=12, color=self.accent_color)
        ax.grid(axis='y', linestyle='--', alpha=0.25)
        
        # Labels
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width()/2., bar.get_height()*1.5,
                f"{bar.get_height():.4f}",
                ha='center', fontsize=10, color=self.accent_color
            )
    
    def _plot_triangles(self, ax, spam_df, legit_df):
        """Panel 6: Bar chart of triangle participation."""
        ax.set_facecolor('white')
        
        tri_means = [
            spam_df['triangles'].mean() if 'triangles' in spam_df.columns else 0,
            legit_df['triangles'].mean() if 'triangles' in legit_df.columns else 0
        ]
        
        bars = ax.bar(['Spam', 'Legit'], [tri_means[0] + 1e-6, tri_means[1]],
                      color=[self.spam_color, self.legit_color], edgecolor='black')
        
        ax.set_ylabel('Average Triangles (log scale)', fontsize=11, fontweight='bold', color=self.accent_color)
        ax.set_yscale('log')
        ax.set_title('Triangle Participation', fontweight='bold', fontsize=12, color=self.accent_color)
        ax.grid(axis='y', linestyle='--', alpha=0.25)
    
    def _plot_network_sample(self, ax, spam_df):
        """Panel 7: Network visualization of spam hub structure."""
        ax.set_facecolor('white')
        ax.set_title('Network Sample\nSpam Hubs Broadcasting to Isolated Recipients',
                     fontsize=12, fontweight='bold', color=self.accent_color, pad=10)
        
        # Select top spam hubs
        top_spammers = spam_df.nlargest(12, 'out_degree')['sender'].tolist()
        
        # Gather their receivers (limited for clarity)
        sample_nodes = set(top_spammers)
        for s in top_spammers:
            receivers = list(self.G.successors(s))[:8]
            sample_nodes.update(receivers)
        
        # Extract subgraph
        subG = self.G.subgraph(list(sample_nodes))
        
        # Graph layout
        pos = nx.spring_layout(subG, k=0.85, iterations=100, seed=42)
        
        # Color and size based on spam/legit
        node_colors = [
            self.spam_color if n in spam_df['sender'].values else self.legit_color
            for n in subG.nodes()
        ]
        
        node_sizes = [
            250 if n in spam_df['sender'].values else 120
            for n in subG.nodes()
        ]
        
        # Draw edges
        nx.draw_networkx_edges(
            subG, pos,
            ax=ax,
            arrows=True,
            arrowstyle='-|>',
            arrowsize=8,
            alpha=0.25,
            width=0.6,
            edge_color='#7f8c8d'
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            subG, pos,
            ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            linewidths=0.6,
            edgecolors='white'
        )
        
        ax.axis('off')
    
    def _plot_key_insights(self, ax):
        """Panel 8: Text panel with key insights."""
        ax.axis('off')
        ax.set_facecolor('white')
        
        insights_text = """
KEY FINDINGS

• High Out-Degree  
  Spammers contact significantly more recipients.

• One-Way Communication  
  Reciprocity is much lower among spam senders.

• Structural Isolation  
  Spammers show near-zero clustering and very few triangles.

• Peripheral Position  
  Low closeness and eigenvector centrality indicate weak integration.

Overall, graph-based metrics reveal clear structural
differences between spam and legitimate senders.
"""
        
        ax.text(
            0.05, 0.95, insights_text,
            fontsize=10.5, fontfamily='monospace',
            color=self.accent_color, va='top'
        )
    
    def create_flow_graph(self, save_path: Optional[str] = None) -> None:
        """
        Create communication flow graph showing Spam ↔ Non-Spam interactions.
        
        Args:
            save_path: Optional path to save the figure
        """
        print("Creating communication flow graph...")
        
        # Classify edges
        spam_to_spam = []
        spam_to_non = []
        non_to_spam = []
        non_to_non = []
        
        for u, v, data in self.G.edges(data=True):
            sender_spam = u in self.spammers
            receiver_spam = v in self.spammers
            
            if sender_spam and receiver_spam:
                spam_to_spam.append((u, v, data))
            elif sender_spam and not receiver_spam:
                spam_to_non.append((u, v, data))
            elif not sender_spam and receiver_spam:
                non_to_spam.append((u, v, data))
            else:
                non_to_non.append((u, v, data))
        
        # Build 2-node flow graph
        H = nx.DiGraph()
        
        H.add_node("Spam", color='#ff6b6b')
        H.add_node("Non-Spam", color='#51cf66')
        
        # Add weighted edges
        H.add_edge("Spam", "Spam", weight=len(spam_to_spam))
        H.add_edge("Spam", "Non-Spam", weight=len(spam_to_non))
        H.add_edge("Non-Spam", "Spam", weight=len(non_to_spam))
        H.add_edge("Non-Spam", "Non-Spam", weight=len(non_to_non))
        
        # Calculate percentages
        total_emails = len(spam_to_spam) + len(spam_to_non) + len(non_to_spam) + len(non_to_non)
        spam_to_spam_pct = (len(spam_to_spam) / total_emails) * 100
        spam_to_non_pct = (len(spam_to_non) / total_emails) * 100
        non_to_spam_pct = (len(non_to_spam) / total_emails) * 100
        non_to_non_pct = (len(non_to_non) / total_emails) * 100
        
        # Normalize edge thickness
        max_w = max(len(spam_to_spam), len(spam_to_non), len(non_to_spam), len(non_to_non))
        def scale(w): return 1 + (w / max_w) * 8
        
        pos = {
            "Spam": (-1, 0),
            "Non-Spam": (1, 0)
        }
        
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        ax.set_facecolor('#f8f9fa')
        
        # Draw nodes
        nx.draw_networkx_nodes(
            H, pos,
            node_color=['#ff6b6b', '#51cf66'],
            node_size=4000,
            alpha=0.95,
            linewidths=2,
            edgecolors=['#c92a2a', '#2b8a3e']
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            H, pos,
            font_size=16, 
            font_weight='bold',
            font_family='sans-serif'
        )
        
        # Draw edges
        edge_colors = {
            ("Spam", "Spam"): '#ff8787',
            ("Spam", "Non-Spam"): '#ff8787', 
            ("Non-Spam", "Spam"): '#69db7c',
            ("Non-Spam", "Non-Spam"): '#69db7c'
        }
        
        for (u, v, data) in H.edges(data=True):
            nx.draw_networkx_edges(
                H, pos,
                edgelist=[(u, v)],
                arrowstyle='-|>',
                arrowsize=20,
                width=scale(data['weight']),
                edge_color=edge_colors[(u, v)],
                alpha=0.8,
                connectionstyle='arc3,rad=0.25' if u == v else 'arc3,rad=0.1'
            )
        
        # Add edge labels
        edge_labels = {
            ("Spam", "Spam"): f"{len(spam_to_spam):,}\n({spam_to_spam_pct:.1f}%)",
            ("Spam", "Non-Spam"): f"{len(spam_to_non):,}\n({spam_to_non_pct:.1f}%)",
            ("Non-Spam", "Spam"): f"{len(non_to_spam):,}\n({non_to_spam_pct:.1f}%)", 
            ("Non-Spam", "Non-Spam"): f"{len(non_to_non):,}\n({non_to_non_pct:.1f}%)"
        }
        
        label_pos = {
            ("Spam", "Spam"): (-1.3, 0.3),
            ("Spam", "Non-Spam"): (0, 0.15),
            ("Non-Spam", "Spam"): (0, -0.15),
            ("Non-Spam", "Non-Spam"): (1.3, 0.3)
        }
        
        for edge, label in edge_labels.items():
            plt.annotate(label, 
                        xy=label_pos[edge], 
                        xytext=label_pos[edge],
                        ha='center', va='center',
                        fontsize=11,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'),
                        fontweight='bold')
        
        plt.title("Email Communication Flow: Spam vs Non-Spam Accounts", 
                  fontsize=16, fontweight='bold', pad=20, color='#2b2d42')
        plt.axis('off')
        plt.tight_layout()
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff6b6b', markersize=10, label='Spam Accounts'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#51cf66', markersize=10, label='Non-Spam Accounts'),
            plt.Line2D([0], [0], color='#ff8787', lw=3, label='Spam →'),
            plt.Line2D([0], [0], color='#69db7c', lw=3, label='Non-Spam →')
        ]
        plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1), 
                   ncol=2, frameon=True, fancybox=True, shadow=True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
            print(f"Flow graph saved to {save_path}")
        
        plt.show()
        
        # Print summary statistics
        print(f"\n{'='*60}")
        print("FLOW SUMMARY STATISTICS")
        print(f"{'='*60}")
        print(f"Spam → Spam:       {len(spam_to_spam):,} emails ({spam_to_spam_pct:.1f}%)")
        print(f"Spam → Non-Spam:   {len(spam_to_non):,} emails ({spam_to_non_pct:.1f}%)") 
        print(f"Non-Spam → Spam:   {len(non_to_spam):,} emails ({non_to_spam_pct:.1f}%)")
        print(f"Non-Spam → Non-Spam: {len(non_to_non):,} emails ({non_to_non_pct:.1f}%)")
        print(f"{'='*60}")
        print(f"Total emails:      {total_emails:,}")
        print("✓ Flow graph created successfully!\n")
    
    def create_all_figures(self, dashboard_path: str = 'spam_eda_dashboard.png',
                          flow_path: str = 'spam_flow_graph.png') -> None:
        """
        Create all visualizations.
        
        Args:
            dashboard_path: Path to save the dashboard figure
            flow_path: Path to save the flow graph figure
        """
        print("="*70)
        print(" CREATING ALL VISUALIZATIONS")
        print("="*70 + "\n")
        
        self.create_dashboard(dashboard_path)
        self.create_flow_graph(flow_path)
        
        print("="*70)
        print(" ALL VISUALIZATIONS CREATED SUCCESSFULLY!")
        print("="*70)


# Convenience function for quick usage
def visualize_email_graph(features_df: pd.DataFrame, G: nx.DiGraph, df: pd.DataFrame,
                          dashboard_path: str = 'spam_eda_dashboard.png',
                          flow_path: str = 'spam_flow_graph.png') -> None:
    """
    Quick function to create all visualizations.
    
    Args:
        features_df: DataFrame with extracted features
        G: NetworkX directed graph
        df: Original email dataframe
        dashboard_path: Path to save the dashboard figure
        flow_path: Path to save the flow graph figure
        
    Example:
        >>> from feature_extraction import EmailGraphFeatureExtractor
        >>> from visualization import visualize_email_graph
        >>> 
        >>> extractor = EmailGraphFeatureExtractor()
        >>> features_df = extractor.get_features("data.csv")
        >>> visualize_email_graph(features_df, extractor.G, extractor.df)
    """
    visualizer = EmailGraphVisualizer(features_df, G, df)
    visualizer.create_all_figures(dashboard_path, flow_path)


if __name__ == "__main__":
    print("Email Graph Visualization Module")
    print("="*70)
    print("\nUsage example:")
    print("  from feature_extraction import EmailGraphFeatureExtractor")
    print("  from visualization import EmailGraphVisualizer")
    print("")
    print("  extractor = EmailGraphFeatureExtractor()")
    print("  features_df = extractor.get_features('data.csv')")
    print("")
    print("  visualizer = EmailGraphVisualizer(features_df, extractor.G, extractor.df)")
    print("  visualizer.create_dashboard('dashboard.png')")
    print("  visualizer.create_flow_graph('flow.png')")
    print("")
    print("Or use the convenience function:")
    print("  from visualization import visualize_email_graph")
    print("  visualize_email_graph(features_df, extractor.G, extractor.df)")
