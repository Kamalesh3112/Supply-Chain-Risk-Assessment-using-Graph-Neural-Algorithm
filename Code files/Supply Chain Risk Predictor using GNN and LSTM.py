import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, GAE
from torch import nn
import networkx as nx
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ==================== DATA LOADING ====================

class RealDataLoader:
    """Load real DataCo supply chain CSV data"""

    @staticmethod
    def load_csv_data(csv_path, sample_size=10000):
        """Load CSV data with error handling"""
        print(f"Loading data from: {csv_path}")
        try:
            df = pd.read_csv(csv_path)
            print(f"✓ Loaded {len(df)} rows")
            if sample_size and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
                print(f"✓ Sampled to {sample_size} rows")
            return df
        except FileNotFoundError:
            print(f"File not found: {csv_path}")
            print("\nFor Colab: from google.colab import files; uploaded = files.upload()")
            print("For Local: csv_path = 'CoSupplyChainData.csv'")
            return None

# ------ Preprocessing real supply chain data & Feature Engineering --------#
class DataPreprocessor:
    """Preprocess real supply chain data"""

    def __init__(self):
        self.scalers = {}
        self.encoders = {}

    def engineer_features(self, df):
        """Create features from raw data"""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        # Binary indicators
        if 'Days for shipping (real)' in df.columns:
            df['high_lead_time'] = (
                df['Days for shipping (real)'] >
                df['Days for shipping (real)'].median()
            ).astype(int)

        if 'Late_delivery_risk' in df.columns:
            df['late_delivery'] = (
                pd.to_numeric(df['Late_delivery_risk'], errors='coerce')
                .fillna(0)
                .astype(int)
            )

        if 'Order Profit Per Order' in df.columns:
            df['low_profit'] = (
                df['Order Profit Per Order'] <
                df['Order Profit Per Order'].quantile(0.25)
            ).astype(int)

        if 'Order Item Discount' in df.columns:
            df['high_discount'] = (
                df['Order Item Discount'] >
                df['Order Item Discount'].quantile(0.75)
            ).astype(int)

        if 'Shipping Date' in df.columns and 'Order Date' in df.columns:
            df["shipping_delay_days"] = (
                pd.to_datetime(df["Shipping Date"], errors = "coerce") -
                pd.to_datetime(df["Order Date"], errors = "coerce")
            ).dt.days.clip(lower=0)

        if 'Delivery Status' in df.columns:
            df["delivery_delayed"] = (
                df["Delivery Status"]
                .astype(str)
                .map({"Late": 1, "On Time": 0})
                .fillna(0)
                .astype(int)
            )


        if 'Order Status' in df.columns:
            df["order_cancelled"] = df["Order Status"].str.contains(
                "CANCEL", case = False, na = False
            ).astype(int)

        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            df["lat_bin"] = pd.to_numeric(df["Latitude"], errors="coerce").round(1)
            df["lon_bin"] = pd.to_numeric(df["Longitude"], errors="coerce").round(1)



        # Location field
        if 'Customer Country' in df.columns:
            df['location'] = df['Customer Country']
        elif 'Order Region' in df.columns:
            df['location'] = df['Order Region']
        else:
            df['location'] = 'Unknown'

        return df

    def compute_risk_score(self, df, learned = False):
        """Compute composite risk score"""
        df = df.copy()
        risk_score = np.zeros(len(df))

        # Shipping delay component
        if 'Late_delivery_risk' in df.columns:
            late_del = (
                pd.to_numeric(df['Late_delivery_risk'], errors='coerce')
                .fillna(0)
                .values
            )
            risk_score += late_del * 0.25

        # Profit component
        if 'Order Profit Per Order' in df.columns:
            profit_norm = (
                (df['Order Profit Per Order'] - df['Order Profit Per Order'].min()) /
                (df['Order Profit Per Order'].max() - df['Order Profit Per Order'].min() + 1e-8)
            )
            risk_score += (1 - profit_norm) * 0.25

        # Lead time component
        if 'Days for shipping (real)' in df.columns:
            days_norm = (
                (df['Days for shipping (real)'] - df['Days for shipping (real)'].min()) /
                (df['Days for shipping (real)'].max() - df['Days for shipping (real)'].min() + 1e-8)
            )
            risk_score += days_norm * 0.20

        # Discount component
        if 'Order Item Discount' in df.columns:
            disc_norm = (
                (df['Order Item Discount'] - df['Order Item Discount'].min()) /
                (df['Order Item Discount'].max() - df['Order Item Discount'].min() + 1e-8)
            )
            risk_score += disc_norm * 0.15

        # Shipping mode component
        if 'Shipping Mode' in df.columns:
            shipping_risk = (df['Shipping Mode'].astype(str) != 'Standard Class').astype(int)
            risk_score += shipping_risk * 0.15

        if 'shipping_delay_days' in df.columns:
            delay_norm = (
                (df['shipping_delay_days'] - df['shipping_delay_days'].min()) /
                (df['shipping_delay_days'].max() - df['shipping_delay_days'].min() + 1e-8)
            )
            risk_score += delay_norm * 0.10


        df['risk_score'] = risk_score
        df['Risk_level'] = pd.qcut(       #Quantile-aware labeling
            df['risk_score'],
            q = [0, 0.6, 0.85, 1.0],
            labels=['Low', 'Medium', 'High']
        )

        return df

    def preprocess(self, df):
        """Full preprocessing pipeline"""
        df = df.copy()

        # Step 1: Feature engineering
        df = self.engineer_features(df)

        # Step 2: Risk score computation
        df = self.compute_risk_score(df)

        # --------------------------------------------------
        # Supplier-level & Region-level risk
        # --------------------------------------------------

        if 'Supplier_ID' in df.columns:
            df['supplier_risk_avg'] = (
                df.groupby('Supplier_ID')['risk_score']
                  .transform('mean')
            )
        else:
            df['supplier_risk_avg'] = df['risk_score']

        df['region_risk_density'] = (
            df.groupby('location')['risk_score']
              .transform('mean')
        )

        # --------------------------------------------------
        # Feature selection
        # --------------------------------------------------

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in ['risk_score']][:20]

        # Encode categorical
        for col in ['location', 'Order Status', 'Delivery Status', 'Shipping Mode']:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_enc'] = le.fit_transform(
                    df[col].astype(str).fillna('Unknown')
                )
                self.encoders[col] = le
                feature_cols.append(f'{col}_enc')

        feature_cols = feature_cols[:20]

        X = df[feature_cols].fillna(0).values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['scaler'] = scaler

        df['risk_encoded'] = pd.Categorical(
            df['Risk_level'],
            categories=['Low', 'Medium', 'High']
        ).codes

        y = df['risk_encoded'].values

        return df, X_scaled, y, feature_cols

#-----------------------Disruption Generation------------------------#
import random
import pandas as pd
import networkx as nx
import numpy as np
import torch


class DisruptionGenerator:
    def __init__(self, df):
        self.df = df

        # Map original DataFrame index → positional index (0..N-1)
        self.original_to_positional_map = {
            idx: i for i, idx in enumerate(df.index)
        }

        # Reverse map (positional → original index)
        self.positional_to_original_map = {
            i: idx for idx, i in self.original_to_positional_map.items()
        }

    # --------------------------------------------------
    # Generate disruption SEEDS
    # --------------------------------------------------
    def generate_disruptions(self, n=50):
        sampled = self.df.sample(n=min(n, len(self.df)), replace=False)

        disruptions = []
        for original_idx, row in sampled.iterrows():
            pos_idx = self.original_to_positional_map.get(original_idx)
            if pos_idx is None:
                continue

            disruptions.append({
                "supplier_id": int(pos_idx),
                "region": row.get("location", "Unknown"),
                "severity": round(
                    0.5 * float(row["risk_score"]) +
                    0.3 * float(row.get("shipping_delay_days", 0)) +
                    0.2 * random.random(),
                    2
                ),
                "type": random.choice(
                    ["natural", "geopolitical", "operational"]
                )
            })


        return pd.DataFrame(disruptions)

    # --------------------------------------------------
    # BASELINE CASCADE
    # --------------------------------------------------
    def cascade_impact(self, G, disruptions, cutoff=3):
        """
        Returns:
        - affected: number of suppliers impacted per disruption
        - score: weighted impact score (risk-weighted)
        """

        affected_counts = []
        impact_scores = []

        for d in disruptions.itertuples(index=False):
            start_node = int(d.supplier_id)

            if start_node not in G.nodes():
                continue

            # -------- GRAPH SPREAD (TOPOLOGY) --------
            affected_nodes = nx.single_source_shortest_path_length(
                G,
                start_node,
                cutoff=cutoff
            )

            affected_counts.append(len(affected_nodes))

            # -------- IMPACT SCORE (RISK-WEIGHTED) --------
            impact = 0.0
            for pos_node in affected_nodes:
                original_idx = self.positional_to_original_map.get(pos_node)
                if original_idx is None:
                    continue
                if original_idx in self.df.index:
                    impact += float(self.df.loc[original_idx]["risk_score"])

            impact_scores.append(round(impact, 3))

        # ---------------- Debug samples ----------------
        print("[DEBUG] affected sample:", affected_counts[:5])
        print("[DEBUG] impact sample:", impact_scores[:5])

        return {
            "affected": affected_counts,
            "score": impact_scores
        }


# =========================================================
# TEMPORAL EXTENSIONS
# =========================================================

def build_temporal_snapshots(df, T=6, noise_std=0.05):
    """
    Generate temporal snapshots of node features
    by slightly perturbing risk_score over time.

    Returns:
    - list[pd.DataFrame] of length T
    """

    snapshots = []
    base = df.copy()

    for t in range(T):
        df_t = base.copy()

        if "risk_score" in df_t.columns:
            df_t["risk_score"] = (
                df_t["risk_score"]
                + np.random.normal(0, noise_std, len(df_t))
            ).clip(0, 1)

        df_t["time"] = t
        snapshots.append(df_t)

    return snapshots


def temporal_pyg_data(G, snapshots, feature_cols):
    """
    Convert temporal snapshots into PyTorch tensors
    suitable for Temporal GNN / LSTM models.

    Returns:
    - X_seq: Tensor [T, N, F]
    - edge_index: Tensor [2, E]
    """

    X_seq = []

    for df_t in snapshots:
        X_seq.append(
            torch.tensor(
                df_t[feature_cols].values,
                dtype=torch.float32
            )
        )

    X_seq = torch.stack(X_seq)  # [T, N, F]

    # Safe edge_index construction
    if G.number_of_edges() > 0:
        edge_index = torch.tensor(
            np.array(list(G.edges())).T,
            dtype=torch.long
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return X_seq, edge_index

# ==================== FAIRNESS ANALYSIS ====================
class FairnessAnalysis:
    """Fairness and equity analysis"""

    @staticmethod
    def analyze(df, predictions):
        """Analyze fairness by location"""
        fairness = {}
        for loc in df['location'].unique():
            mask = df['location'] == loc
            preds = predictions[mask]
            high_risk = (preds == 2).sum() / len(preds) if len(preds) > 0 else 0
            fairness[str(loc)] = {
                'high_risk_rate': high_risk,
                'count': len(preds)
            }
        return fairness

# ==================== GRAPH CONSTRUCTION ====================
class GraphBuilder:
    """Build multi-layer supply chain graphs"""

    def __init__(self, df, threshold=0.75):
        self.df = df
        self.threshold = threshold

    def build_supplier_layer(self, X):
        """Layer 1: Supplier relationships"""
        from sklearn.metrics.pairwise import cosine_similarity
        G = nx.Graph()
        n = min(len(self.df), len(X))
        for i in range(n):
            G.add_node(i)

        sim = cosine_similarity(X[:n])
        for i in range(n):
            for j in np.where(sim[i] > self.threshold)[0]:
                if i < j:
                    G.add_edge(i, j)
        return G

    def build_product_layer(self):
        """Layer 2: Product categories"""
        G = nx.Graph()
        n = min(len(self.df), 2000)
        for i in range(n):
            G.add_node(i)

        for col in ['Product Category Id', 'Category Id']:
            if col in self.df.columns:
                df_sub = self.df.iloc[:n]
                groups = df_sub.groupby(col).groups
                edges = []
                for indices in groups.values():
                    idx_list = list(indices)
                    for i in range(len(idx_list)):
                        for j in range(i+1, min(i+8, len(idx_list))):
                            edges.append((idx_list[i], idx_list[j]))
                G.add_edges_from(edges)
                break
        return G

    def build_region_layer(self):
        """Layer 3: Geographic regions"""
        G = nx.Graph()
        n = min(len(self.df), 2000)
        for i in range(n):
            G.add_node(i)

        df_sub = self.df.iloc[:n]
        if 'location' in df_sub.columns:
            groups = df_sub.groupby('location').groups
            edges = []
            for indices in groups.values():
                idx_list = list(indices)
                for i in range(len(idx_list)):
                    for j in range(i+1, min(i+6, len(idx_list))):
                        edges.append((idx_list[i], idx_list[j]))
            G.add_edges_from(edges)
        return G

    def build_all(self, X):
        """Build all layers with enriched supplier connectivity"""

        # Base layers
        G_s = self.build_supplier_layer(X)
        G_p = self.build_product_layer()
        G_r = self.build_region_layer()

        df = self.df

        # --------------------------------------------------
        #  1. Region-based supplier links
        # --------------------------------------------------
        if "Order Region" in df.columns:
            for _, group in df.groupby("Order Region"):
                suppliers = group.index.tolist()
                for i in range(len(suppliers) - 1):
                    G_s.add_edge(
                        suppliers[i],
                        suppliers[i + 1],
                        weight=0.5
                    )

        # --------------------------------------------------
        #  2. Department-level similarity
        # --------------------------------------------------
        if "Department Name" in df.columns:
            for _, group in df.groupby("Department Name"):
                suppliers = group.index.tolist()
                for i in range(len(suppliers) - 1):
                    G_s.add_edge(
                        suppliers[i],
                        suppliers[i + 1],
                        weight=0.3
                    )

        # --------------------------------------------------
        # 3. Market-level risk proximity
        # --------------------------------------------------
        if "Market" in df.columns:
            for _, group in df.groupby("Market"):
                suppliers = group.index.tolist()
                for i in range(len(suppliers) - 1):
                    G_s.add_edge(
                        suppliers[i],
                        suppliers[i + 1],
                        weight=0.2
                    )

        return G_s, G_p, G_r

    def get_centrality(self, G):
        """Get centrality measures"""
        if G.number_of_nodes() == 0:
            return {}, {}, {}
        return nx.degree_centrality(G), nx.betweenness_centrality(G), nx.closeness_centrality(G)

#---------------Model Construction and Initiailization------------------------
class GCN(torch.nn.Module):
    """Graph Convolutional Network"""
    def __init__(self, in_dim, hidden=64, out=3, dropout=0.5):
        super().__init__()
        self.gc1 = GCNConv(in_dim, hidden)
        self.gc2 = GCNConv(hidden, hidden)
        self.gc3 = GCNConv(hidden, out)
        self.dropout = dropout

    def forward(self, x, edge_idx):
        x = self.gc1(x, edge_idx)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc2(x, edge_idx)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.gc3(x, edge_idx)

class GAT(torch.nn.Module):
    """Graph Attention Network"""
    def __init__(self, in_dim, hidden=64, out=3, heads=8, dropout=0.5):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden*heads, hidden, heads=1, dropout=dropout)
        self.gat3 = GATConv(hidden, out, heads=1, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_idx):
        x = self.gat1(x, edge_idx)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_idx)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.gat3(x, edge_idx)

class TemporalGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.gnn = GCNConv(input_dim, hidden_dim)
        self.lstm = torch.nn.LSTM(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, 3)

    def forward(self, X_seq, edge_index):
        """
        X_seq: [T, N, F] or [N, F] if a single snapshot
        """
        # Adapt X_seq if it's a single snapshot [N, F] to [1, N, F]
        if X_seq.ndim == 2:
            X_seq = X_seq.unsqueeze(0) # Becomes [1, N, F]

        gnn_out = []

        for t in range(X_seq.shape[0]): # X_seq.shape[0] will be T (or 1)
            h = self.gnn(X_seq[t], edge_index) # X_seq[t] will be [N, F]
            gnn_out.append(h)

        gnn_out = torch.stack(gnn_out)  # [T, N, H]

        lstm_out, _ = self.lstm(gnn_out)
        out = self.fc(lstm_out[-1])
        return out


class Ensemble(torch.nn.Module):
    """Ensemble of all models"""
    def __init__(self, in_dim, hidden=64, out=3, dropout=0.5): # Removed 'steps' as it's not used by TemporalGNN in its current form
        super().__init__()
        self.gcn = GCN(in_dim, hidden, out, dropout)
        self.gat = GAT(in_dim, hidden, out, 4, dropout)
        self.temporal = TemporalGNN(in_dim, hidden)
        self.fuse = nn.Linear(out*3, out)

    def forward(self, x, edge_idx):
        g = self.gcn(x, edge_idx)
        a = self.gat(x, edge_idx)
        t = self.temporal(x, edge_idx)
        return self.fuse(torch.cat([g, a, t], dim=1))

# ==================== TRAINING ====================
class Trainer:
    """Model trainer"""
    def __init__(self, device: torch.device, class_weights = None):
        self.device = device
        self.train_loss = []
        self.val_loss = []


        if class_weights is None:
          self.class_weights = torch.tensor([1.0, 0.7, 1.5], device = device)

    def train_step(self, model, opt, data):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask], weight = self.class_weights)
        loss.backward()
        opt.step()
        return loss.item()


    def eval_step(self, model, data):
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out[data.test_mask], data.y[data.test_mask], weight = self.class_weights)
            pred = out.argmax(dim=1)
        return loss.item(), out

    def train(self, model, data, epochs=100, lr=0.01):
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        for ep in range(epochs):
            tl = self.train_step(model, opt, data)
            vl, _ = self.eval_step(model, data)
            self.train_loss.append(tl)
            self.val_loss.append(vl)
            if (ep+1) % 20 == 0:
                print(f'Epoch {ep+1}/{epochs} - Train: {tl:.4f}, Val: {vl:.4f}')
        return model

#True Cascade Propagation: Affected Suppliers (Entity-Level Tracking)
class CascadeEngine:
    def __init__(self, graph, decay=0.7, max_depth=3):
        """
        Parameters
        ----------
        graph : networkx.Graph or DiGraph
            Supply chain graph (nodes = suppliers, edges = dependencies)
        decay : float
            Decay factor applied per hop
        max_depth : int
            Maximum propagation depth
        """
        self.graph = graph
        self.decay = decay
        self.max_depth = max_depth

    # --------------------------------------------------
    # Scenario-specific policy rules
    # --------------------------------------------------
    def _apply_rules(self, disruption, prob):
        """
        Modify propagation probability based on disruption type.
        """
        dtype = disruption.get("type", "operational")

        if dtype == "geopolitical":
            return prob * 0.6

        elif dtype == "natural":
            return prob * 1.3
        return prob

    # --------------------------------------------------
    # Cascade execution
    # --------------------------------------------------
    def run(self, disruption, scenario_id=0, t0=0):
        """
        Run a single cascade simulation.

        Parameters
        ----------
        disruption : dict
            {
                "supplier_id": int,
                "severity": float,
                "type": str
            }
        scenario_id : int
            Scenario identifier
        t0 : int
            Start time offset (for temporal experiments)

        Returns
        -------
        dict
            Cascade result with affected suppliers and paths
        """

        src = disruption["supplier_id"]
        severity = disruption["severity"]

        affected = {src}
        paths = {src: [src]}
        frontier = [src]
        impact = 0.0

        # Temporal + depth-aware propagation
        for t, depth in enumerate(range(t0, t0 + self.max_depth)):
            next_frontier = []

            for node in frontier:
                for nbr in self.graph.neighbors(node):
                    if nbr not in affected:
                        # Base probability with decay
                        base_prob = severity * (self.decay ** depth)

                        # Apply scenario-specific rules
                        prob = self._apply_rules(disruption, base_prob)

                        if np.random.rand() < prob:
                            affected.add(nbr)
                            paths[nbr] = paths[node] + [nbr]
                            next_frontier.append(nbr)
                            impact += prob

            frontier = next_frontier

            # Early stop if cascade dies out
            if not frontier:
                break

        return {
            "scenario_id": scenario_id,
            "source_supplier": src,
            "affected_suppliers": affected,
            "propagation_paths": paths,
            "final_impact": impact
        }

# ==================== VISUALIZATION ====================
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# ==================== VISUALIZATION ====================

class Viz:
    """Visualization functions"""

    # --------------------------------------------------
    # Risk distribution
    # --------------------------------------------------
    @staticmethod
    def plot_risk(df):
        counts = df['Risk_level'].value_counts()
        colors = {'Low': '#00cc96', 'Medium': '#ffa15a', 'High': '#ef553b'}

        fig = go.Figure([
            go.Bar(
                x=counts.index,
                y=counts.values,
                marker_color=[colors[x] for x in counts.index]
            )
        ])
        fig.update_layout(
            title='Risk Distribution',
            xaxis_title='Risk Level',
            yaxis_title='Count'
        )
        fig.show()

    # --------------------------------------------------
    # Original cascade
    # --------------------------------------------------
    @staticmethod
    def plot_cascade(df):
        if df is None or len(df) == 0:
            print("⚠ Empty cascade data")
            return

        x = list(range(len(df["affected"])))

        fig = go.Figure()

        # Affected suppliers (LEFT axis)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df["affected"],
                name="Affected Suppliers",
                mode="lines+markers",
                line=dict(color="#ef553b", width=2),
                yaxis="y1"
            )
        )

        # Impact score (RIGHT axis)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df["score"],
                name="Impact Score",
                mode="lines+markers",
                line=dict(color="#636EFA", width=2),
                yaxis="y2"
            )
        )

        fig.update_layout(
            title="Disruption Cascade",
            xaxis=dict(title="Disruption Scenario"),

            # Left Y-axis
            yaxis=dict(
                title="Affected Suppliers",
                showgrid=True
            ),

            # Right Y-axis
            yaxis2=dict(
                title="Impact Score",
                overlaying="y",
                side="right",
                showgrid=False
            ),

            legend=dict(x=1.02, y=1),
            height=600
        )

        fig.show()


    @staticmethod
    def plot_cascade_graph(cascade, max_nodes=300):
        """
        Visualize a single cascade result clearly using Plotly
        """

        if not cascade or len(cascade.get("affected_suppliers", [])) == 0:
            print("⚠ Empty cascade graph")
            return

        # ---------------------------
        # REQUIRED INPUT
        # ---------------------------
        source_node = cascade["source_supplier"]
        affected_nodes = set(cascade["affected_suppliers"])
        propagation_paths = cascade.get("propagation_paths", {})

        # ---------------------------
        # Build graph from cascade
        # ---------------------------
        G = nx.DiGraph()

        for path in propagation_paths.values():
            for i in range(len(path) - 1):
                G.add_edge(path[i], path[i + 1])

        # Safety cap (visual clarity)
        if G.number_of_nodes() > max_nodes:
            G = G.subgraph(list(G.nodes())[:max_nodes]).copy()

        pos = nx.spring_layout(G, seed=42, k=0.35)

        node_x, node_y = [], []
        node_color, node_size = [], []
        hover_text = []

        # ---------------------------
        # Node styling
        # ---------------------------
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            if node == source_node:
                node_color.append("#EF553B")   # red
                node_size.append(22)
                hover_text.append(f"Source Supplier {node}")
            elif node in affected_nodes:
                node_color.append("#636EFA")   # blue
                node_size.append(14)
                hover_text.append(f"Affected Supplier {node}")
            else:
                node_color.append("#B0BEC5")   # grey
                node_size.append(8)
                hover_text.append(f"Supplier {node}")

        # ---------------------------
        # Edge trace
        # ---------------------------
        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=0.6, color="#BDBDBD"),
            hoverinfo="none"
        )

        # ---------------------------
        # Node trace
        # ---------------------------
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            hovertext=hover_text,
            hoverinfo="text",
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=0.5, color="#333")
            )
        )

        # ---------------------------
        # Plot
        # ---------------------------
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title="Disruption Cascade Graph (Plotly)",
            showlegend=False,
            height=750,
            hovermode="closest"
        )

        fig.show()

    # --------------------------------------------------
    # Fairness
    # --------------------------------------------------
    @staticmethod
    def plot_fairness(fair):
        locs = list(fair.keys())
        rates = [fair[l]['high_risk_rate'] for l in locs]

        fig = go.Figure([go.Bar(x=locs, y=rates)])
        fig.update_layout(
            title='Fairness by Location',
            xaxis_title='Location',
            yaxis_title='High Risk Rate'
        )
        fig.show()

    @staticmethod
    def plot_disruption_cascade(affected_suppliers, impact_scores):
        """
        Time-series view of cascade severity across scenarios.
        """
        import plotly.graph_objects as go

        scenarios = list(range(1, len(affected_suppliers) + 1))

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=scenarios,
            y=affected_suppliers,
            mode='lines+markers',
            name='Affected Suppliers',
            yaxis='y1'
        ))

        fig.add_trace(go.Scatter(
            x=scenarios,
            y=impact_scores,
            mode='lines+markers',
            name='Impact Score',
            yaxis='y2'
        ))

        fig.update_layout(
            title='Disruption Cascade',
            xaxis=dict(title='Disruption Scenario'),
            yaxis=dict(title='Affected Suppliers'),
            yaxis2=dict(
                title='Impact Score',
                overlaying='y',
                side='right'
            ),
            legend=dict(x=1.05, y=1),
            template='plotly_white'
        )

        fig.show()

    # --------------------------------------------------
    # Model comparison
    # --------------------------------------------------
    @staticmethod
    def plot_models(metrics):
        fig = go.Figure()
        models = list(metrics.keys())

        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            fig.add_trace(go.Bar(
                name=metric,
                x=models,
                y=[metrics[m].get(metric, 0) for m in models]
            ))

        fig.update_layout(
            title='Model Comparison',
            barmode='group'
        )
        fig.show()

    # --------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------
    @staticmethod
    def plot_confusion(y_true, y_pred, name):
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)
        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=['Low', 'Medium', 'High'],
                y=['Low', 'Medium', 'High'],
                colorscale='Blues'
            )
        )
        fig.update_layout(title=f'Confusion Matrix - {name}')
        fig.show()

    # --------------------------------------------------
    # SUPPLY CHAIN NETWORK
    # --------------------------------------------------
    @staticmethod
    def plot_network(G, df, pred, title='Supply Chain Network'):
        """
        CLEAN network plot:
        - No node labels
        - No edge labels
        - Hover-only info
        - Scales to large graphs
        """

        pos = nx.spring_layout(G, k=0.4, seed=42)

        # ---------- Edges ----------
        ex, ey = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            ex.extend([x0, x1, None])
            ey.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=ex,
            y=ey,
            mode='lines',
            line=dict(width=0.4, color='rgba(150,150,150,0.35)'),
            hoverinfo='none',
            showlegend=False
        )

        # ---------- Nodes ----------
        nx_list, ny_list, node_colors, hover_text = [], [], [], []
        colors = {0: '#00cc96', 1: '#ffa15a', 2: '#ef553b'}

        for n in G.nodes():
            if n < len(df):
                x, y = pos[n]
                nx_list.append(x)
                ny_list.append(y)
                node_colors.append(colors[int(pred[n])])
                hover_text.append(f"Supplier_ID: {n}")

        node_trace = go.Scatter(
            x=nx_list,
            y=ny_list,
            mode='markers',
            marker=dict(
                size=6,
                color=node_colors,
                opacity=0.85
            ),
            hoverinfo='text',
            text=hover_text,
            showlegend=False
        )

        fig = go.Figure([edge_trace, node_trace])

        fig.update_layout(
            title=title,
            hovermode='closest',
            showlegend=False,
            height=700,
            margin=dict(l=20, r=20, t=60, b=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            annotations=[]
        )

        fig.show()

    # --------------------------------------------------
    # Cluster overview
    # --------------------------------------------------
    @staticmethod
    def plot_cluster_overview(df_clusters):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_clusters['Num_Suppliers'],
            y=df_clusters['Resilience_Score'],
            mode='markers+text',
            marker=dict(
                size=df_clusters['Num_Suppliers'] / 6,
                color=df_clusters['Resilience_Score'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Resilience Score'),
                opacity=0.85
            ),
            text=df_clusters['Cluster_ID'],
            textposition='top center'
        ))

        fig.update_layout(
            title='Cluster Overview',
            xaxis_title='Number of Suppliers',
            yaxis_title='Resilience Score',
            height=500,
            showlegend=False
        )
        fig.show()

    # --------------------------------------------------
    # Cascade comparison
    # --------------------------------------------------
    @staticmethod
    def plot_cascade_comparison(cascade_results):
        baseline = cascade_results['Baseline_Cascade']['Mean_Nodes']
        isolated = cascade_results['Isolated_Cascade']['Mean_Nodes']

        if baseline == 0 and isolated == 0:
            print("⚠ Cascade comparison skipped (zero impact)")
            return

        fig = go.Figure(go.Bar(
            x=['Baseline', 'Isolated'],
            y=[baseline, isolated],
            marker_color=['#ef553b', '#00cc96']
        ))

        fig.update_layout(
            title='Cascade Impact Reduction (Mean Affected Nodes)',
            yaxis_title='Mean Affected Nodes'
        )

        fig.show()

#-----------Improved Centrality Calculation With Normalization---------------
import numpy as np
import pandas as pd
import networkx as nx
from typing import cast
from sklearn.preprocessing import MinMaxScaler
from collections.abc import Hashable

def identify_critical_hubs_enhanced(G, top_n=50, use_weighted=False):
    try:
        weight_kw = "weight" if use_weighted else None

        degree: dict[Hashable, float] = nx.degree_centrality(G)
        betweenness: dict[Hashable, float] = nx.betweenness_centrality(G, weight=weight_kw)

        # Connectivity-aware closeness
        if G.is_directed():
            comps = nx.weakly_connected_components(G)  # directed-safe choice [web:4]
        else:
            comps = nx.connected_components(G)          # undirected [web:2]

        largest_cc = max(comps, key=len)
        G_cc = G.subgraph(largest_cc).copy()
        closeness: dict[Hashable, float] = nx.closeness_centrality(G_cc)

        # fill nodes outside component
        closeness = {n: float(closeness.get(n, 0.0)) for n in G.nodes()}

        try:
            eigen = nx.eigenvector_centrality(G, max_iter=2000, tol=1e-06)
        except nx.PowerIterationFailedConvergence:
            eigen = nx.eigenvector_centrality_numpy(G)

        pagerank: dict[Hashable, float] = nx.pagerank(G, weight=weight_kw)
        load = cast(dict[Hashable, float], nx.load_centrality(G))

        # --- Assemble feature matrix in a stable node order ---
        nodes = list(G.nodes())
        scores = np.array([
            [degree.get(n, 0.0),
             betweenness.get(n, 0.0),
             closeness.get(n, 0.0),
             eigen.get(n, 0.0),
             pagerank.get(n, 0.0),
             load.get(n, 0.0)]
            for n in nodes
        ], dtype=float)

        # --- Robust min-max scaling (guard constant columns) ---
        col_min = scores.min(axis=0)
        col_max = scores.max(axis=0)
        denom = np.where((col_max - col_min) == 0, 1.0, (col_max - col_min))
        normalized = (scores - col_min) / denom

        # --- Composite score ---
        weights = np.array([0.20, 0.35, 0.15, 0.15, 0.10, 0.05], dtype=float)
        crit_values = normalized @ weights
        critical_score = dict(zip(nodes, crit_values))

        # --- Output table ---
        sorted_hubs = sorted(critical_score.items(), key=lambda x: x[1], reverse=True)[:top_n]
        df_hubs = pd.DataFrame(sorted_hubs, columns=["Supplier_ID", "Criticality_Score"])

        df_hubs["Degree"] = df_hubs["Supplier_ID"].map(lambda n: G.degree(n))
        df_hubs["Betweenness"] = df_hubs["Supplier_ID"].map(lambda n: betweenness.get(n, 0.0))
        df_hubs["PageRank"] = df_hubs["Supplier_ID"].map(lambda n: pagerank.get(n, 0.0))
        df_hubs["Closeness"] = df_hubs["Supplier_ID"].map(lambda n: closeness.get(n, 0.0))

        p75 = df_hubs["Criticality_Score"].quantile(0.75)
        p50 = df_hubs["Criticality_Score"].quantile(0.50)
        df_hubs["Risk_Level"] = df_hubs["Criticality_Score"].apply(
            lambda x: "CRITICAL" if x > p75 else "HIGH" if x > p50 else "MEDIUM"
        )

        return df_hubs, critical_score

    except Exception as e:
        print(f"Error in centrality calculation: {e}")
        return pd.DataFrame(), {}

#------------------Advanced Community Detection with Resolution Parameter-------
from typing import TypedDict, List, Dict
from collections import defaultdict
from networkx.algorithms import community
import numpy as np
import pandas as pd
import networkx as nx


# --------------------------------------------------
# Schema for cluster statistics
# --------------------------------------------------
class ClusterStats(TypedDict):
    nodes: List[int]
    internal_edges: int
    external_edges: int
    density: float
    avg_degree: float


def create_resilient_clusters_enhanced(G, resolution: float = 1.0, seed: int = 42):
    """
    ENHANCED: Better clustering with resolution tuning.

    Fixes included:
    - Strict typing via TypedDict (no Pylance errors)
    - Integer-safe graph math
    - Stable Louvain / fallback clustering
    """

    np.random.seed(seed)

    # -----------------------------
    # Community detection
    # -----------------------------
    try:
        communities_list = list(
            community.louvain_communities(
                G,
                seed=seed,
                weight='weight' if nx.get_edge_attributes(G, 'weight') else None
            )
        )
    except Exception:
        communities_list = list(
            community.greedy_modularity_communities(G)
        )

    # -----------------------------
    # Node → cluster mapping
    # -----------------------------
    node_to_cluster: Dict[int, int] = {}

    for cid, comm in enumerate(communities_list):
        for node in comm:
            node_to_cluster[node] = cid

    # -----------------------------
    # Cluster statistics (STRICT + SAFE)
    # -----------------------------
    cluster_stats: Dict[int, ClusterStats] = defaultdict(
        lambda: {
            'nodes': [],
            'internal_edges': 0,
            'external_edges': 0,
            'density': 0.0,
            'avg_degree': 0.0
        }
    )

    # -----------------------------
    # Compute metrics
    # -----------------------------
    for cid in range(len(communities_list)):
        nodes_in_cluster: List[int] = [
            n for n, c in node_to_cluster.items() if c == cid
        ]

        cluster_stats[cid]['nodes'] = nodes_in_cluster

        internal = 0
        external = 0

        for u, v in G.edges():
            cu = node_to_cluster[u]
            cv = node_to_cluster[v]

            if cu == cv == cid:
                internal += 1
            elif cu == cid or cv == cid:
                external += 1

        cluster_stats[cid]['internal_edges'] = internal
        cluster_stats[cid]['external_edges'] = external

        n_nodes = len(nodes_in_cluster)
        max_edges = n_nodes * (n_nodes - 1) // 2  # INT-safe

        if max_edges > 0:
            cluster_stats[cid]['density'] = internal / max_edges

        if n_nodes > 0:
            cluster_stats[cid]['avg_degree'] = (
                sum(G.degree(n) for n in nodes_in_cluster) / n_nodes
            )

    # -----------------------------
    # DataFrame output
    # -----------------------------
    df_clusters = pd.DataFrame(
        [
            {
                'Cluster_ID': cid,
                'Num_Suppliers': len(stats['nodes']),
                'Internal_Edges': stats['internal_edges'],
                'External_Edges': stats['external_edges'],
                'Avg_Connectivity': stats['avg_degree'],
                'Cluster_Density': stats['density'],
                'Isolation_Score': stats['internal_edges']
                    / (stats['internal_edges'] + stats['external_edges'] + 1),
                'Resilience_Score': stats['density'] * 100
            }
            for cid, stats in cluster_stats.items()
        ]
    ).sort_values('Resilience_Score', ascending=False)

    # -----------------------------
    # Modularity score
    # -----------------------------
    modularity = community.modularity(G, communities_list)

    return df_clusters, node_to_cluster, cluster_stats, modularity

#------------------Improved Inter-Cluster Bottleneck Detection-------------------
def identify_critical_bottlenecks(G, node_to_cluster):
    """
    ENHANCED: Identify bottleneck suppliers with detailed criticality metrics.

    Improvements:
    - Ranks bridges by criticality (not just edge count)
    - Identifies bottleneck clusters
    - Calculates disruption radius
    """

    bridge_nodes = []

    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        neighbor_clusters = set(node_to_cluster[n] for n in neighbors)

        if len(neighbor_clusters) > 1:  # Connects multiple clusters
            cross_cluster_edges = sum(
                1 for n in neighbors if node_to_cluster[n] != node_to_cluster[node]
            )

            # Calculate disruption impact (how many clusters affected if this node fails)
            disruption_radius = len(neighbor_clusters)

            bridge_criticality = cross_cluster_edges * disruption_radius

            bridge_nodes.append({
                'Supplier_ID': node,
                'Home_Cluster': node_to_cluster[node],
                'Connected_Clusters': len(neighbor_clusters),
                'Cross_Cluster_Edges': cross_cluster_edges,
                'Disruption_Radius': disruption_radius,
                'Bridge_Criticality': bridge_criticality,
                'Risk_Type': 'BRIDGE_BOTTLENECK',
                'Severity': 'CRITICAL' if bridge_criticality > 6
                          else 'HIGH' if bridge_criticality > 3
                          else 'MEDIUM'
            })

    df_bridges = pd.DataFrame(bridge_nodes).sort_values(
        'Bridge_Criticality', ascending=False
    )

    return df_bridges

#-----------Cascade propagation---------------------
def cascade_propagation(
    G,
    start_node,
    model="bfs",
    severity_map=None,
    max_hops=3,
    base_prob=0.35
):
    visited = set([start_node])
    frontier = [(start_node, 0)]
    impact_score = 0.0

    while frontier:
        node, depth = frontier.pop(0)

        if depth >= max_hops:
            continue

        for nbr in G.neighbors(node):
            if nbr in visited:
                continue

            # -------------------------------
            # Propagation control
            # -------------------------------
            propagate = False

            if model == "bfs":
                propagate = True

            elif model == "probabilistic":
                propagate = np.random.rand() < base_prob

            elif model == "degree_weighted":
                deg = G.degree(nbr)
                propagate = np.random.rand() < min(0.1 + deg / 100, 0.8)

            if not propagate:
                continue

            visited.add(nbr)
            frontier.append((nbr, depth + 1))

            # severity weighting
            if severity_map and nbr in severity_map:
                impact_score += severity_map[nbr]
            else:
                impact_score += 1.0

    return visited, impact_score

#-------------Creating isolated graph------------------
import networkx as nx

def create_isolated_graph(G, node_to_cluster, disrupted_node):
    """
    Creates a graph where cross-cluster edges from the disrupted node's cluster
    are removed (cluster isolation).
    """

    if disrupted_node not in node_to_cluster:
        return G.copy()

    disrupted_cluster = node_to_cluster[disrupted_node]
    G_iso = G.copy()

    for u, v in list(G_iso.edges()):
        if (
            node_to_cluster.get(u) == disrupted_cluster and
            node_to_cluster.get(v) != disrupted_cluster
        ) or (
            node_to_cluster.get(v) == disrupted_cluster and
            node_to_cluster.get(u) != disrupted_cluster
        ):
            G_iso.remove_edge(u, v)

    return G_iso

#-------------------Improved Cascade Analysis with Propagation Modeling-------------
def improved_cascade_analysis_v2(
    G,
    node_to_cluster,
    sample_size=100,
    isolation_enabled=True,
    propagation_model="bfs",
    disrupted_nodes_list=None,
    severity_map=None
):
    """
    ENHANCED cascade analysis with disruption seeding.
    """

    affected_nodes_baseline = []
    affected_nodes_isolated = []
    impact_scores_baseline = []
    impact_scores_isolated = []

    # -------- SAMPLE DISRUPTED NODES --------
    if disrupted_nodes_list:
        sample_nodes = [n for n in disrupted_nodes_list if n in G.nodes()]
    else:
        sample_nodes = list(G.nodes())

    sample_nodes = np.random.choice(
        sample_nodes,
        size=min(sample_size, len(sample_nodes)),
        replace=False
    )

    # -------- CASCADE SIMULATION --------
    for disrupted_node in sample_nodes:

        # BASELINE
        affected_base, impact_base = cascade_propagation(
            G,
            disrupted_node,
            model=propagation_model,
            severity_map=severity_map
        )

        affected_nodes_baseline.append(len(affected_base))
        impact_scores_baseline.append(impact_base)

        # ISOLATED
        if isolation_enabled:
            G_iso = create_isolated_graph(G, node_to_cluster, disrupted_node)
            affected_iso, impact_iso = cascade_propagation(
                G_iso,
                disrupted_node,
                model=propagation_model,
                severity_map=severity_map
            )

            affected_nodes_isolated.append(len(affected_iso))
            impact_scores_isolated.append(impact_iso)

    # -------- METRICS --------
    mean_base = np.mean(affected_nodes_baseline)
    mean_iso = np.mean(affected_nodes_isolated) if affected_nodes_isolated else 0

    reduction = (
        (1 - mean_iso / mean_base) * 100
        if isolation_enabled and mean_base > 0
        else 0
    )

    return {
        "Baseline_Cascade": {
            "Mean_Nodes": mean_base,
            "Max_Nodes": max(affected_nodes_baseline, default=0),
            "Mean_Impact": np.mean(impact_scores_baseline)
        },
        "Isolated_Cascade": {
            "Mean_Nodes": mean_iso if isolation_enabled else None,
            "Max_Nodes": max(affected_nodes_isolated, default=0),
            "Mean_Impact": np.mean(impact_scores_isolated) if impact_scores_isolated else 0
        },
        "Reduction_Percentage": reduction
    }

#-----------------Intelligent Backup supplier Recommendation--------------------
def recommend_backup_suppliers_enhanced(G, node_to_cluster, critical_hubs,
                                        num_backups=2, geographic_diversity=True):
    """
    ENHANCED: Smarter backup recommendations.

    Improvements:
    - Geographic/regional diversity
    - Similarity scoring (shared customers, capacity)
    - Financial stability consideration
    - Availability filtering
    """

    backup_recommendations = []

    for _, hub_row in critical_hubs.head(50).iterrows():
        hub_id = hub_row['Supplier_ID']
        hub_cluster = node_to_cluster[hub_id]
        hub_degree = G.degree(hub_id)

        # Find candidates from OTHER clusters (diversification)
        candidates = []
        for other_node in G.nodes():
            if (node_to_cluster[other_node] != hub_cluster and other_node != hub_id):
                # Similarity scoring
                shared = len(set(G.neighbors(hub_id)) & set(G.neighbors(other_node)))
                degree_match = G.degree(other_node)

                # Score = shared customers + degree similarity
                similarity_score = (
                    (shared / max(G.degree(hub_id), 1)) * 0.6 +
                    (min(degree_match, hub_degree) / max(degree_match, hub_degree)) * 0.4
                )

                candidates.append((other_node, similarity_score, node_to_cluster[other_node]))

        candidates.sort(key=lambda x: x[1], reverse=True)

        # Select backups with cluster diversity
        selected_backups = []
        selected_clusters = set()

        for candidate_id, similarity, candidate_cluster in candidates:
            if geographic_diversity and candidate_cluster in selected_clusters:
                continue  # Skip if same cluster already selected

            selected_backups.append((candidate_id, similarity, candidate_cluster))
            selected_clusters.add(candidate_cluster)

            if len(selected_backups) >= num_backups:
                break

        # If not enough backups, fill from same cluster as fallback
        if len(selected_backups) < num_backups:
            for candidate_id, similarity, candidate_cluster in candidates:
                if (candidate_id, similarity, candidate_cluster) not in selected_backups:
                    selected_backups.append((candidate_id, similarity, candidate_cluster))
                    if len(selected_backups) >= num_backups:
                        break

        # Create recommendations
        for rank, (backup_id, similarity, backup_cluster) in enumerate(selected_backups, 1):
            backup_recommendations.append({
                'Critical_Supplier': hub_id,
                'Criticality_Score': hub_row['Criticality_Score'],
                'Backup_Rank': rank,
                'Backup_Supplier': backup_id,
                'Backup_Cluster': backup_cluster,
                'Similarity_Score': similarity,
                'Backup_Degree': G.degree(backup_id),
                'Geographic_Diversity': 'YES' if backup_cluster != node_to_cluster[hub_id] else 'NO',
                'Priority': 'IMMEDIATE' if hub_row['Risk_Level'] == 'CRITICAL' else 'HIGH'
            })

    df_backups = pd.DataFrame(backup_recommendations).sort_values(
        ['Critical_Supplier', 'Backup_Rank']
    )

    return df_backups

#---------------Network Reduction with Importance Preservation--------------
def recommend_edge_removal_enhanced(G, node_to_cluster, critical_score,
                                    reduction_target=0.20, safety_margin=0.05):
    """
    ENHANCED: Smarter edge removal preserving critical paths.

    Improvements:
    - Preserves edges connected to critical nodes
    - Maintains network connectivity with safety margin
    - Scores edges by criticality
    - Batch removal validation
    """

    edge_importance = {}
    for u, v in G.edges():
        importance = (critical_score.get(u, 0) + critical_score.get(v, 0)) / 2
        edge_importance[(u, v)] = importance

    sorted_edges = sorted(edge_importance.items(), key=lambda x: x[1])

    edges_to_remove = []
    G_test = G.copy()

    target_edges_to_remove = int(G.number_of_edges() * reduction_target)
    safety_edges_buffer = int(G.number_of_edges() * safety_margin)

    for edge, importance in sorted_edges:
        if len(edges_to_remove) >= (target_edges_to_remove - safety_edges_buffer):
            break

        u, v = edge

        # SKIP if either endpoint is a critical hub (score > 0.7)
        if critical_score.get(u, 0) > 0.7 or critical_score.get(v, 0) > 0.7:
            continue

        # SKIP intra-cluster edges (maintain internal resilience)
        if node_to_cluster[u] == node_to_cluster[v]:
            continue

        G_test.remove_edge(u, v)

        if nx.is_connected(G_test):
            edges_to_remove.append({
                'Edge': f"{u}-{v}",
                'From_Cluster': node_to_cluster[u],
                'To_Cluster': node_to_cluster[v],
                'Edge_Importance': importance,
                'Safe_to_Remove': True,
                'Reason': 'Non-critical inter-cluster bridge'
            })
        else:
            # Restore edge if removal disconnects graph
            G_test.add_edge(u, v)

    df_removal = pd.DataFrame(edges_to_remove)

    return df_removal, G_test

#----------------Integrating Neo4j's Graph Intelligence in Graph Neural Network-----
from neo4j import GraphDatabase


class Neo4jConnector:
    """
    Neo4j persistence layer for supply-chain graph and cascade simulations.
    """

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        if self.driver is not None:
            self.driver.close()

    # ------------------------------------------------------------------
    # SINGLE NODE (kept for compatibility)
    # ------------------------------------------------------------------
    def create_supplier(self, Supplier_ID, props):
        with self.driver.session() as session:
            session.run(
                """
                MERGE (s:Supplier {id: $id})
                SET s += $props
                """,
                id=int(Supplier_ID),
                props=props
            )

    # ------------------------------------------------------------------
    # BATCH SUPPLIER INGESTION (FAST)
    # ------------------------------------------------------------------
    def create_suppliers_batch(self, suppliers):
        """
        suppliers: list of dicts
        [
          {"id": 1, "props": {...}},
          {"id": 2, "props": {...}}
        ]
        """
        with self.driver.session() as session:
            session.run(
                """
                UNWIND $suppliers AS s
                MERGE (n:Supplier {id: s.id})
                SET n += s.props
                """,
                suppliers=suppliers
            )

    # ------------------------------------------------------------------
    # SINGLE RELATIONSHIP (kept for compatibility)
    # ------------------------------------------------------------------
    def create_relationships(self, u, v, weight=1.0):
        with self.driver.session() as session:
            session.run(
                """
                MERGE (a:Supplier {id: $u})
                MERGE (b:Supplier {id: $v})
                MERGE (a)-[r:SIMILAR_TO]->(b)
                SET r.weight = $w
                """,
                u=int(u),
                v=int(v),
                w=float(weight)
            )

    # ------------------------------------------------------------------
    # BATCH RELATIONSHIP INGESTION (CRITICAL FOR PERFORMANCE)
    # ------------------------------------------------------------------
    def create_relationships_batch(self, edges, batch_size=1000):
        """
        edges: list of tuples [(u, v), (u, v), ...]
        """
        with self.driver.session() as session:
            for i in range(0, len(edges), batch_size):
                batch = edges[i:i + batch_size]

                session.run(
                    """
                    UNWIND $edges AS e
                    MERGE (a:Supplier {id: e.u})
                    MERGE (b:Supplier {id: e.v})
                    MERGE (a)-[:SIMILAR_TO]->(b)
                    """,
                    edges=[{"u": int(u), "v": int(v)} for u, v in batch]
                )

    # ------------------------------------------------------------------
    # BATCH CLUSTER ASSIGNMENT
    # ------------------------------------------------------------------
    def assign_clusters_batch(self, node_to_cluster):
        """
        node_to_cluster: dict {supplier_id: cluster_id}
        """
        payload = [
            {"sid": int(sid), "cid": int(cid)}
            for sid, cid in node_to_cluster.items()
        ]

        with self.driver.session() as session:
            session.run(
                """
                UNWIND $rows AS r
                MERGE (c:Cluster {id: r.cid})
                WITH c, r
                MATCH (s:Supplier {id: r.sid})
                MERGE (s)-[:BELONGS_TO]->(c)
                """,
                rows=payload
            )

    # ------------------------------------------------------------------
    # MARK SOURCE DISRUPTIONS
    # ------------------------------------------------------------------
    def mark_disruptions(self, disruptions):
        """
        disruptions: list of dicts with keys
        supplier_id, severity, type
        """
        with self.driver.session() as session:
            session.run(
                """
                UNWIND $rows AS d
                MATCH (s:Supplier {id: d.supplier_id})
                SET s.disrupted = true,
                    s.severity = d.severity,
                    s.disruption_type = d.type
                """,
                rows=disruptions
            )

    # ------------------------------------------------------------------
    # NEW: PERSIST CASCADE RESULTS
    # ------------------------------------------------------------------
    def push_cascade(self, cascade_result):
        """
        Persist cascade propagation paths into Neo4j.

        cascade_result format (from CascadeEngine):
        {
            "scenario_id": int,
            "source_supplier": int,
            "affected_suppliers": set,
            "propagation_paths": dict,
            "final_impact": float
        }
        """

        scenario_id = int(cascade_result["scenario_id"])
        source = int(cascade_result["source_supplier"])

        edges = []
        for target, path in cascade_result["propagation_paths"].items():
            for i in range(len(path) - 1):
                edges.append({
                    "src": int(path[i]),
                    "dst": int(path[i + 1]),
                    "scenario": scenario_id
                })

        with self.driver.session() as session:
            # Scenario node
            session.run(
                """
                MERGE (sc:Scenario {id: $sid})
                SET sc.impact = $impact
                """,
                sid=scenario_id,
                impact=float(cascade_result["final_impact"])
            )

            # Source link
            session.run(
                """
                MATCH (s:Supplier {id: $src})
                MATCH (sc:Scenario {id: $sid})
                MERGE (s)-[:INITIATED]->(sc)
                """,
                src=source,
                sid=scenario_id
            )

            # Propagation edges
            session.run(
                """
                UNWIND $edges AS e
                MATCH (a:Supplier {id: e.src})
                MATCH (b:Supplier {id: e.dst})
                MERGE (a)-[r:CASCADE_TO {scenario: e.scenario}]->(b)
                """,
                edges=edges
            )

    # ------------------------------------------------------------------
    # CRITICAL SUPPLIER ANALYTICS
    # ------------------------------------------------------------------
    def critical_suppliers(self, top_k=10):
        """
        Returns suppliers ranked by cascade participation frequency.
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (s:Supplier)-[r:CASCADE_TO]->()
                RETURN s.id AS supplier_id, COUNT(r) AS cascade_count
                ORDER BY cascade_count DESC
                LIMIT $k
                """,
                k=int(top_k)
            )
            return [dict(record) for record in result]

    # ------------------------------------------------------------------
    # OPTIONAL: CLEAR CASCADE DATA (SAFE RESET)
    # ------------------------------------------------------------------
    def clear_cascades(self):
        with self.driver.session() as session:
            session.run(
                """
                MATCH ()-[r:CASCADE_TO]->()
                DELETE r
                """
            )

    # ------------------------------------------------------------------
    # QUICK CONNECTION TEST
    # ------------------------------------------------------------------
    def test_connection(self):
        with self.driver.session() as session:
            result = session.run("RETURN 1 AS ok")
            record = result.single()
            return record is not None and record.get("ok") == 1

#--------------EVALUATION & VALIDATION ---------------------
class EvaluationMetrics:
    """
    Evaluation utilities for cascade simulations.
    Used for analysis, reporting, and research validation.
    """

    @staticmethod
    def cascade_depth(cascade):
        """
        Maximum propagation depth in a cascade.
        """
        return max(
            len(path)
            for path in cascade["propagation_paths"].values()
        )

    @staticmethod
    def spread_velocity(cascade):
        """
        Number of suppliers affected in the cascade.
        """
        return len(cascade["affected_suppliers"])

    @staticmethod
    def criticality_score(node, cascades):
        """
        How often a supplier appears across cascades.
        """
        return sum(
            node in c["affected_suppliers"]
            for c in cascades
        )

import os
from IPython.display import display, Image


os.environ["NEO4J_URI"] = "neo4j+s://f097f7cb.databases.neo4j.io"
os.environ["NEO4J_USER"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "<neoj-aura-instance-password"

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

#-----Main Training Pipeline---------------
def main_training_pipeline():
    print("="*80)
    print("SUPPLY CHAIN GNN TRAINING PIPELINE")
    print("="*80)

    # [1] Load & preprocess
    df_raw = RealDataLoader.load_csv_data("/content/CoSupplyChainData.csv", sample_size=10000)
    prep = DataPreprocessor()
    df, X, y, fcols = prep.preprocess(df_raw)
    df = df.reset_index(drop=True)

    # [2] Build graph
    gb = GraphBuilder(df)
    G_s, _, _ = gb.build_all(X)

    print("\n[Graph] Pruning incoherent edges based on risk score...")

    edges_to_remove = [
        (u, v) for u, v in G_s.edges()
        if abs(df.iloc[u]["risk_score"] - df.iloc[v]["risk_score"]) > 0.9
    ]

    G_s.remove_edges_from(edges_to_remove)



    if len(edges_to_remove) == 0:
        print("✓ No incoherent edges to remove")
    else:
        print(f"✓ Removed {len(edges_to_remove)} incoherent edges")

    print(f"✓ Removed {len(edges_to_remove)} noisy edges")
    print(f"✓ Remaining edges: {G_s.number_of_edges()}")

    print("{Now PIPELINE CHECK IS ON}")
    print("The number of Nodes are:", G_s.number_of_nodes())
    print("The number of Edges are:", G_s.number_of_edges())

    # [3] PyG data
    edges = list(G_s.edges())
    edge_idx = torch.tensor(np.array(edges).T, dtype=torch.long)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)

    n = len(df)
    tr_idx = np.random.choice(n, int(0.7*n), replace=False)
    te_idx = np.setdiff1d(np.arange(n), tr_idx)

    tr_mask = torch.zeros(n, dtype=torch.bool)
    te_mask = torch.zeros(n, dtype=torch.bool)
    tr_mask[tr_idx] = True
    te_mask[te_idx] = True

    data = Data(
        x=X_t, edge_index=edge_idx, y=y_t,
        train_mask=tr_mask, test_mask=te_idx
    )

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(dev)

    mdict, metrdict = {}, {}

    # ==================== [8] MODEL TRAINING ====================
    print("\n[8a] Training of Graph Convolutional Network (GCN) starts...")
    gcn = GCN(X.shape[1]).to(dev)
    tgcn = Trainer(dev)  # type: ignore
    gcn = tgcn.train(gcn, data, epochs=150)
    _, gcn_logits = tgcn.eval_step(gcn, data)
    gpred = gcn_logits.argmax(dim=1)

    gmet = {
        'accuracy': accuracy_score(y_t[te_mask].cpu().numpy(), gpred[te_mask].cpu().numpy()),
        'precision': precision_score(y_t[te_mask].cpu().numpy(), gpred[te_mask].cpu().numpy(), average='weighted', zero_division=0.0),   #type: ignore
        'recall': recall_score(y_t[te_mask].cpu().numpy(), gpred[te_mask].cpu().numpy(), average='weighted', zero_division=0.0),         #type: ignore
        'f1': f1_score(y_t[te_mask].cpu().numpy(), gpred[te_mask].cpu().numpy(), average='weighted', zero_division=0.0)                   #type: ignore
    }
    mdict['GCN'], metrdict['GCN'] = gcn, gmet
    print(f"✓ Acc: {gmet['accuracy']:.4f}, F1: {gmet['f1']:.4f}")

    print("\n[8b] Training of Graph Attention Network (GAT) starts...")
    gat = GAT(X.shape[1]).to(dev)
    tgat = Trainer(dev)
    gat = tgat.train(gat, data, epochs=150)
    _, gat_logits = tgat.eval_step(gat, data)
    apred = gat_logits.argmax(dim = 1)

    amet = {
        'accuracy': accuracy_score(y_t[te_mask].cpu().numpy(), apred[te_mask].cpu().numpy()),
        'precision': precision_score(y_t[te_mask].cpu().numpy(), apred[te_mask].cpu().numpy(), average='weighted', zero_division=0.0),     #type: ignore
        'recall': recall_score(y_t[te_mask].cpu().numpy(), apred[te_mask].cpu().numpy(), average='weighted', zero_division=0.0),              #type: ignore
        'f1': f1_score(y_t[te_mask].cpu().numpy(), apred[te_mask].cpu().numpy(), average='weighted', zero_division=0.0)                    #type: ignore
    }
    mdict['GAT'], metrdict['GAT'] = gat, amet
    print(f"✓ Acc: {amet['accuracy']:.4f}, F1: {amet['f1']:.4f}")

    print("\n[8c] Training of Temporal GNN + LSTM...")
    tgnn = TemporalGNN(X.shape[1]).to(dev)
    ttgnn = Trainer(dev)  # type: ignore
    tgnn = ttgnn.train(tgnn, data, epochs=200)
    _, temp_logits = ttgnn.eval_step(tgnn, data)
    tpred = temp_logits.argmax(dim = 1)

    tmet = {
        'accuracy': accuracy_score(y_t[te_mask].cpu().numpy(), tpred[te_mask].cpu().numpy()),
        'precision': precision_score(y_t[te_mask].cpu().numpy(), tpred[te_mask].cpu().numpy(), average='weighted', zero_division=0.0),   #type: ignore
        'recall': recall_score(y_t[te_mask].cpu().numpy(), tpred[te_mask].cpu().numpy(), average='weighted', zero_division=0.0),         #type: ignore
        'f1': f1_score(y_t[te_mask].cpu().numpy(), tpred[te_mask].cpu().numpy(), average='weighted', zero_division=0.0)                  #type: ignore
    }
    mdict['Temporal'], metrdict['Temporal'] = tgnn, tmet
    print(f"✓ Accuracy: {tmet['accuracy']:.4f}, F1: {tmet['f1']:.4f}")

    # --------------------------------------------------
    # [8d] Weighted Ensemble
    # --------------------------------------------------
    print("\n[8d] Weighted Ensemble (no training)...")

    # Convert tensor logits to probabilities
    gcn_prob = torch.softmax(gcn_logits, dim=1)       # to -> 0.1,0.2,0.3......
    gat_prob = torch.softmax(gat_logits, dim=1)
    temp_prob = torch.softmax(temp_logits, dim=1)

    # Weighted combination (empirically stable)
    ensemble_prob = (
        0.15 * gcn_prob +
        0.25 * gat_prob +
        0.60 * temp_prob
    )

    epred = ensemble_prob.argmax(dim=1)

    emet = {
        'accuracy': accuracy_score(
            y_t[te_mask].cpu().numpy(),
            epred[te_mask].cpu().numpy()
        ),
        'precision': precision_score(
            y_t[te_mask].cpu().numpy(),
            epred[te_mask].cpu().numpy(),
            average='weighted',
            zero_division=0.0                  #type: ignore
        ),
        'recall': recall_score(
            y_t[te_mask].cpu().numpy(),
            epred[te_mask].cpu().numpy(),
            average='weighted',
            zero_division=0.0                 #type: ignore
        ),
        'f1': f1_score(
            y_t[te_mask].cpu().numpy(),
            epred[te_mask].cpu().numpy(),
            average='weighted',
            zero_division=0.0                  #type: ignore
        )
    }

    metrdict['Ensemble'] = emet
    print(f"✓ Ensemble Acc: {emet['accuracy']:.4f}, F1: {emet['f1']:.4f}")

    # [8] Fairness + Viz
    fair = FairnessAnalysis.analyze(df, epred.cpu().numpy())
    Viz.plot_fairness(fair)
    Viz.plot_network(G_s, df, epred.cpu().numpy(), "Supply Chain Graph Neural Knowledge Network")
    print("✓ Model training & evaluation completed")
    return epred

#----------------Main Graph Pipeline------------------
def main_graph_pipeline(epred):
    print("=" * 80)
    print("\nSUPPLY CHAIN GRAPH ANALYTICS & NEO4J PIPELINE")
    print("=" * 80)

    df_raw = RealDataLoader.load_csv_data(
        "/content/CoSupplyChainData.csv", sample_size=2000
    )
    prep = DataPreprocessor()
    df, X, y, fcols = prep.preprocess(df_raw)

    # --------------------------------------------------
    # Canonical supplier ID alignment
    # --------------------------------------------------
    df = df.reset_index(drop=True)
    df.index.name = "supplier_id"

    print("\nDF index (supplier_id) range:",
          df.index.min(), "to", df.index.max())


    Viz.plot_risk(df)

    if "supplier_id" in df.columns:
      df = df.set_index("supplier_id")
    elif "Supplier_ID" in df.columns:
      df = df.set_index("Supplier_ID")
    # --------------------------------------------------
    # [2] Generate DISRUPTION SCENARIOS
    # --------------------------------------------------
    print("\n[Disruption] Generating disruption scenarios...")
    dgen = DisruptionGenerator(df)
    drupt = dgen.generate_disruptions(50)

    # --------------------------------------------------
    # [3] Build graph
    # --------------------------------------------------
    gb = GraphBuilder(df)
    G_s, G_p, G_r = gb.build_all(X)

    # --------------------------------------------------
    # [3.5] Degree-aware disruption seeds
    # --------------------------------------------------
    print("\n[Disruption] Selecting high-impact cascade seeds...")

    # original disrupted supplier IDs (from data)
    disrupted_nodes = (
        drupt['supplier_id']
        .astype(int)
        .unique()
        .tolist()
    )


    # graph degree
    deg = dict(list(G_s.degree))             #type: ignore
    ranked_nodes = sorted(deg.items(), key=lambda x:x[1], reverse=True)
    ranked_nodes = [n for n, _ in ranked_nodes]

    # intersect disruptions with high-degree nodes
    seed_nodes = [n for n in ranked_nodes if n in disrupted_nodes]

    # fallback: take top-degree nodes if too few
    if len(seed_nodes) < 10:
      seed_nodes = ranked_nodes[:50]

    print(f"✓ Using {len(seed_nodes)} high-impact disruption seeds")


    # --------------------------------------------------
    # [5] Enforce minimum seeds
    # --------------------------------------------------
    if len(disrupted_nodes) < 10:
        print("⚠ Too few valid disruption seeds, expanding randomly")
        disrupted_nodes = list(
            np.random.choice(
                list(G_s.nodes()),
                size=min(50, G_s.number_of_nodes()),
                replace=False
            )
        )

    print(f"✓ Disruption seeds used: {len(disrupted_nodes)} suppliers")

    # --------------------------------------------------
    # [6] Centrality analysis
    # --------------------------------------------------
    df_hubs, centrality_score = identify_critical_hubs_enhanced(G_s)

    # --------------------------------------------------
    # [7] Community detection
    # --------------------------------------------------
    df_clusters, node_to_cluster, cluster_stats, modularity = (
        create_resilient_clusters_enhanced(G_s)
    )
    Viz.plot_cluster_overview(df_clusters)

    # --------------------------------------------------
    # [8] Bottleneck detection
    # --------------------------------------------------
    identify_critical_bottlenecks(G_s, node_to_cluster)

    print("Graph nodes:", G_s.number_of_nodes())

    valid_seeds = [n for n in disrupted_nodes if n in G_s.nodes()]
    invalid_seeds = [n for n in disrupted_nodes if n not in G_s.nodes()]

    print("Valid seeds:", valid_seeds[:10])
    print("Invalid seeds:", invalid_seeds[:10])
    if (invalid_seeds == []):
      print(f"\nThe nodes are valid, you are good to go for further analysis with valid nodes of {len(valid_seeds)} seeds with Cascade impact analysis....")

    # --------------------------------------------------
    # [9] BASELINE CASCADE (rule-based)
    # --------------------------------------------------
    print("\n[Cascade] Baseline cascade using disruption scenarios...")
    casc_baseline = dgen.cascade_impact(G_s, drupt)

    if len(casc_baseline.get("affected", [])) > 0:
        Viz.plot_cascade(casc_baseline)
    else:
        print("⚠ No cascade data to plot")

    # --------------------------------------------------
    # [10] ENGINE-BASED CASCADE (simulation)
    # --------------------------------------------------
    print("\n[Cascade] Engine-based cascade analysis...")

    engine = CascadeEngine(G_s, decay=0.7, max_depth=4)
    all_cascades = []

    print("Disrupted nodes (sample):", disrupted_nodes[:10])
    print("DF index range:", df.index.min(), "to", df.index.max())

    for i, sid in enumerate(disrupted_nodes[:5]):

        # Supplier ID is the DataFrame index
        # No need for 'if sid not in df.index' as disrupted_nodes are filtered by G_s.nodes()
        risk_val = float(df.loc[sid, "risk_score"])

        c = engine.run(
            disruption={
                "supplier_id": int(sid),
                "severity": risk_val,
                "type": "operational"
            },
            scenario_id=i,
            t0=0
        )

        all_cascades.append(c)

    # --------------------------------------------------
    # Extract cascade metrics
    # --------------------------------------------------
    affected_counts = [len(c["affected_suppliers"]) for c in all_cascades]
    impact_scores = [c["final_impact"] for c in all_cascades]

    # --------------------------------------------------
    # [11] CASCADE EVALUATION
    # --------------------------------------------------
    cascade_criticality_score = EvaluationMetrics.criticality_score(
        node=12,
        cascades=all_cascades
    )

    depth = EvaluationMetrics.cascade_depth(all_cascades[0])
    velocity = EvaluationMetrics.spread_velocity(all_cascades[0])

    print(f"Cascade depth: {depth}")
    print(f"Spread velocity: {velocity}")
    print(f"Criticality score (supplier 12): {cascade_criticality_score}")

    if not all_cascades:
      print("⚠ No valid engine cascades generated")
      return df

    Viz.plot_cascade_graph(all_cascades[0])

    # --------------------------------------------------
    # [12] IMPROVED CASCADE v2 (SEEDED)
    # --------------------------------------------------
    print("\n[Cascade] Improved cascade analysis (seeded)...")
    cascade_results_v2 = improved_cascade_analysis_v2(
        G_s,
        node_to_cluster,
        disrupted_nodes_list=seed_nodes,
        sample_size=50,
        isolation_enabled=True
    )
    Viz.plot_cascade_comparison(cascade_results_v2)

    # --------------------------------------------------
    # [13] Backup supplier recommendation
    # --------------------------------------------------
    recommend_backup_suppliers_enhanced(
        G_s, node_to_cluster, df_hubs
    )

    # --------------------------------------------------
    # [14] Network reduction
    # --------------------------------------------------
    recommend_edge_removal_enhanced(
        G_s, node_to_cluster, centrality_score
    )

    # --------------------------------------------------
    # [15] NEO4J INGESTION
    # --------------------------------------------------
    print("\n[Neo4j] Uploading graph to Neo4j...")
    display(
        Image(
            url="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSmqWmFe1erf0cT0_zN2VRgrGfrRnRe4qPVEA&s"
        )
    )

    neo4j = Neo4jConnector(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD
    )

    suppliers_payload = [
        {
            "id": int(row["Supplier_ID"]),
            "props": {
                "criticality_score": float(row["Criticality_Score"]),
                "risk_level": str(row["Risk_Level"]),
                "disrupted": int(row["Supplier_ID"]) in disrupted_nodes,

                # Analytics enrichment
                "risk_score": float(row.get("risk_score", 0)),
                "supplier_risk_avg": float(row.get("supplier_risk_avg", 0)),
                "region_risk_density": float(row.get("region_risk_density", 0)),

                # Operational context
                "shipping_delay_days": int(row.get("shipping_delay_days", 0)),
                "shipping_mode": str(row.get("Shipping Mode", "Unknown")),
                "order_status": str(row.get("Order Status", "Unknown")),

                # Geography / market
                "region": str(row.get("Order Region", "Unknown")),
                "market": str(row.get("Market", "Unknown"))
            }
        }
        for _, row in df_hubs.iterrows()
    ]

    # --------------------------------------------------
    # Persist cascades into Neo4j
    # --------------------------------------------------
    print("\n[Neo4j] Persisting cascade scenarios...")

    print(f"✓ {len(all_cascades)} cascade scenarios stored in Neo4j")

    print("Disruptions:", len(drupt))
    print("Valid seeds:", len(valid_seeds))
    print("Invalid seeds:", len(invalid_seeds))
    print("\nRows:", df.shape)
    print("Risk score stats:", df["risk_score"].describe())
    print("Locations:", df["location"].nunique())
    print("Graph nodes:", G_s.number_of_nodes())
    print("Graph edges:", G_s.number_of_edges())

    print("Supplier nodes:", G_s.number_of_nodes())
    print("Supplier edges:", G_s.number_of_edges())
    print("Avg degree:", sum(dict(G_s.degree()).values()) / G_s.number_of_nodes())                   #type: ignore

    neo4j.create_suppliers_batch(suppliers_payload)
    neo4j.mark_disruptions(drupt.to_dict(orient="records"))
    neo4j.create_relationships_batch(list(G_s.edges()))
    neo4j.assign_clusters_batch(node_to_cluster)

    for c in all_cascades:
      neo4j.push_cascade(c)

    neo4j.close()
    print("✓ Graph analytics, cascades & insights has been ingested into Neo4j")

    # ----------------------
    # Attaching Predictions back to dataframe (Using mapping predictions to labels)
    #----------------------
    df_out = df.copy()
    risk_map = {0: "Low", 1: "Medium", 2: "High"}
    df_out["Predicted_Risk"] = pd.Series(epred.cpu().numpy()).map(risk_map)

    print("\n[PREDICTION] Risk levels assigned to suppliers")
    print(df_out["Predicted_Risk"].value_counts())

    print("\n[INSIGHT] High-risk concentration by Market & Region")

    risk_by_region = (
        df_out[df_out["Predicted_Risk"] == "High"]
        .groupby(["Market", "Order Region"])
        .size()
        .sort_values(ascending=False)
        .head(10)
    )
    print(risk_by_region)

    print("\n[INSIGHT] Delivery status impact on predicted risk")
    delivery_risk = (
        df_out.groupby(["Delivery Status", "Predicted_Risk"])
        .size()
        .unstack(fill_value=0)
    )
    print(delivery_risk)


    print("\n[INSIGHT] Discount levels associated with High Risk")
    high_risk_discount = (
        df_out[df_out["Predicted_Risk"] == "High"]["Order Item Discount"]
        .describe()
    )
    print(high_risk_discount)

    print("\n[INSIGHT] Top countries & states with predicted supply chain risk")
    geo_risk = (
        df_out[df_out["Predicted_Risk"] == "High"]
        .groupby(["Order Country", "Order State"])
        .size()
        .sort_values(ascending=False)
        .head(10)
    )
    print(geo_risk)

    print("\n[EARLY WARNING] Risk trend by order date")

    if "order date (DateOrders)" in df_out.columns:
        date_col = "order date (DateOrders)"
    elif "Order Date (DateOrders)" in df_out.columns:
        date_col = "Order Date (DateOrders)"
    elif "DateOrders" in df_out.columns:
        date_col = "DateOrders"
    else:
        date_col = None

    if date_col:
        df_out["Order_Date"] = pd.to_datetime(
            df_out[date_col], errors="coerce"
        )
    else:
        print("⚠ No order date column found — skipping temporal risk trend")


    time_risk = (
        df_out[df_out["Predicted_Risk"] == "High"]
        .groupby(df_out["Order_Date"].dt.to_period("M"))
        .size()
    )
    print(time_risk.tail(6))

    print("\n[DECISION SUPPORT INSIGHTS]")
    print("- Regions with frequent HIGH risk should be prioritized for backup suppliers")
    print("- Late deliveries + high discounts are strong early indicators of disruption")
    print("- Temporal GNN captures rising risk trends before failures occur")
    print("- Use predicted High-Risk suppliers as cascade-prevention anchors")


    # ----------------------
    # Persist final analytics output for UI tools (Lux / PyGWalker / etc.)
    # ----------------------
    df_out.to_csv("graph_pipeline_results.csv", index=False)
    print("✓ UI-ready dataset saved as graph_pipeline_results.csv")

    return df_out

if __name__ == "__main__":
  epred_result = main_training_pipeline()
  df_ui =   main_graph_pipeline(epred_result)
  import pygwalker as pyg
  pyg.walk(df_ui)
