import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from shiny import App, Inputs, Outputs, Session, reactive, render, ui
import tempfile
from matplotlib.lines import Line2D
import EoN
import random

# --------------------- basic functions ---------------------

def select_node_outbreak(prob_dict):
    nodes = list(prob_dict.keys())
    probs = list(prob_dict.values())
    if not probs or sum(probs) == 0:
        return random.choice(nodes)
    return random.choices(nodes, weights=probs, k=1)[0]

def calculate_objective(G, prob_dict, selected_nodes, simulations=1000):
    total_gain = 0
    if not selected_nodes:
        return 0
    for _ in range(simulations):
        selected_node = select_node_outbreak(prob_dict)
        sim = EoN.Gillespie_SIR(G, tau=0.5, gamma=1., initial_infecteds=selected_node, return_full_data=True)
        b, c = list(sim.t()), sim.I() + sim.R()
        g_max = 0
        for s in selected_nodes:
            m1 = sim.node_history(s)
            if len(m1[0]) == 3:
                g1 = c[-1] - c[b.index(m1[0][1])]
            elif len(m1[0]) == 2:
                g1 = c[-1] - c[0]
            else:
                g1 = 0
            g_max = max(g_max, g1)
        total_gain += g_max
    return total_gain / simulations

def greedy_max_influence_rfsm(G, prob_dict, rounds=6, simulations=50):
    node_list = list(G.nodes())
    best = []
    available_nodes = set(node_list)
    for _ in range(rounds):
        candidates = [best + [x] for x in available_nodes]
        if not candidates: break
        mean_gains = np.zeros(len(candidates))
        for _ in range(simulations):
            selected_node = select_node_outbreak(prob_dict)
            sim = EoN.Gillespie_SIR(G, tau=0.5, gamma=1., initial_infecteds=selected_node, return_full_data=True)
            b, c = list(sim.t()), sim.I() + sim.R()
            for i, candidate in enumerate(candidates):
                g_max = 0
                for s in candidate:
                    m1 = sim.node_history(s)
                    if len(m1[0]) == 3:
                        g1 = c[-1] - c[b.index(m1[0][1])]
                    elif len(m1[0]) == 2:
                        g1 = c[-1] - c[0]
                    else:
                        g1 = 0
                    g_max = max(g_max, g1)
                mean_gains[i] += g_max
        mean_gains /= simulations
        best_idx = np.argmax(mean_gains)
        best = candidates[best_idx]
        available_nodes.remove(best[-1])
    
    obj_value = calculate_objective(G, prob_dict, best, simulations=1000)
    return best, obj_value

def greedy_max_influence(G, prob_dict, rounds=6, simulations=1000):
    return greedy_max_influence_rfsm(G, prob_dict, rounds, simulations=simulations)

def load_network(file_path: str, original_name: str):
    if original_name.endswith(".gml"):
        G = nx.read_gml(file_path)
    elif original_name.endswith(".csv"):
        df = pd.read_csv(file_path)
        G = nx.from_pandas_edgelist(df, source=df.columns[0], target=df.columns[1])
    elif original_name.endswith(".xlsx"):
        df = pd.read_excel(file_path)
        G = nx.from_pandas_edgelist(df, source=df.columns[0], target=df.columns[1])
    else:
        raise ValueError("Unsupported network file format!")
    
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    
    old_nodes = list(G.nodes())
    mapping = {old_node: str(i) for i, old_node in enumerate(old_nodes)}
    G = nx.relabel_nodes(G, mapping)
    
    reverse_mapping = {v: k for k, v in mapping.items()}
    return G, reverse_mapping

def load_prob(file_path: str, original_name: str) -> dict:
    if original_name.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif original_name.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    elif original_name.endswith(".txt"):
        df = pd.read_csv(file_path, header=None)
    else:
        raise ValueError("Unsupported probability file format.")

    if df.shape[1] == 2:
        return dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(float)))
    elif df.shape[1] == 1:
        return dict(zip(df.index.astype(str), df.iloc[:, 0].astype(float)))
    else:
        raise ValueError("Probability file format error: Should have 1 or 2 columns.")

def plot_network(G, prob_dict, selected, pos, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    node_colors = [prob_dict.get(n, 0) for n in G.nodes()]
    
    min_color, max_color = (0, 1) if min(node_colors) == max(node_colors) else (min(node_colors), max(node_colors))
    norm = Normalize(vmin=min_color, vmax=max_color)
    cmap = plt.cm.Blues
    
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', alpha=0.5)
    
    nodes_plot = nx.draw_networkx_nodes(G, pos, ax=ax, 
                                       node_color=node_colors,
                                       cmap=cmap,
                                       vmin=min_color,
                                       vmax=max_color,
                                       edgecolors='black',
                                       linewidths=1,
                                       node_size=120)
    
    # Draw all node labels with adaptive colors
    label_dict = {n: n for n in G.nodes()}
    for node in G.nodes():
        node_prob = prob_dict.get(node, 0)
        # Normalize the probability to get color intensity
        normalized_value = (node_prob - min_color) / (max_color - min_color) if max_color > min_color else 0.5
        # Use white text for dark nodes (high probability), black for light nodes
        text_color = 'white' if normalized_value > 0.5 else 'black'
        nx.draw_networkx_labels(G, pos, labels={node: node}, 
                               font_size=8, font_color=text_color, 
                               font_weight='normal', ax=ax)
    
    if selected:
        selected_nodes = [n for n in G.nodes() if n in selected]
        if selected_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=selected_nodes, ax=ax,
                                   node_color=[prob_dict.get(n, 0) for n in selected_nodes],
                                   cmap=cmap,
                                   vmin=min_color,
                                   vmax=max_color,
                                   edgecolors='red',
                                   linewidths=3,
                                   node_size=140)
            # Redraw labels for selected nodes with bold font
            for node in selected_nodes:
                node_prob = prob_dict.get(node, 0)
                normalized_value = (node_prob - min_color) / (max_color - min_color) if max_color > min_color else 0.5
                text_color = 'white' if normalized_value > 0.5 else 'black'
                nx.draw_networkx_labels(G, pos, labels={node: node}, 
                                       font_size=8, font_color=text_color, 
                                       font_weight='bold', ax=ax)
    
    ax.set_title(title, fontsize=14)
    ax.axis("off")
    
    sm = ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax, label='Emergence Probability', shrink=0.8)
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Selected node', 
               markerfacecolor='gray', markeredgecolor='red', markersize=10, markeredgewidth=2)
    ]
    ax.legend(handles=legend_elements, loc='best', frameon=True)
    
    return fig

def probability_generate(G, alpha, beta, corr, s):
    importance = list(nx.degree_centrality(G).values())
    
    mean = [0, 0]
    cov = [[1, corr], [corr, 1]]
    samples = np.random.multivariate_normal(mean, cov, s)
    
    x_ranks = np.argsort(np.argsort(samples[:, 0]))
    y_ranks = np.argsort(np.argsort(samples[:, 1]))
    importance_ranks = np.argsort(np.argsort(importance))

    y_corr_ranks = np.zeros(s, dtype=int)
    y_corr_ranks[importance_ranks] = y_ranks
    
    probabilities = sorted(np.random.beta(alpha, beta, s))
    
    final_probs = np.zeros(s)
    final_probs[y_corr_ranks] = probabilities
    
    return dict(zip(G.nodes(), final_probs))


# --------------------- Shiny App UI ---------------------
app_ui = ui.page_fluid(
    ui.h2("Sentinel Node Selection for Disease Surveillance", style="text-align: center; margin-top: 20px;"),
    ui.card(
        ui.markdown(
            """
            This application helps you select the optimal **Sentinel Nodes** in a complex network to detect outbreaks (e.g., diseases) as early as possible.
            You can upload your own network and emergence probability files, or use the parameters below to generate a synthetic probability distribution.
            **Cases Prevented**: This score represents the average number of nodes 'saved' from infection due to early warning from your selected sentinel nodes. **A higher score is better.**
            """
        ),
        style="background-color: #f8f9fa; margin-bottom: 20px;"
    ),
    
    # First Row: Control Panels
    ui.layout_columns(
        # Column 1: Data Upload
        ui.card(
            ui.h5("1. Load Data", style="margin-bottom: 15px;"),
            ui.input_file("network_file", "Network file (.gml/.csv/.xlsx)", accept=[".gml", ".csv", ".xlsx"]),
            ui.input_file("prob_file", "Probability file (.csv/.xlsx/.txt)", accept=[".csv", ".xlsx", ".txt"]),
            ui.hr(),
            ui.download_button("download_net", "Example Network", class_="w-100 mb-2"),
            ui.download_button("download_prob", "Example Probability", class_="w-100"),
        ),
        
        # Column 2: Choose Method
        ui.card(
            ui.h5("2. Choose a Method", style="margin-bottom: 15px;"),
            ui.input_numeric("k", "Number of sentinel nodes (k)", value=6, min=1),
            ui.HTML(
                """
                <p style="font-size: 12px; color: #6c757d; margin-top: 10px; margin-bottom: 10px; line-height: 1.3;">
                • <strong>RFSM:</strong> Select nodes with high ranking based on trained random forest model.<br><br>
                • <strong>Greedy:</strong> Select nodes stepwise by choosing the optimal node determined through simulation at each stage.<br><br>
                • <strong>Manual:</strong> Select nodes manually and evaluate performance.
                </p>
                """
            ),
            ui.input_radio_buttons(
                "method", "Selection Method",
                choices={"RFSM": "RFSM", "Greedy": "Greedy", "Manual": "Manual Selection"},
                selected="RFSM"
            ),
            ui.output_ui("method_info")
        ),
        
        # Column 3: Probability Generation
        ui.card(
            ui.h5("3. Configure Probabilities (Optional)", style="margin-bottom: 15px;"),
            ui.p("Only used if no probability file is uploaded.", style="font-size: 12px; color: #6c757d;"),
            ui.input_numeric("alpha", "Alpha", value=0.1, min=0.1),
            ui.p("Controls the probability distribution. Smaller values mean lower probabilities for most nodes.", style="font-size: 12px; color: #6c757d;"),
            ui.input_numeric("beta", "Beta", value=5.0, min=0.1),
            ui.p("Controls the probability distribution. Larger values also contribute to lower probabilities for most nodes.", style="font-size: 12px; color: #6c757d;"),
            ui.input_numeric("corr", "Correlation", value=-0.7, min=-1, max=1),
            ui.p("Correlation between node degree and emergence probability. Negative values mean central nodes are less likely to be sources.", style="font-size: 12px; color: #6c757d;")
        ),
        
        # Column 4: Manual Selection
        ui.card(
            ui.h5("4. Manual Selection (Optional)", style="margin-bottom: 15px;"),
            ui.output_ui("manual_selection_panel")
        ),
        
        col_widths=[3, 3, 3, 3]
    ),
    
    # Second Row: Results
    ui.layout_columns(
        ui.card(
            ui.output_ui("results_title"),
            ui.output_ui("results_description"),
            ui.output_text_verbatim("main_output_text"),
            full_screen=True,
            style="min-height: 800px;"
        ),
        ui.card(
            ui.h5("Network Visualization"),
            ui.output_plot("main_output_plot"),
            full_screen=True,
            style="min-height: 800px;"
        ),
        col_widths=[5, 7]
    )
)

# --------------------- Shiny Server ---------------------
def server(input: Inputs, output: Outputs, session: Session):

    manual_results = reactive.Value(None)

    @reactive.Calc
    def graph_data():
        if not input.network_file():
            return None, None, None, None
        
        net_file = input.network_file()[0]
        G, reverse_mapping = load_network(net_file["datapath"], net_file["name"])
        pos = nx.spring_layout(G, seed=42)
        
        prob = {}
        if not input.prob_file():
            prob = probability_generate(G, input.alpha(), input.beta(), input.corr(), len(G.nodes()))
        else:
            prob_file = input.prob_file()[0]
            prob_raw = load_prob(prob_file["datapath"], prob_file["name"])
            
            for node_id in G.nodes():
                original_id = str(reverse_mapping.get(node_id))
                if original_id in prob_raw:
                     prob[node_id] = prob_raw[original_id]
                else:
                     prob[node_id] = 0.001
        
        prob_sum = sum(prob.values())
        if prob_sum > 0:
            prob = {k: v / prob_sum for k, v in prob.items()}
        
        return G, prob, pos, reverse_mapping

    @output
    @render.ui
    def method_info():
        method = input.method()
        info = {
            "RFSM": "Select node with the high ranking based on the trained random forest model.",
            "Greedy": "Select nodes stepwise by choosing the optimal node determined through simulation at each stage.",
            "Manual": "Select nodes manually and evaluate performance."
        }
        return ui.p(info.get(method, ""), style="font-size: 12px; color: #6c757d; margin-top: 10px;")

    @output
    @render.ui
    def manual_selection_panel():
        method = input.method()
        if method != "Manual":
            return ui.p("Switch to 'Manual Selection' method to select nodes.", style="font-size: 12px; color: #6c757d;")
        
        data = graph_data()
        if data[0] is None:
            return ui.p("Please upload a network file first.", class_="text-danger")
        
        G, _, _, reverse_mapping = data
        choices = {node: f"{node}: {reverse_mapping.get(node, 'N/A')}" for node in G.nodes()}
        
        return ui.TagList(
            ui.input_selectize(
                "manual_nodes",
                "Select nodes:",
                choices=choices,
                multiple=True,
                options={"placeholder": "Select sentinel nodes..."}
            ),
            ui.input_action_button("calc_manual", "Calculate", class_="btn-primary w-100 mt-2")
        )

    @reactive.Calc
    def rfsm_results():
        G, prob, _, _ = graph_data()
        if G is None or prob is None: return None
        return greedy_max_influence_rfsm(G, prob, rounds=input.k(), simulations=50)

    @reactive.Calc
    def greedy_results():
        G, prob, _, _ = graph_data()
        if G is None or prob is None: return None
        return greedy_max_influence(G, prob, rounds=input.k(), simulations=1000)

    @reactive.Effect
    @reactive.event(input.calc_manual)
    def _():
        G, prob, _, _ = graph_data()
        manual_input = input.manual_nodes()
        if G and prob and manual_input:
            obj_value = calculate_objective(G, prob, list(manual_input), simulations=1000)
            manual_results.set((list(manual_input), obj_value))
        else:
            manual_results.set(None)

    @output
    @render.ui
    def results_title():
        method = input.method()
        return ui.h5(f"{method} Selection Results", style="margin-bottom: 10px;")

    @output
    @render.ui
    def results_description():
        method = input.method()
        descriptions = {
            "RFSM": "Select node with the high ranking based on the trained random forest model.",
            "Greedy": "Select nodes stepwise by choosing the optimal node determined through simulation at each stage.",
            "Manual": "Performance evaluation of your custom node selection."
        }
        return ui.p(descriptions.get(method, ""), style="font-size: 13px; color: #6c757d; margin-bottom: 15px;")

    @output
    @render.text
    def main_output_text():
        data = graph_data()
        if data[0] is None: return "Please upload a network file to begin."
        
        method = input.method()
        results = None
        
        with ui.Progress(min=0, max=1) as p:
            p.set(0.1, message="Initializing...")
            if method == "RFSM":
                p.set(0.3, message="Running RFSM algorithm...")
                results = rfsm_results()
            elif method == "Greedy":
                p.set(0.3, message="Running Greedy algorithm...")
                results = greedy_results()
            elif method == "Manual":
                results = manual_results.get()
                if not results: return "Select nodes and click 'Calculate' button."
            p.set(1, message="Complete!")

        if not results: return "Error: Calculation failed or no data available."
        selected, obj_value = results
        return f"Selected Nodes: {selected}\n\nCases Prevented: {obj_value:.4f}\n\nThis score represents the average number of infections prevented by early detection at these sentinel nodes."

    @output
    @render.plot
    def main_output_plot():
        data = graph_data()
        G, prob, pos, _ = data
        if G is None or prob is None:
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.text(0.5, 0.5, 'Upload a network file to see visualization', 
                    ha='center', va='center', fontsize=13, color='gray')
            ax.axis('off')
            return fig

        method = input.method()
        results = None
        if method == "RFSM": results = rfsm_results()
        elif method == "Greedy": results = greedy_results()
        elif method == "Manual": results = manual_results.get()

        selected = []
        if results:
            selected, _ = results
        
        if method == "Manual" and not selected:
             fig, ax = plt.subplots(figsize=(7, 6))
             ax.text(0.5, 0.5, 'Select nodes and click "Calculate"', 
                     ha='center', va='center', fontsize=13, color='gray')
             ax.axis('off')
             return fig

        return plot_network(G, prob, selected, pos, f"{method} Selection")

    @session.download(filename="example_network.gml")
    def download_net():
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".gml") as f:
            nx.write_gml(nx.karate_club_graph(), f.name)
            return f.name

    @session.download(filename="example_emergence_probability.xlsx")
    def download_prob():
        G = nx.karate_club_graph()
        df = pd.DataFrame({
            "NodeID": list(G.nodes()),
            "Probability": np.random.beta(0.2, 5.0, len(G.nodes()))
        })
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as f:
            df.to_excel(f.name, index=False)
            return f.name

app = App(app_ui, server)