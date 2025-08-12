import textwrap
import re
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import networkx as nx
import pysmiles
from cgsmiles import read_fragments
import cgsmiles
from cgsmiles.drawing import draw_molecule

def node_match(n1, n2):
    return n1['element'] == n2['element']

def edge_match(e1, e2):
    return e1['order'] == e2['order']

def bead_to_numeric(beadtype):
    pol_to_idx =  {"Q":0, "P":1, "N": 2, "C": 3, "X":4}
    sort_idx = {6:0, 5:1, 4:2, 3:3, 2:4, 1:5}
    # size, polarity, level, labels (a, h)
    bead_idx = [0, 0, 0, 0]
    if beadtype.startswith('S'):
        beadtype = beadtype[1:]
        bead_idx[0] = 1
    elif beadtype.startswith('T'):
        bead_idx[0] = 2
        beadtype = beadtype[1:]
    bead_idx[1] = pol_to_idx[beadtype[0]]
    bead_idx[2] = sort_idx[int(beadtype[1])]
    if len(beadtype) > 2:
        bead_idx[3] = 1
    return tuple(bead_idx)

def make_fragment_list(library_data_frame):
    molecule_dict = {}
    bead_type_dict = {}
    beads = []
    fragment_dict = defaultdict(list)
    counter = 0
    for mol_name, mol_tag, smiles_str, cgsmiles_str in zip(library_data_frame.get('Name'),
                                                           library_data_frame.get('MOLTAG'),
                                                           library_data_frame.get('SMILES'),
                                                           library_data_frame.get('CGSmiles')):
        _, frag_str = re.findall(r"\{[^\}]+\}", cgsmiles_str)
        frag_dict = read_fragments(frag_str)
        if len(frag_dict) == 1:
            continue

        try:
            cg_mol, aa_mol = cgsmiles.MoleculeResolver.from_string(cgsmiles_str, legacy=True).resolve_all()
        except SyntaxError:
            continue

        if len(aa_mol) > 20:
           continue

        pysmiles.smiles_helper.remove_explicit_hydrogens(aa_mol)
        cg_mol = cgsmiles.graph_utils.annotate_fragments(cg_mol, aa_mol)
        molecule_dict[(mol_name, mol_tag)] = aa_mol
        for aa_node in aa_mol.nodes:
            for fragname, target in  aa_mol.nodes[aa_node]['mapping']:
                frag_dict[fragname].nodes[target]['hcount'] = aa_mol.nodes[aa_node]['hcount']

        # we discard all fragments that are real graph isomorphs
        # that means totally identical for what we wan't to do
        # however we keep thos with different bead types
        cgs = re.findall(r"\{[^\}]+\}", cgsmiles_str)[0]
        for bead_type, graph in frag_dict.items():
            if len(graph) == 1:
                continue
            # some cleaning of the bead types from the CGSmilesDB
            if not bead_type[-1].isdigit()  and bead_type[-1].isupper():
                bead_type = bead_type[:-1]

            bead_idx = bead_to_numeric(bead_type)
            frag_nodes = []
            for node in aa_mol.nodes:
                if aa_mol.nodes[node]['fragname'] == bead_type:
                    frag_nodes.append(aa_mol.nodes[node]['fragid'])

            if bead_type in fragment_dict:
                for other_graph in fragment_dict[bead_type]:
                    if nx.is_isomorphic(graph, other_graph, node_match=node_match, edge_match=edge_match):
                        break
                else:
                    fragment_dict[bead_type].append(graph)
                    bead_type_dict[counter] = [mol_tag, mol_name, bead_type, graph, frag_nodes, cgs]
                    beads.append([bead_idx,[mol_tag, mol_name, bead_type, graph, frag_nodes, cgs]])
                    counter += 1
            else:
                fragment_dict[bead_type].append(graph)
                bead_type_dict[counter] = [mol_tag, mol_name, bead_type, graph, frag_nodes, cgs]
                beads.append([bead_idx,[mol_tag, mol_name, bead_type, graph, frag_nodes, cgs]])
                counter += 1

    return beads, bead_type_dict, molecule_dict, counter

def read_library(data_file, lib_name='prod'):
    """
    Read the library file of fragments and returns a dict
    of unique fragments as well as matching bead_types.
    """
    m3_df = pd.read_excel(data_file, sheet_name=lib_name)
    beads, fragment_dict, bead_type_dict, total = make_fragment_list(m3_df)
    return beads, fragment_dict, bead_type_dict, total

def create_bead_drawing(ax, bead_info, molecule_dict, idx):
    """Create a sample drawing on the given axes"""
    mol_tag, mol_name, bead_type, graph, frag_nodes, cgs = bead_info[1]
    mol_graph = molecule_dict[(mol_name, mol_tag)]
    if len(mol_graph) < 25:
        graph = mol_graph
        draw_mapping = True
        colors = {}
        for node, fragid in mol_graph.nodes(data='fragid'):
            colors[fragid[0]] = 'gray'
            if fragid in frag_nodes:
                colors[fragid[0]] = 'tab:blue'
    else:
        draw_mapping = False
        colors = {node: 'orange' for node in graph.nodes}

    labels = {}
    for node in graph.nodes:
        ele = graph.nodes[node]['element']
        hcount = graph.nodes[node]['hcount']
        if hcount:
            labels[node] = f'{ele}H{hcount}'
        else:
            labels[node] = ele
    ax, pos = draw_molecule(graph,
                            ax=ax,
                            colors=colors,
                            cg_mapping=draw_mapping,
                            labels=labels,
                            scale=0.55,
                            align_with='x',
                            layout_kwargs={'target_energy': 10000})

    ax.text(0.2, 0.8, str(idx),
        transform=ax.transAxes,
        #rotation=0,
        verticalalignment='center',
        horizontalalignment='center',
        fontsize=10,
        fontweight='bold')

    ax.text(0.2, 0.5, bead_type,
        transform=ax.transAxes,
        #rotation=0,
        verticalalignment='center',
        horizontalalignment='center',
        fontsize=10,
        fontweight='bold')

    if len(mol_name) > 30:
        mol_name = mol_tag

    ax.text(0.2, 0.65, mol_name,
        transform=ax.transAxes,
        #rotation=0,
        verticalalignment='center',
        horizontalalignment='center',
        fontsize=10,
        fontweight='bold')

    wrapped_text = '\n'.join(textwrap.wrap(cgs, width=40))
    ax.text(0.8, 0.5, wrapped_text,
        transform=ax.transAxes,
        #rotation=0,
        verticalalignment='center',
        horizontalalignment='center',
        fontsize=10,
        fontweight='bold')

    #ax.set_title(bead_type, fontsize=10, fontweight='bold')
    ax.set_aspect('equal')

def create_drawings_table_pdf(bead_type_dict, molecule_dict, total, filename='drawings_table.pdf',
                            rows_per_page=3, cols_per_page=3):
    """
    Create a multi-page PDF with a table layout of drawings

    Parameters:
    - drawings_data: list of dictionaries with 'title' and optionally 'data'
    - filename: output PDF filename
    - rows_per_page: number of rows per page
    - cols_per_page: number of columns per page
    """

    drawings_per_page = rows_per_page * cols_per_page
    total_pages = (total + drawings_per_page - 1) // drawings_per_page

    with PdfPages(filename) as pdf:
        for page_num in range(total_pages):
            fig, axes = plt.subplots(rows_per_page,
                                     cols_per_page,
                                    figsize=(11, 8.5))  # Letter size
            # Flatten axes array for easier indexing
            if rows_per_page == 1 and cols_per_page == 1:
                axes = [axes]
            elif rows_per_page == 1 or cols_per_page == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()

            # Calculate range of drawings for this page
            start_idx = page_num * drawings_per_page
            end_idx = min(start_idx + drawings_per_page, len(bead_type_dict))

            # Create drawings for this page
            for i in range(drawings_per_page):
                drawing_idx = start_idx + i

                if drawing_idx < len(bead_type_dict):
                    # Create actual drawing
                    drawing = bead_type_dict[drawing_idx]
                    create_bead_drawing(axes[i], drawing, molecule_dict, drawing_idx)
                else:
                    # Hide unused subplots
                    axes[i].set_visible(False)

            # Add page title
            fig.suptitle(f'Martini Bible - Page {page_num + 1} of {total_pages}', 
                        fontsize=14, fontweight='bold', y=0.95)

            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(top=0.90, hspace=0.3, wspace=0.3)

            # Save page to PDF
            pdf.savefig(fig, dpi=150, bbox_inches='tight')
            plt.close(fig)

    print(f"PDF saved as: {filename}")

# Example usage
if __name__ == "__main__":
    data_file = 'MartiniCGSmilesDB_v6.xlsx'
    beads, bead_dict, molecule_dict, nfrags = read_library(data_file, lib_name='All')
    sorted_data = sorted(beads, key=lambda x: x[0])
    # Create the PDF with 3x3 grid (9 drawings per page)
    create_drawings_table_pdf(sorted_data,
                              molecule_dict,
                              total=nfrags,
                              filename='martini_bibile.pdf',
                              rows_per_page=5, cols_per_page=1)
