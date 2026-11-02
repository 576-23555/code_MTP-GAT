from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, DataStructs
import numpy as np
import torch
from tqdm import tqdm


def calculate_maccs_fingerprints(mols):
    fingerprints = []

    for mol in tqdm(mols, desc="Calculating MACCS fingerprints"):
        if mol:
            fp = AllChem.GetMACCSKeysFingerprint(mol)
            fingerprints.append(fp)
        else:
            fingerprints.append(None)

    return fingerprints


def calculate_pubchem_fingerprints(mols):
    fingerprints = []

    for mol in tqdm(mols, desc="Calculating PubChem fingerprints"):
        if mol:
            fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
                mol, nBits=881
            )
            fingerprints.append(fp)
        else:
            fingerprints.append(None)

    return fingerprints


def calculate_ecfp4_fingerprints(mols):
    fingerprints = []

    for mol in tqdm(mols, desc="Calculating ECFP4 fingerprints"):
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=2, nBits=1024
            )
            fingerprints.append(fp)
        else:
            fingerprints.append(None)

    return fingerprints


def get_longest_alkyl_chain_length(mol):

    if mol is None:
        return 0

    carbon_graph = {}

    for atom in mol.GetAtoms():

        if (
            atom.GetAtomicNum() == 6
            and not atom.GetIsAromatic()
            and not atom.IsInRing()
            and atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3
        ):
            carbon_graph[atom.GetIdx()] = []

    for bond in mol.GetBonds():

        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()

        idx1 = atom1.GetIdx()
        idx2 = atom2.GetIdx()

        if (
            idx1 in carbon_graph
            and idx2 in carbon_graph
            and bond.GetBondType() == Chem.rdchem.BondType.SINGLE
        ):
            carbon_graph[idx1].append(idx2)
            carbon_graph[idx2].append(idx1)

    if not carbon_graph:
        return 0

    def bfs_farthest(start, component):

        distances = {start: 0}
        queue = [start]

        for node in queue:

            for neighbor in carbon_graph[node]:

                if neighbor in component and neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)

        farthest_node = max(distances, key=distances.get)

        return farthest_node, distances[farthest_node]

    visited = set()
    max_chain_length = 0

    for start in carbon_graph:

        if start in visited:
            continue

        component = set()
        stack = [start]

        while stack:

            node = stack.pop()

            if node in component:
                continue

            component.add(node)
            visited.add(node)

            stack.extend(
                neighbor
                for neighbor in carbon_graph[node]
                if neighbor not in component
            )

        endpoint, _ = bfs_farthest(start, component)
        _, max_distance = bfs_farthest(endpoint, component)

        chain_length = max_distance + 1

        max_chain_length = max(
            max_chain_length,
            chain_length
        )

    return max_chain_length


def calculate_tanimoto_similarity_matrix(fps):

    n = len(fps)
    sim_mat = np.zeros((n, n))

    for i in range(n):

        for j in range(i, n):

            if fps[i] is None or fps[j] is None:
                sim = 0.0
            else:
                sim = DataStructs.TanimotoSimilarity(
                    fps[i], fps[j]
                )

            sim_mat[i, j] = sim
            sim_mat[j, i] = sim

    return sim_mat


def calculate_penalized_similarity(
    fp1,
    fp2,
    mol1,
    mol2,
    penalty_strength=0.1
):

    if fp1 is None or fp2 is None:
        return 0.0

    base_sim = DataStructs.FingerprintSimilarity(
        fp1, fp2
    )

    chain_len1 = get_longest_alkyl_chain_length(mol1)
    chain_len2 = get_longest_alkyl_chain_length(mol2)

    delta_L = abs(chain_len1 - chain_len2)

    penalty = penalty_strength * np.log(1 + delta_L)

    adjusted_sim = base_sim * np.exp(-penalty)

    return adjusted_sim


def calculate_penalized_similarity_matrix(
    mols,
    fps,
    penalty_strength=0.1
):

    n = len(mols)
    sim_mat = np.zeros((n, n))

    for i in tqdm(
        range(n),
        desc="Calculating penalized similarity"
    ):

        for j in range(i, n):

            sim = calculate_penalized_similarity(
                fps[i],
                fps[j],
                mols[i],
                mols[j],
                penalty_strength=penalty_strength
            )

            sim_mat[i, j] = sim
            sim_mat[j, i] = sim

    return sim_mat


def compute_all_acs_matrices_with_penalty(
    mols,
    penalty_strength=0.1
):

    maccs_fps = calculate_maccs_fingerprints(mols)
    pubchem_fps = calculate_pubchem_fingerprints(mols)
    ecfp4_fps = calculate_ecfp4_fingerprints(mols)

    sim_maccs = calculate_penalized_similarity_matrix(
        mols,
        maccs_fps,
        penalty_strength
    )

    sim_pubchem = calculate_penalized_similarity_matrix(
        mols,
        pubchem_fps,
        penalty_strength
    )

    sim_ecfp4 = calculate_penalized_similarity_matrix(
        mols,
        ecfp4_fps,
        penalty_strength
    )

    acs_matrices = [
        torch.tensor(sim_pubchem, dtype=torch.float32),
        torch.tensor(sim_maccs, dtype=torch.float32),
        torch.tensor(sim_ecfp4, dtype=torch.float32)
    ]

    return acs_matrices


def select_neg_acs_matrices(
    acs_matrices,
    batch_size,
    d_an_local_size,
    num_explicit_neg
):

    selected_matrices = []

    for sim in acs_matrices:

        sim = sim.clone()
        sim.fill_diagonal_(0.0)

        sim_local = (
            sim[:, :d_an_local_size]
            if d_an_local_size > 0
            else torch.empty(
                (batch_size, 0),
                device=sim.device
            )
        )

        sim_inter = []

        for i in range(batch_size):

            inter_row = torch.cat(
                [sim[i, :i], sim[i, i + 1:]],
                dim=0
            )

            sim_inter.append(inter_row)

        sim_inter = torch.stack(
            sim_inter,
            dim=0
        )

        sim_selected = torch.cat(
            [sim_local, sim_inter],
            dim=1
        )

        selected_matrices.append(sim_selected)

    return selected_matrices