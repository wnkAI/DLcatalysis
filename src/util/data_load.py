import sys
from pathlib import Path
from typing import Any, List, Tuple, Dict
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from util.constants import AA_NAME_SYM, BACKBONE_NAMES
from util.tools import set_seed

import pytorch_lightning as pl
from torch_geometric.transforms import Compose
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch_geometric
import torch
#from torch_scatter import scatter

import numpy as np
import pandas as pd
from typing import Dict
from rdkit import Chem
from rdkit.Chem.rdchem import BondType, HybridizationType
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig

#++++++++++++++++++++++++++++++++++++++++++++++# README #+++++++++++++++++++++++++++++++++++++++++#   
# StructureComplexData: Pocket结构数据点(包含蛋白质部分以及配体部分)， 该数据结构直接被储存在lmdb中
# SequenceData: 蛋白质的序列信息数据点， 该数据结构直接被储存在lmdb中
# PDBProtein: 用于解析PDB文件, 提取蛋白质特征
# get_ligand_atom_features: 提取配体原子的特征
# parse_sdf_file_mol: 解析sdf文件， 提取配体分子特征
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++# 


class SequenceData(torch_geometric.data.Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_dict(seq_dict: Dict):
        instance = SequenceData()
        """"
        实际上这个地方只储存一个embedding张量
        """

        for key, item in seq_dict.items():
            instance[key] = item

        return instance
    
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return self['embedding'].size(0)
        else:
            return super().__inc__(key, value)

def padding_seq_embedding(seq_data: Dict, max_len: int) -> Dict:
    """
    对序列embedding进行padding填充
    seq_data: embedding, sequence
    """
    # 获取embedding
    embedding = seq_data["embedding"]
    
    # 处理torch Tensor
    if hasattr(embedding, 'cpu'):  # 是torch Tensor
        # 重要：先detach()再cpu()再numpy()
        embedding = embedding.detach().cpu().numpy()
    elif not isinstance(embedding, np.ndarray):
        embedding = np.array(embedding)
    
    cur_len = embedding.shape[0]
    
    # 检查是否需要截断
    if cur_len > max_len:
        print(f"Warning: Sequence length {cur_len} > max_len {max_len}, truncating")
        embedding = embedding[:max_len, :]
        cur_len = max_len
    
    # 创建padding mask (1表示padding位置)
    seq_data['seq_padding_mask'] = np.ones((1, max_len), dtype=bool)
    seq_data['seq_padding_mask'][0, :cur_len] = False
    
    # padding embedding
    if cur_len < max_len:
        seq_data['embedding'] = np.pad(
            embedding,
            ((0, max_len - cur_len), (0, 0)),
            'constant',
            constant_values=0
        )
    else:
        seq_data['embedding'] = embedding
    
    return seq_data




class StructureComplexData(Data):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
    @staticmethod
    def from_protein_ligand_dicts(protein_dict=None, ligand_dict=None, **kwargs) -> Any:
        instance = StructureComplexData(**kwargs)

        if protein_dict is not None:
            for key, item in protein_dict.items():
                instance['protein_' + key] = item

        if ligand_dict is not None:
            for key, item in ligand_dict.items():
                instance['ligand_' + key] = item
        
        return instance

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'complex_edge_index':
            return self['mask_ligand'].size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

def padding_struct_lig_embedding(ligand_data: Dict, max_ligand_atoms: int) ->Dict:
    # StructureComplexData的时候会添加
    ligand_data["padding_mask"] = np.zeros((max_ligand_atoms), dtype=bool)[None, ...]
    ligand_data['padding_mask'][0, ligand_data['element'].shape[0]:] = True

    return ligand_data





class PDBProtein(object):

    
    AA_NAME_NUMBER = {
        k: i for i, (k, _) in enumerate(AA_NAME_SYM.items())
    }

    def __init__(self, data, mode='auto'):
        super().__init__()
        self.BACKBONE_NAMES = BACKBONE_NAMES
        self.fn = data
        if (data[-4:].lower() == '.pdb' and mode == 'auto') or mode == 'path':
            with open(data, 'r') as f:
                self.block = f.read()
        else:
            self.block = data

        self.ptable = Chem.GetPeriodicTable()

        # Molecule properties
        self.title = None
        # Atom properties
        self.atoms = []
        self.element = []
        self.atomic_weight = []
        self.pos = []
        self.atom_name = []
        self.is_backbone = []
        self.atom_to_aa_type = []
        # Residue properties
        self.residues = []
        self.amino_acid = []
        self.center_of_mass = []
        self.pos_CA = []
        self.pos_C = []
        self.pos_N = []
        self.pos_O = []

        self._parse()

    def _enum_formatted_atom_lines(self):
        for line in self.block.splitlines():
            if line[0:6].strip() == 'ATOM':
                element_symb = line[76:78].strip().capitalize()
                if len(element_symb) == 0:
                    element_symb = line[13:14]
                yield {
                    'line': line,
                    'type': 'ATOM',
                    'atom_id': int(line[6:11]),
                    'atom_name': line[12:16].strip(),
                    'res_name': line[17:20].strip(),
                    'chain': line[21:22].strip(),
                    'res_id': int(line[22:26]),
                    'res_insert_id': line[26:27].strip(),
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54]),
                    'occupancy': float(line[54:60]),
                    'segment': line[72:76].strip(),
                    'element_symb': element_symb,
                    'charge': line[78:80].strip(),
                }
            elif line[0:6].strip() == 'HEADER':
                yield {
                    'type': 'HEADER',
                    'value': line[10:].strip()
                }
            elif line[0:6].strip() == 'ENDMDL':
                break   # Some PDBs have more than 1 model.

    def _parse(self):
        # Process atoms
        residues_tmp = {}
        for atom in self._enum_formatted_atom_lines():
            if atom['type'] == 'HEADER':
                self.title = atom['value'].lower()
                continue
            self.atoms.append(atom)
            atomic_number = self.ptable.GetAtomicNumber(atom['element_symb'])
            next_ptr = len(self.element)
            self.element.append(atomic_number)
            self.atomic_weight.append(self.ptable.GetAtomicWeight(atomic_number))
            self.pos.append(np.array([atom['x'], atom['y'], atom['z']], dtype=np.float32))
            self.atom_name.append(atom['atom_name'])
            self.is_backbone.append(atom['atom_name'] in self.BACKBONE_NAMES)
            if atom['res_name'] not in self.AA_NAME_NUMBER:
                self.atom_to_aa_type.append(self.AA_NAME_NUMBER['UNK'])
            else:
                self.atom_to_aa_type.append(self.AA_NAME_NUMBER[atom['res_name']])

            chain_res_id = '%s_%s_%d_%s' % (atom['chain'], atom['segment'], atom['res_id'], atom['res_insert_id'])
            if chain_res_id not in residues_tmp:
                residues_tmp[chain_res_id] = {
                    'name': atom['res_name'],
                    'atoms': [next_ptr],
                    'chain': atom['chain'],
                    'segment': atom['segment'],
                }
            else:
                assert residues_tmp[chain_res_id]['name'] == atom['res_name']
                assert residues_tmp[chain_res_id]['chain'] == atom['chain']
                residues_tmp[chain_res_id]['atoms'].append(next_ptr)

        # Process residues
        self.residues = [r for _, r in residues_tmp.items()]
        for residue in self.residues:
            sum_pos = np.zeros([3], dtype=np.float32)
            sum_mass = 0.0
            for atom_idx in residue['atoms']:
                sum_pos += self.pos[atom_idx] * self.atomic_weight[atom_idx]
                sum_mass += self.atomic_weight[atom_idx]
                if self.atom_name[atom_idx] in self.BACKBONE_NAMES:
                    residue['pos_%s' % self.atom_name[atom_idx]] = self.pos[atom_idx]
            residue['center_of_mass'] = sum_pos / sum_mass
        
        # Process backbone atoms of residues
        for residue in self.residues:
            if residue['name'] not in self.AA_NAME_NUMBER:
                self.amino_acid.append(self.AA_NAME_NUMBER['UNK'])
            else:
                self.amino_acid.append(self.AA_NAME_NUMBER[residue['name']])
            self.center_of_mass.append(residue['center_of_mass'])
            for name in self.BACKBONE_NAMES:
                pos_key = 'pos_%s' % name   # pos_CA, pos_C, pos_N, pos_O
                if pos_key in residue:
                    getattr(self, pos_key).append(residue[pos_key])
                else:
                    getattr(self, pos_key).append(residue['center_of_mass'])

    def to_dict_atom(self):
        return {
            'element': np.array(self.element, dtype=int),
            'molecule_name': self.title,
            'pos': np.array(self.pos, dtype=np.float32),
            'is_backbone': np.array(self.is_backbone, dtype=bool),
            'atom_name': self.atom_name,
            'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=int)
        }

    def to_dict_residue(self):
        return {
            'amino_acid': np.array(self.amino_acid, dtype=int),
            'center_of_mass': np.array(self.center_of_mass, dtype=np.float32),
            'pos_CA': np.array(self.pos_CA, dtype=np.float32),
            'pos_C': np.array(self.pos_C, dtype=np.float32),
            'pos_N': np.array(self.pos_N, dtype=np.float32),
            'pos_O': np.array(self.pos_O, dtype=np.float32),
        }

    def query_residues_radius(self, center, radius, criterion='center_of_mass'):
        center = np.array(center).reshape(3)
        selected = []
        for residue in self.residues:
            distance = np.linalg.norm(residue[criterion] - center, ord=2)
            print(residue[criterion], distance)
            if distance < radius:
                selected.append(residue)
        return selected

    def query_residues_ligand(self, ligand, radius, criterion='center_of_mass'):
        selected = []
        sel_idx = set()
        # The time-complexity is O(mn).
        for center in ligand['pos']:
            for i, residue in enumerate(self.residues):
                distance = np.linalg.norm(residue[criterion] - center, ord=2)
                if distance < radius and i not in sel_idx:
                    selected.append(residue)
                    sel_idx.add(i)
        return selected

    def residues_to_pdb_block(self, residues, name='POCKET'):
        block = "HEADER    %s\n" % name
        block += "COMPND    %s\n" % name
        for residue in residues:
            for atom_idx in residue['atoms']:
                block += self.atoms[atom_idx]['line'] + "\n"
        block += "END\n"
        return block

def get_ligand_atom_features(rdmol):
    """Extract atom features from RDKit molecule"""
    num_atoms = rdmol.GetNumAtoms()
    atomic_number = []
    aromatic = []
    hybrid = []
    degree = []
    
    for atom_idx in range(num_atoms):
        atom = rdmol.GetAtomWithIdx(atom_idx)
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization = atom.GetHybridization()
        HYBRID_TYPES = {t: i for i, t in enumerate(HybridizationType.names.values())}
        hybrid.append(HYBRID_TYPES[hybridization])
        degree.append(atom.GetDegree())
    
    # 创建节点类型张量
    node_type = torch.tensor(atomic_number, dtype=torch.long)

    # 构建边索引
    row, col = [], []
    for bond in rdmol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
    
    # 计算氢原子数量
    hs = (node_type == 1).to(torch.float)
    num_hs = torch.zeros(num_atoms, dtype=torch.float)
    if len(row) > 0:  # 避免空图
        row_tensor = torch.tensor(row, dtype=torch.long)
        col_tensor = torch.tensor(col, dtype=torch.long)
        for i in range(len(row_tensor)):
            num_hs[col_tensor[i]] += hs[row_tensor[i]]

    # 创建特征矩阵
    feat_mat = np.array([atomic_number, aromatic, degree, num_hs.numpy(), hybrid], dtype=int).transpose()
    return feat_mat

# used for preparing the dataset
def parse_sdf_file_mol(path, mol=None, heavy_only=True):

    if mol is None:
        mol = next(iter(Chem.SDMolSupplier(path, removeHs=heavy_only, sanitize=False)))
    feat_mat = get_ligand_atom_features(mol)

    # fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    # factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    # rdmol = next(iter(Chem.SDMolSupplier(path, removeHs=heavy_only)))
    # rd_num_atoms = rdmol.GetNumAtoms()
    # feat_mat = np.zeros([rd_num_atoms, len(ATOM_FAMILIES)], dtype=int)
    # for feat in factory.GetFeaturesForMol(rdmol):
    #     feat_mat[feat.GetAtomIds(), ATOM_FAMILIES_ID[feat.GetFamily()]] = 1

    ptable = Chem.GetPeriodicTable()

    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()
    pos = mol.GetConformer().GetPositions()

    element = []
    indexs = []
    accum_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    accum_mass = 0.0
    for atom_idx in range(num_atoms):
        atom = mol.GetAtomWithIdx(atom_idx)
        atomic_number = atom.GetAtomicNum()
        element.append(atomic_number)
        # 只有一个分子，我们直接用atom_idx
        indexs.append(atom_idx)
        x, y, z = pos[atom_idx]
        atomic_weight = ptable.GetAtomicWeight(atomic_number)
        accum_pos += np.array([x, y, z]) * atomic_weight
        accum_mass += atomic_weight
    center_of_mass = np.array(accum_pos / accum_mass, dtype=np.float32)
    element = np.array(element, dtype=int)
    pos = np.array(pos, dtype=np.float32)
    indexs = np.array(indexs, dtype=int)
    
    row, col, edge_type = [], [], []
    
    BOND_TYPES = {}
    BOND_TYPES[BondType.SINGLE] = 1
    BOND_TYPES[BondType.DOUBLE] = 2
    BOND_TYPES[BondType.TRIPLE] = 3
    BOND_TYPES[BondType.AROMATIC] = 4
    BOND_TYPES[BondType.UNSPECIFIED] = 5

    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [BOND_TYPES[bond.GetBondType()]]
    edge_index = np.array([row, col], dtype=int)
    edge_type = np.array(edge_type, dtype=int)
    perm = (edge_index[0] * num_atoms + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    data = {
        'element': element,
        'pos': pos,
        'bond_index': edge_index,
        'bond_type': edge_type,
        'center_of_mass': center_of_mass,
        'atom_feature': feat_mat,
        'index': indexs
    }
    return data



