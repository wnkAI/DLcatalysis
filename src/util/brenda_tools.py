import json
import csv
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

_LARGEST_FRAG = rdMolStandardize.LargestFragmentChooser()


def _desalt_smiles(smi: str) -> str:
    """Remove counter-ions (Na+, K+, Cl-, etc.) but keep physiological charges.
    [Na+].OC(=O)c1ccccc1 → OC(=O)c1ccccc1
    [Mg+2].[ATP SMILES] → [ATP SMILES] (keep largest fragment)
    ATP^4- stays charged (biologically relevant at pH 7)
    """
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return smi
        mol = _LARGEST_FRAG.choose(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return smi


def gen_id(idx: int, prefix: str) -> str:
    if idx < 0 or idx > 1000000:
            raise ValueError('idx must be between 0 and 1000000')
    return "{}{:07d}".format(prefix, idx)

def process_brenda_json(json_fp: str, data_out_fp: str, seq_out_fp: str, smi_out_fp: str) -> None:

    with open(json_fp, 'r', encoding='utf-8') as f:
        data = json.load(f)

    smis = {}
    seqs = {}

    smi_count = 0
    seq_cont = 0

    for i_dct in data:
        if (i_seq:=i_dct["sequence"]) not in seqs.keys():
            seqs[i_seq] = [gen_id(seq_cont, "SEQ"), 1]
            seq_cont += 1
        else:
            seqs[i_seq][-1] += 1

        if (i_smi:=_desalt_smiles(i_dct["smiles"])) not in smis.keys():
            smis[i_smi] = [gen_id(smi_count, "SMI"), 1]
            smi_count += 1
        else:
            smis[i_smi][-1] += 1

    with open(data_out_fp, "w+", newline='') as F:
        writer = csv.writer(F)
        writer.writerow(["DATA_ID", "SEQ_ID", "SMI_ID", "EC_NUMBER", "Y_VALUE", "KCAT_VALUE", "KM_VALUE"])
        for i_dic in data:
            kcat_raw = i_dic.get("kcat", "")
            km_raw = i_dic.get("Km", "")
            # Guard: if kcat/Km is a list (multi-value), take first; if not scalar, leave empty
            if isinstance(kcat_raw, list):
                kcat_raw = kcat_raw[0] if kcat_raw else ""
            if isinstance(km_raw, list):
                km_raw = km_raw[0] if km_raw else ""
            writer.writerow([
                i_dic["id"],
                seqs[i_dic["sequence"]][0],
                smis[_desalt_smiles(i_dic["smiles"])][0],
                i_dic["ec"],
                i_dic["value"],
                kcat_raw,
                km_raw,
            ])
    with open(seq_out_fp, "w+", newline='') as F:
        writer = csv.writer(F)
        writer.writerow(["SEQ_ID", "SEQUENCE"])
        for i_seq, (i_id, _) in seqs.items():
            writer.writerow([i_id, i_seq])
    with open(smi_out_fp, "w+", newline='') as F:
        writer = csv.writer(F)
        writer.writerow(["SMI_ID", "SMILES"])
        for i_smi, (i_id, _) in smis.items():
            writer.writerow([i_id, i_smi])
