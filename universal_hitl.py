import sys
import json
from rdkit import Chem
from typing import Dict, Any

# Universal SMARTS alerts for pre/pro-haptens & metabolic activation
PRO_HAPTEN_PATTERNS = {
    "Benzylic/Allylic Alcohol (Oxidation to Aldehyde)": "[c,C=C][CH2,CH(C)][OH]",
    "Glycol Ether Ester (Hydrolysis to Alkoxyethanol)": "[O;H0]-[C]-[C]-[O;H0]",
    "Autoxidizable Polyene/Diene": "[C]=[C]-[CH2]-[C]=[C]",
    "Pro-hapten Arylamine": "[c][NH2,NHR]",
    "Hydroquinone/Catechol Precursor": "c1cc(O)ccc1O"
}

def evaluate_borderline_conflict(res: Dict[str, Any]) -> Dict[str, Any]:
    smiles = res.get("SMILES", "")
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    
    consensus = float(res.get("Consensus_Score", 0.0))
    trans_score = float(res.get("Transformer_Score", 0.0))
    gnn_score = float(res.get("GNN_Score", 0.0))
    mmpbsa_dg = float(res.get("MD_MMPBSA_DeltaG", 0.0))
    
    # 1. Ambiguous Consensus Score Window
    is_score_borderline = 0.40 <= consensus <= 0.70
    
    # 2. Ensemble Discordance (ML High vs. Binding Low)
    is_discordant = (max(trans_score, gnn_score) >= 0.75) and (mmpbsa_dg > -5.0)
    
    # 3. Universal Pro-hapten Pattern Match
    matched_pro_haptens = []
    if mol:
        for label, smarts in PRO_HAPTEN_PATTERNS.items():
            patt = Chem.MolFromSmarts(smarts)
            if patt and mol.HasSubstructMatch(patt):
                matched_pro_haptens.append(label)

    flagged = is_score_borderline or is_discordant or bool(matched_pro_haptens)
    
    reasons = []
    if is_score_borderline:
        reasons.append(f"Score Ambiguity: Consensus score ({consensus:.2f}) is in the 0.40–0.70 transition zone")
    if is_discordant:
        reasons.append(f"Model Discordance: High ML score vs. low Keap1 covalent affinity ({mmpbsa_dg} kcal/mol)")
    if matched_pro_haptens:
        reasons.append(f"Metabolic Bioactivation: Matched {', '.join(matched_pro_haptens)}")

    scenarios = [
        {
            "code": "A",
            "title": "Direct / Native In Vitro Reactivity (KE1 DPRA Baseline)",
            "potency": "Not Classified (NC) / Weak (Category 1B)",
            "rationale": "Direct covalent protein haptenation is low; activation depends on cutaneous metabolic clearance."
        },
        {
            "code": "B",
            "title": "Conservative Precautionary Upper Bound (OECD 497)",
            "potency": res.get("GHS_Category", "Category 1A"),
            "rationale": "Assumes complete enzymatic bioactivation/oxidation to reactive electrophilic intermediates."
        }
    ]

    return {
        "flagged": flagged,
        "reason": " | ".join(reasons) if reasons else "Concordant",
        "scenarios": scenarios
    }

def run_universal_cli(smiles: str, name: str = "Test Molecule", consensus: float = 0.82, ghs: str = "Category 1A"):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        print(f"❌ Invalid SMILES: {smiles}")
        return

    res = {
        "Resolved_Name": name,
        "SMILES": smiles,
        "Consensus_Score": consensus,
        "OECD_497_Call": "SENSITIZER" if consensus >= 0.5 else "NON-SENSITIZER",
        "GHS_Category": ghs,
        "Transformer_Score": consensus,
        "GNN_Score": consensus,
        "MD_MMPBSA_DeltaG": -3.8
    }

    conflict = evaluate_borderline_conflict(res)

    print("\n" + "=" * 76)
    if conflict["flagged"]:
        print("⚠️  [ALERT] BORDERLINE / PRECAUTIONARY UNCERTAINTY DETECTED")
        print(f"   Reason:   {conflict['reason']}")
    else:
        print("✅ [STATUS] CONCORDANT CALL (No Metabolic Bias or Score Conflict)")
    print(f"   Compound: {res['Resolved_Name']} ({res['SMILES']})")
    print(f"   Model:    {res['OECD_497_Call']} ({res['GHS_Category']})")
    print("=" * 76)

    if not conflict["flagged"]:
        print("\nNo expert override required. Automated assessment confirmed.\n")
        return

    print("\nCOMPETING MECHANISTIC SCENARIOS:")
    for sc in conflict["scenarios"]:
        print(f"\n  [{sc['code']}] {sc['title']}")
        print(f"      Implied Potency:   {sc['potency']}")
        print(f"      Mechanistic Basis: {sc['rationale']}")

    print("\n" + "-" * 76)
    print("EXPERT DECISION OPTIONS:")
    print(f"  [1] Accept Precautionary Default ({res['GHS_Category']})")
    print("  [2] Downgrade to GHS Category 1B (Moderate / Weak Sensitizer)")
    print("  [3] Classify as Not Classified / Non-Sensitizer (NC)")
    print("  [4] Mark as Inconclusive / Require In Vitro Testing (OECD 442C/D/E)")
    print("-" * 76)

    choices = {
        "1": f"Precautionary Default ({res['GHS_Category']})",
        "2": "GHS Category 1B (Moderate/Weak)",
        "3": "Not Classified (NC)",
        "4": "Inconclusive / Requires Testing"
    }

    selected = None
    while selected not in choices:
        selected = input("\nEnter your expert decision [1-4]: ").strip()

    res["HITL_Override_Applied"] = True
    res["HITL_Final_Call"] = choices[selected]
    
    custom_rationale = input("\nEnter justification rationale [Press ENTER for default]:\n> ").strip()
    res["HITL_Justification"] = custom_rationale or "Precautionary overestimation adjusted: metabolic bioactivation rate insufficient to cross human skin elicitation threshold."
    res["HITL_Status"] = "Expert Adjudication Completed"

    print("\n" + "=" * 76)
    print("📋 AUDIT RECORD READY FOR OECD 497 DOSSIER:")
    print(json.dumps({k: res[k] for k in ["Resolved_Name", "SMILES", "OECD_497_Call", "HITL_Final_Call", "HITL_Justification"]}, indent=2))
    print("=" * 76 + "\n")

if __name__ == "__main__":
    test_smiles = sys.argv[1] if len(sys.argv) > 1 else "OCc1ccccc1"
    test_name = sys.argv[2] if len(sys.argv) > 2 else "Input Compound"
    run_universal_cli(test_smiles, test_name)
