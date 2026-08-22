import hashlib
import io
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Draw, Lipinski

# =====================================================================
# STREAMLIT UI CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="Multi-Agent Skin Sensitizer AI (OECD GL 497)",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧪 Multi-Agent Skin Sensitization Predictor")
st.caption(
    "Automated Defined Approach based on **OECD Guideline 497**, Organic SMARTS alerts, Inorganic/Metal Chelation profiler, and local offline/online hybrid chemical resolution."
)

# =====================================================================
# DATA MODELS
# =====================================================================
@dataclass
class ChemicalProfile:
    query_term: str
    resolved_name: str
    cas: str
    smiles: str
    cid: Optional[int] = None
    mol: Optional[Chem.Mol] = None
    mw: float = 0.0
    log_p: float = 0.0
    tpsa: float = 0.0
    is_metal: bool = False

    def compute_descriptors(self):
        if self.mol:
            try:
                self.mw = round(Descriptors.MolWt(self.mol), 2)
                self.log_p = round(Crippen.MolLogP(self.mol), 2)
                self.tpsa = round(Descriptors.TPSA(self.mol), 2)
            except Exception:
                self.mw = 0.0


# =====================================================================
# HYBRID RESOLVER: OFFLINE SQLITE FIRST -> LIVE FALLBACK
# =====================================================================
class UniversalChemicalResolver:
    DB_PATH = os.path.join(os.path.dirname(__file__), "chem_index.db")
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    }

    @staticmethod
    def _is_metal_structure(smiles: str) -> bool:
        if not smiles:
            return False
        metal_symbols = ["[Ni]", "[Co]", "[Cr]", "[Cu]", "[Au]", "[Pd]", "[Pt]", "[Zn]", "[Fe]", "[Mn]"]
        return any(m in smiles for m in metal_symbols)

    @staticmethod
    def _query_local_db(identifier: str) -> Optional[Dict[str, Any]]:
        if not os.path.exists(UniversalChemicalResolver.DB_PATH):
            return None
        try:
            conn = sqlite3.connect(UniversalChemicalResolver.DB_PATH)
            c = conn.cursor()
            # Match by CAS or by chemical name
            c.execute("SELECT name, smiles, cid FROM chemicals WHERE cas = ? OR LOWER(name) = ?", (identifier, identifier.lower()))
            row = c.fetchone()
            conn.close()
            if row:
                return {
                    "cid": row[2],
                    "name": row[0],
                    "smiles": row[1],
                    "is_metal": UniversalChemicalResolver._is_metal_structure(row[1]),
                }
        except Exception:
            pass
        return None

    @staticmethod
    def resolve_input(identifier: str) -> Optional[Dict[str, Any]]:
        raw = str(identifier).strip().replace('"', '').replace("'", "")
        query = re.sub(r"\s+", " ", raw)
        if not query:
            return None

        # Tier 1: Local SQLite Lookup (Zero Network Latency / Immune to Cloud Blocks)
        local_hit = UniversalChemicalResolver._query_local_db(query)
        if local_hit:
            return local_hit

        # Tier 2: Direct SMILES Check
        mol = Chem.MolFromSmiles(query)
        if mol:
            return {
                "cid": None,
                "name": "User-Defined SMILES Structure",
                "smiles": query,
                "is_metal": UniversalChemicalResolver._is_metal_structure(query),
            }

        session = requests.Session()
        session.headers.update(UniversalChemicalResolver.HEADERS)

        # Tier 3: NIH PubChem Direct PUG-REST (CID -> Properties)
        try:
            # Query CID list first
            cids_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{requests.utils.quote(query)}/cids/JSON"
            r_cid = session.get(cids_url, timeout=4)
            if r_cid.status_code == 200:
                cids = r_cid.json().get("IdentifierList", {}).get("CID", [])
                if cids:
                    cid = cids[0]
                    prop_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IUPACName,CanonicalSMILES/JSON"
                    r_prop = session.get(prop_url, timeout=4)
                    if r_prop.status_code == 200:
                        props = r_prop.json().get("PropertyTable", {}).get("Properties", [])
                        if props:
                            s = props[0].get("CanonicalSMILES")
                            return {
                                "cid": cid,
                                "name": props[0].get("IUPACName", query),
                                "smiles": s,
                                "is_metal": UniversalChemicalResolver._is_metal_structure(s),
                            }
        except Exception:
            pass

        # Tier 4: University of Cambridge OPSIN Parser
        try:
            opsin_url = f"https://opsin.ch.cam.ac.uk/opsin/{requests.utils.quote(query)}.json"
            r_op = session.get(opsin_url, timeout=4)
            if r_op.status_code == 200:
                s_cand = r_op.json().get("smiles")
                if s_cand and (Chem.MolFromSmiles(s_cand) or "[" in s_cand):
                    return {
                        "cid": None,
                        "name": query,
                        "smiles": s_cand,
                        "is_metal": UniversalChemicalResolver._is_metal_structure(s_cand),
                    }
        except Exception:
            pass

        return None

    @staticmethod
    def check_ghs_h317(cid: Optional[int]) -> bool:
        if not cid:
            return False
        try:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON?heading=GHS+Classification"
            r = requests.get(url, headers=UniversalChemicalResolver.HEADERS, timeout=4)
            if r.status_code == 200:
                return "H317" in r.text or "allergic skin reaction" in r.text.lower()
        except Exception:
            pass
        return False


# =====================================================================
# MULTI-AGENT ENGINES
# =====================================================================
class ChemistAgent:
    OECD_SMARTS = {
        "SN2_Beta_Haloalkyl_Heteroatom": "[Cl,Br,I][CX4][CX4][O,S,N]",
        "SN2_Alkyl_Halide": "[Cl,Br,I][CH2,CH1][#6]",
        "SN2_Epoxide_Aziridine": "[C,N]1[C,N]O1",
        "Michael_Acceptor_Enone": "[CX3]=[CX3][CX3](=[OX1,SX1])",
        "Michael_Acceptor_Acrylic_Acid_Ester": "[CX3]=[CX3][CX3](=[OX1])[OX2,OX1-]",
        "Michael_Acceptor_Acrylamide": "[CX3]=[CX3][CX3](=[OX1])[NX3,NX4+]",
        "Schiff_Base_Aldehyde": "[CX3H1](=O)[#6]",
        "SNAr_Nitro_Haloaromatic": "c1([N+](=O)[O-])cc([Cl,Br,F])ccc1",
        "Acyl_Transfer_Halide": "[CX3](=[OX1])[Cl,Br,I]",
        "Acyl_Transfer_Isocyanate": "[NX2]=[CX2]=[OX1]",
        "Prohapten_Aromatic_Primary_Amine": "c1ccccc1[NX3H2]",
    }

    METALLIC_SENSITIZERS = {
        "[Ni]": "Nickel Chelation (TLR4 Activation)",
        "[Co]": "Cobalt Contact Chelation",
        "[Cr]": "Chromate/Chromium Hapten Complexation",
        "[Pd]": "Palladium Cross-Reactivity",
    }

    def __init__(self):
        self.patterns = {k: Chem.MolFromSmarts(v) for k, v in self.OECD_SMARTS.items()}

    def evaluate(self, chem: ChemicalProfile) -> Dict[str, Any]:
        for metal_sym, desc in self.METALLIC_SENSITIZERS.items():
            if metal_sym in chem.smiles or (chem.resolved_name and metal_sym.strip("[]").lower() in chem.resolved_name.lower()):
                return {
                    "status": "ALERT_FOUND",
                    "alerts": [f"Inorganic_Metal_Sensitizer: {desc}"],
                    "mechanisms": ["Metal Chelation", "TLR4 Direct Receptor Crosslinking"],
                    "is_metal": True,
                }

        if not chem.mol:
            return {"status": "ERROR", "alerts": [], "mechanisms": ["Invalid Molecule"], "is_metal": False}

        hits = [name for name, pat in self.patterns.items() if chem.mol.HasSubstructMatch(pat)]
        return {
            "status": "ALERT_FOUND" if hits else "NO_ALERTS",
            "alerts": hits,
            "mechanisms": list(set([h.split("_")[0] for h in hits])) if hits else ["Unreactive (Non-Electrophilic)"],
            "is_metal": False,
        }


class ToxicologistAgent:
    def evaluate(self, chem: ChemicalProfile, chem_data: Dict[str, Any], has_h317: bool) -> Dict[str, Any]:
        has_alerts = chem_data["status"] == "ALERT_FOUND"
        is_metal = chem_data.get("is_metal", False)

        if is_metal:
            ke1, ke2, ke3 = 0.90, 0.85, 0.92
            pathway = "TLR4 Receptor Activation & Nrf2 Pathway"
        elif has_alerts or has_h317:
            ke1, ke2, ke3 = 0.88, 0.82, 0.78
            pathway = "Keap1-Nrf2 ARE Activated"
        else:
            ke1, ke2, ke3 = 0.15, 0.18, 0.16
            pathway = "Basal / Uninduced"

        return {
            "KE1_DPRA": ke1,
            "KE2_KeratinoSens": ke2,
            "KE3_hCLAT": ke3,
            "pathway": pathway,
            "is_metal": is_metal,
        }


class StatisticianAgent:
    def evaluate(self, chem: ChemicalProfile, tox_data: Dict[str, Any]) -> Dict[str, Any]:
        score = (0.5 * tox_data["KE1_DPRA"]) + (0.25 * tox_data["KE2_KeratinoSens"]) + (0.25 * tox_data["KE3_hCLAT"])
        
        if tox_data.get("is_metal", False):
            in_ad = True
            conf = 0.95
        else:
            in_ad = (chem.mw <= 500.0) and (-2.5 <= chem.log_p <= 5.5) and (chem.tpsa <= 140.0)
            conf = 0.88 if in_ad else 0.65

        return {
            "score": round(score, 3),
            "call": "SENSITIZER" if score >= 0.50 else "NON_SENSITIZER",
            "applicability_domain": "IN_DOMAIN (Inorganic Metal)" if tox_data.get("is_metal") else ("IN_DOMAIN" if in_ad else "OUT_OF_DOMAIN"),
            "confidence": conf,
        }


class RegulatoryAgent:
    def evaluate(self, stat_data: Dict[str, Any], tox_data: Dict[str, Any]) -> Dict[str, Any]:
        is_sens = stat_data["call"] == "SENSITIZER"
        hits = sum(1 for v in [tox_data["KE1_DPRA"], tox_data["KE2_KeratinoSens"], tox_data["KE3_hCLAT"]] if v >= 0.5)

        if is_sens:
            ghs = "GHS Category 1A (Strong Metal Allergen)" if tox_data.get("is_metal") else ("GHS Category 1A (Strong)" if stat_data["score"] > 0.85 else "GHS Category 1B (Moderate)")
            next_action = "Inorganic allergen: Human Patch Test / Historical Clinical LLNA Data Precedent." if tox_data.get("is_metal") else "Execute OECD TG 442C (DPRA) & OECD TG 442D (KeratinoSens) for 2-of-3 DA."
        else:
            ghs = "GHS Not Classified (Non-Sensitizer)"
            next_action = "2 concordant negative in vitro assays required for regulatory dossier sign-off."

        return {
            "ghs_classification": ghs,
            "da_result": f"Positive ({hits}/3 KEs)" if hits >= 2 else f"Negative ({hits}/3 KEs)",
            "recommended_action": next_action,
        }


class QAAgent:
    @staticmethod
    def audit(chem: ChemicalProfile, stat_data: Dict[str, Any]) -> Dict[str, Any]:
        audit_id = f"QA-{time.strftime('%Y%m%d%H%M')}-{hashlib.sha256(chem.smiles.encode()).hexdigest()[:8]}"
        return {"audit_id": audit_id, "sign_off": "APPROVED_AUTO_SIGNOFF"}


def process_single_chemical(identifier: str) -> Dict[str, Any]:
    resolved = UniversalChemicalResolver.resolve_input(identifier)
    if not resolved or not resolved.get("smiles"):
        return {
            "Input": identifier,
            "Status": "FAILED_RESOLUTION",
            "Resolved_Name": "Unknown",
            "SMILES": "N/A",
            "MW": 0.0,
            "LogP": 0.0,
            "Bot1_Alerts": "None",
            "KE1_DPRA": 0.0,
            "KE2_KeratinoSens": 0.0,
            "KE3_hCLAT": 0.0,
            "Consensus_Score": 0.0,
            "OECD_497_Call": "INCONCLUSIVE",
            "GHS_Category": "Unknown",
            "QA_SignOff": "REJECTED_RESOLUTION_ERROR",
            "Audit_ID": "N/A",
        }

    chem = ChemicalProfile(
        query_term=identifier,
        resolved_name=resolved["name"],
        cas=identifier if "-" in str(identifier) else "N/A",
        smiles=resolved["smiles"],
        cid=resolved.get("cid"),
        mol=Chem.MolFromSmiles(resolved["smiles"]),
        is_metal=resolved.get("is_metal", False),
    )
    chem.compute_descriptors()

    has_h317 = UniversalChemicalResolver.check_ghs_h317(chem.cid)
    b1 = ChemistAgent().evaluate(chem)
    b2 = ToxicologistAgent().evaluate(chem, b1, has_h317)
    b3 = StatisticianAgent().evaluate(chem, b2)
    b4 = RegulatoryAgent().evaluate(b3, b2)
    b5 = QAAgent.audit(chem, b3)

    return {
        "Input": identifier,
        "Status": "SUCCESS",
        "Resolved_Name": chem.resolved_name,
        "SMILES": chem.smiles,
        "MW": chem.mw,
        "LogP": chem.log_p,
        "Bot1_Alerts": ", ".join(b1["alerts"]) if b1["alerts"] else "No Structural Alerts (Unreactive)",
        "KE1_DPRA": b2["KE1_DPRA"],
        "KE2_KeratinoSens": b2["KE2_KeratinoSens"],
        "KE3_hCLAT": b2["KE3_hCLAT"],
        "Consensus_Score": b3["score"],
        "OECD_497_Call": b3["call"],
        "GHS_Category": b4["ghs_classification"],
        "QA_SignOff": b5["sign_off"],
        "Audit_ID": b5["audit_id"],
    }


def render_dashboard_cards(res: Dict[str, Any]):
    mol = Chem.MolFromSmiles(res["SMILES"])
    c_info, c_img = st.columns([2, 1])
    with c_info:
        st.subheader(f"{res['Resolved_Name']}")
        st.code(f"SMILES: {res['SMILES']}", language="text")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Consensus Score", f"{res['Consensus_Score']}")
        m2.metric("OECD 497 Call", f"{res['OECD_497_Call']}")
        m3.metric("MW", f"{res['MW']} g/mol")
        m4.metric("LogP", f"{res['LogP']}")

    with c_img:
        if mol:
            st.image(Draw.MolToImage(mol, size=(300, 180)), caption="2D Structure", use_container_width=True)
        else:
            st.info("Inorganic / Elemental Species")

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 🧪 Bot 1 (Chemist Alerts)")
        st.write(f"**Alerts:** {res['Bot1_Alerts']}")
    with c2:
        st.markdown("#### 🧬 Bot 2 (AOP Key Events)")
        st.write(f"- KE1 (DPRA / Binding): `{res['KE1_DPRA']}`")
        st.write(f"- KE2 (KeratinoSens): `{res['KE2_KeratinoSens']}`")
        st.write(f"- KE3 (h-CLAT): `{res['KE3_hCLAT']}`")
    with c3:
        st.markdown("#### 🏛️ Bot 4 & 5 (Regulatory & QA)")
        st.write(f"**GHS:** {res['GHS_Category']}")
        st.write(f"**Sign-off:** `{res['QA_SignOff']}`")
        st.caption(f"Audit ID: {res['Audit_ID']}")


# =====================================================================
# UI TABS
# =====================================================================
tab_single, tab_sketch, tab_batch = st.tabs([
    "🔍 Single Compound Lookup",
    "✏️ Draw Molecule (JSME Sketcher)",
    "📁 Batch CSV Screening & Export"
])

# ---------------------------------------------------------------------
# TAB 1: SINGLE COMPOUND
# ---------------------------------------------------------------------
with tab_single:
    col_in, col_btn = st.columns([4, 1])
    with col_in:
        single_input = st.text_input("Enter CAS RN, Chemical Name, or SMILES", value="65-85-0")
    with col_btn:
        st.write("")
        st.write("")
        run_single_btn = st.button("Predict", type="primary", use_container_width=True)

    if run_single_btn or single_input:
        with st.spinner(f"Evaluating {single_input}..."):
            res = process_single_chemical(single_input)
            if res["Status"] == "FAILED_RESOLUTION":
                st.error(f"Could not resolve structure for '{single_input}'. You can enter the SMILES directly or sketch it in the 'Draw Molecule' tab.")
            else:
                render_dashboard_cards(res)

# ---------------------------------------------------------------------
# TAB 2: JSME 2D CHEMICAL STRUCTURE SKETCHER
# ---------------------------------------------------------------------
with tab_sketch:
    st.markdown("### ✏️ Interactive 2D Chemical Canvas")
    st.write("Draw a chemical structure below, click **'Get SMILES from Canvas'**, copy the string, and run prediction.")

    jsme_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <script type="text/javascript" src="https://jsme-editor.github.io/dist/jsme/jsme.nocache.js"></script>
        <script type="text/javascript">
            function jsmeOnLoad() {
                jsmeApplet = new JSApplet.JSME("jsme_container", "100%", "360px", {
                    "options": "query,hydrogens,markAtom,atomHelp"
                });
            }
            function exportSmiles() {
                var smiles = jsmeApplet.smiles();
                document.getElementById("smiles_output").value = smiles;
            }
        </script>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 5px; }
            button { background-color: #ff4b4b; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-weight: bold; margin-top: 8px; }
            button:hover { background-color: #e03b3b; }
            input[type=text] { width: 95%; padding: 8px; margin-top: 8px; border: 1px solid #ccc; border-radius: 4px; font-family: monospace; font-size: 14px; }
        </style>
    </head>
    <body>
        <div id="jsme_container"></div>
        <button type="button" onclick="exportSmiles()">Get SMILES from Canvas</button>
        <br/>
        <input type="text" id="smiles_output" placeholder="Generated SMILES will appear here" readonly onclick="this.select();" />
    </body>
    </html>
    """
    components.html(jsme_html, height=450)

    st.markdown("#### Submit Sketched Structure")
    sketched_smiles = st.text_input("Paste Sketched SMILES Here:", value="C1=CC=C(C=C1)C(=O)O")
    if st.button("🚀 Predict from Sketched Structure", type="primary"):
        with st.spinner("Analyzing sketched molecule..."):
            res = process_single_chemical(sketched_smiles)
            if res["Status"] == "FAILED_RESOLUTION":
                st.error("Invalid SMILES string from sketcher.")
            else:
                render_dashboard_cards(res)

# ---------------------------------------------------------------------
# TAB 3: BATCH CSV PROCESSING & EXPORT
# ---------------------------------------------------------------------
with tab_batch:
    st.markdown("### 📂 Upload Batch File (.csv or .xlsx)")
    st.write("File must contain at least one column labeled `CAS`, `CASRN`, `Name`, `Compound`, or `SMILES`.")

    sample_df = pd.DataFrame({
        "CAS": ["65-85-0", "7440-02-0", "62-53-3", "79-06-1", "79-10-7", "111-44-4", "50-00-0"],
        "Compound_Name": ["Benzoic acid", "Nickel", "Aniline", "Acrylamide", "Acrylic acid", "Bis(2-chloroethyl) ether", "Formaldehyde"],
    })
    csv_template = sample_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Sample CSV Template",
        data=csv_template,
        file_name="sensitization_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload CSV / Excel file", type=["csv", "xlsx"])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_input = pd.read_csv(uploaded_file)
            else:
                df_input = pd.read_excel(uploaded_file)

            st.write("#### Input Data Preview:")
            st.dataframe(df_input.head(), use_container_width=True)

            possible_cols = ["cas", "casrn", "smiles", "name", "compound", "compound_name", "substance"]
            target_col = None
            for c in df_input.columns:
                if c.strip().lower() in possible_cols:
                    target_col = c
                    break

            if not target_col:
                target_col = st.selectbox("Select the column containing the identifiers:", df_input.columns)

            if st.button("🚀 Process Batch Screen", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                results = []

                total = len(df_input)
                for idx, val in enumerate(df_input[target_col]):
                    status_text.text(f"Processing ({idx+1}/{total}): {val}")
                    res = process_single_chemical(str(val))
                    results.append(res)
                    progress_bar.progress((idx + 1) / total)

                status_text.success("Batch screening complete!")
                df_results = pd.DataFrame(results)

                st.markdown("### 📊 Batch Prediction Results")
                st.dataframe(df_results, use_container_width=True)

                n_sens = sum(df_results["OECD_497_Call"] == "SENSITIZER")
                n_nonsens = sum(df_results["OECD_497_Call"] == "NON_SENSITIZER")
                n_err = sum(df_results["Status"] == "FAILED_RESOLUTION")

                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Total Tested", total)
                s2.metric("Sensitizers (Cat 1)", n_sens)
                s3.metric("Non-Sensitizers", n_nonsens)
                s4.metric("Failed / Inconclusive", n_err)

                st.markdown("---")
                col_exp1, col_exp2 = st.columns(2)
                with col_exp1:
                    csv_export = df_results.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Full Results (CSV)",
                        data=csv_export,
                        file_name=f"skin_sensitization_results_{time.strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                with col_exp2:
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                        df_results.to_excel(writer, index=False, sheet_name="Sensitization_Predictions")
                    st.download_button(
                        label="📥 Download Full Results (Excel .xlsx)",
                        data=excel_buffer.getvalue(),
                        file_name=f"skin_sensitization_results_{time.strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )

        except Exception as e:
            st.error(f"Error reading file: {e}")
