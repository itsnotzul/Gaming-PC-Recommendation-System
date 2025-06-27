# app.py
import streamlit as st
import pandas as pd
from pulp import LpProblem, LpMaximize, LpMinimize, LpVariable, lpSum, LpStatus

# Page configuration
st.set_page_config(
    page_title="PC Recommendation System",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# STEP 1: Load Data and Compat
# -----------------------------

def normalize_column(df, column_name):
    min_val = df[column_name].min()
    max_val = df[column_name].max()
    df[column_name + "_Normalized"] = (df[column_name] - min_val) / (max_val - min_val)
    return df

def safe_convert(df):
    return df.apply(pd.to_numeric, errors='ignore')

@st.cache_data
def load_and_prepare_data():
    """Load and prepare all data with caching for performance"""
    base_path = "data/parts/"
    cpus = pd.read_csv(base_path + "CPU_Data.csv")
    gpus = pd.read_csv(base_path + "GPU_Data.csv")
    rams = pd.read_csv(base_path + "RAM_Data.csv")
    mobos = pd.read_csv(base_path + "MOBO_Data.csv")
    psus = pd.read_csv(base_path + "PSU_Data.csv")
    cases = pd.read_csv(base_path + "Case_Data.csv")
    storage = pd.read_csv(base_path + "Storage_Data.csv")
    games = pd.read_csv("data/games/top300.csv")

    # Safe conversion
    cpus = safe_convert(cpus)
    gpus = safe_convert(gpus)
    rams = safe_convert(rams)
    mobos = safe_convert(mobos)
    psus = safe_convert(psus)
    cases = safe_convert(cases)
    storage = safe_convert(storage)
    games = safe_convert(games)

    # Split m2s, and satas
    m2s = storage[storage['Storage Type'] == 'M.2 SSD'].copy()
    satas = storage[storage['Storage Type'] == 'SATA SSD'].copy()

    # Add ranking functionality like in notebook
    cpus['Rank'] = cpus['Score'].rank(ascending=False, method='min').astype(int)
    gpus['Rank'] = gpus['Score'].rank(ascending=False, method='min').astype(int)

    # Normalize data 
    cpus = normalize_column(cpus, "Score")
    gpus = normalize_column(gpus, "Score")

    rams_min = rams["Capacity (GB)"].min()
    rams_max = rams["Capacity (GB)"].max()
    rams["Capacity (GB)_Normalized"] = 0.01 + 0.99 * (rams["Capacity (GB)"] - rams_min) / (rams_max - rams_min)    

    return cpus, gpus, rams, mobos, psus, cases, m2s, satas, games

@st.cache_data
def generate_compatibility_dicts(cpus, mobos, rams, cases):
    cpu_mobo = {(i, j): int(cpu['Socket'] == mobo['Socket']) for i, cpu in cpus.iterrows() for j, mobo in mobos.iterrows()}
    ram_mobo = {(i, j): int(ram['DDR'] == mobo['DDR']) for i, ram in rams.iterrows() for j, mobo in mobos.iterrows()}
    case_mobo = {(i, j): int(case['Size'] == mobo['Size']) for i, case in cases.iterrows() for j, mobo in mobos.iterrows()}
    return cpu_mobo, ram_mobo, case_mobo

def filter_components_by_brands(cpus, gpus, rams, mobos, psus, cases, m2s, satas, 
                               cpu_brands, gpu_brands, ram_brands, mobo_brands, mobo_sockets,
                               psu_brands, psu_ratings, case_brands, case_colour, storage_brands,
                               min_ram_speed, min_storage_read, min_storage_write, mobo_wifi_only):
    """Filter all components by selected brands and specifications"""
    filtered_cpus = cpus[cpus['Brand'].isin(cpu_brands)] if cpu_brands else cpus
    filtered_gpus = gpus[gpus['Brand'].isin(gpu_brands)] if gpu_brands else gpus
    
    # Filter RAM by brand, capacity and speed
    filtered_rams = rams[rams['Brand'].isin(ram_brands)] if ram_brands else rams
    if min_ram_speed:
        filtered_rams = filtered_rams[filtered_rams['Speed (MHz)'] >= min_ram_speed]
    
    # Filter motherboards by brand, socket and WiFi
    filtered_mobos = mobos[mobos['Brand'].isin(mobo_brands)] if mobo_brands else mobos
    if mobo_sockets:
        filtered_mobos = filtered_mobos[filtered_mobos['Socket'].isin(mobo_sockets)]
    if mobo_wifi_only:
        filtered_mobos = filtered_mobos[filtered_mobos['Wifi'] == 1]
    
    filtered_psus = psus[psus['Brand'].isin(psu_brands)] if psu_brands else psus
    if psu_ratings:
        filtered_psus = filtered_psus[filtered_psus['80+ Rating'].isin(psu_ratings)]
    filtered_cases = cases[cases['Brand'].isin(case_brands)] if case_brands else cases
    if case_colour:
        filtered_cases = filtered_cases[filtered_cases['Colour'].isin(case_colour)]
    
    # Filter storage by brand and read/write speeds
    filtered_m2s = m2s[m2s['Brand'].isin(storage_brands)] if storage_brands else m2s
    if min_storage_read:
        filtered_m2s = filtered_m2s[filtered_m2s['Read (MB)'] >= min_storage_read]
    if min_storage_write:
        filtered_m2s = filtered_m2s[filtered_m2s['Write (MB)'] >= min_storage_write]
    
    filtered_satas = satas[satas['Brand'].isin(storage_brands)] if storage_brands else satas
    if min_storage_read:
        filtered_satas = filtered_satas[filtered_satas['Read (MB)'] >= min_storage_read]
    if min_storage_write:
        filtered_satas = filtered_satas[filtered_satas['Write (MB)'] >= min_storage_write]
    
    return (filtered_cpus.reset_index(drop=True), filtered_gpus.reset_index(drop=True), 
            filtered_rams.reset_index(drop=True), filtered_mobos.reset_index(drop=True),
            filtered_psus.reset_index(drop=True), filtered_cases.reset_index(drop=True),
            filtered_m2s.reset_index(drop=True), filtered_satas.reset_index(drop=True))

def calculate_combined_game_requirements(games_data):
    """Calculate highest requirements from multiple selected games"""
    if len(games_data) == 0:
        return {"CPU": 0, "GPU": 0, "memory": 8}
    
    max_cpu = games_data['CPU'].max()
    max_gpu = games_data['GPU'].max()
    max_memory = games_data['memory'].max()
    
    return {"CPU": max_cpu, "GPU": max_gpu, "memory": max_memory}

# Load data once at startup
cpus, gpus, rams, mobos, psus, cases, m2s, satas, games = load_and_prepare_data()

# Generate compatibility dictionaries once at startup 
mobo_cpu_compat, mobo_ram_compat, mobo_case_compat = generate_compatibility_dicts(cpus, mobos, rams, cases)

# -----------------------------
# STEP 2: ILP Solver
# -----------------------------
def add_leading_zeros(i, width=4):
    return str(i).zfill(width)

def build_and_solve_ilp(cpusf, gpusf, ramsf, mobosf, psusf, casesf, m2sf, satasf,
                        budget, min_storage, min_ram_capacity, use_sata, min_m2, min_sata, game_data,
                        mobo_cpu_compatibility_dict, mobo_ram_compatibility_dict, mobo_case_compatibility_dict,
                        maximize_performance=True):
    """
    Build and solve the Integer Linear Programming problem for PC optimization
    
    Args:
        maximize_performance (bool): If True, maximize performance within budget.
                                   If False, minimize cost while meeting requirements.
    """
    # --- lenComponents ---
    lenCPU = range(len(cpusf))
    lenGPU = range(len(gpusf))
    lenPSU = range(len(psusf))
    lenMOBO = range(len(mobosf))
    lenCase = range(len(casesf))
    lenRAM = range(len(ramsf))
    lenM2 = range(len(m2sf))
    lenSATA = range(len(satasf))
    
    # --- Creating LP Problem --- 
    problem_type = LpMaximize if maximize_performance else LpMinimize
    problem = LpProblem("Desktop_Optimization", problem_type)    # --- Define PC Components ---
    cpu_vars = [LpVariable(f"cpu_{add_leading_zeros(i)}", cat="Binary") for i in lenCPU]
    gpu_vars = [LpVariable(f"gpu_{add_leading_zeros(i)}", cat="Binary") for i in lenGPU]
    psu_vars = [LpVariable(f"psu_{add_leading_zeros(i)}", cat="Binary") for i in lenPSU]
    mobo_vars = [LpVariable(f"mb_{add_leading_zeros(i)}", cat="Binary") for i in lenMOBO]
    case_vars = [LpVariable(f"case_{add_leading_zeros(i)}", cat="Binary") for i in lenCase]
    ram_vars = [LpVariable(f"ram_{add_leading_zeros(i)}", cat="Binary") for i in lenRAM]
    ram_count = LpVariable("ram_count", lowBound=1, upBound=mobosf['RAM Slot'].max(), cat="Integer")
    ram_count_selected = [LpVariable(f"ram_count_selected_{add_leading_zeros(i)}", lowBound=0, upBound=mobosf['RAM Slot'].max(), cat="Integer") for i in lenRAM] #Auxiliary Variable

    m2_vars = [LpVariable(f"m2_{add_leading_zeros(i)}", cat="Binary") for i in lenM2]
    m2_count = LpVariable("m2_count", lowBound=1, upBound=mobosf['NVMe Slot'].max(), cat="Integer")
    m2_count_selected = [LpVariable(f"m2_count_selected_{add_leading_zeros(i)}", lowBound=0, upBound=mobosf['NVMe Slot'].max(), cat="Integer") for i in lenM2] #Auxiliary Variable

    sata_vars = [LpVariable(f"sata_{add_leading_zeros(i)}", cat="Binary") for i in lenSATA]
    sata_count = LpVariable("sata_count", lowBound=1, upBound=mobosf['SATA Slot'].max(), cat="Integer")
    sata_count_selected = [LpVariable(f"sata_count_selected_{add_leading_zeros(i)}", lowBound=0, upBound=mobosf['SATA Slot'].max(), cat="Integer") for i in lenSATA] #Auxiliary Variable    # ''' ---Part Count Constraint--- '''
    
    # Only select exactly one of each major part
    problem += lpSum(cpu_vars) == 1, "Select_One_CPU"
    problem += lpSum(gpu_vars) == 1, "Select_One_GPU"
    problem += lpSum(ram_vars) == 1, "Select_One_RAM"
    problem += lpSum(psu_vars) == 1, "Select_One_PSU"
    problem += lpSum(mobo_vars) == 1, "Select_One_MOBO"
    problem += lpSum(case_vars) == 1, "Select_One_Case"

    # ''' ---GPU Wattage Constraints--- '''
    # GPU x Power Supply
    problem += (
        lpSum(gpu_vars[i] * gpusf.iloc[i]["Recommended Power"] for i in lenGPU) <=
        lpSum(psu_vars[i] * psusf.iloc[i]["Wattage"] for i in lenPSU),
        "PSU_Power_Constraint",
    )

    # ''' Storage Constraints '''
    problem += (
        lpSum(m2_count_selected[i] * m2sf.iloc[i]["Capacity (GB)"] for i in lenM2) +
        lpSum(sata_count_selected[i] * satasf.iloc[i]["Capacity (GB)"] for i in lenSATA)
        >= min_storage,
        "Minimum_Total_Storage"
    )

    # ''' Default: 1 M.2 SSD 512GB Storage '''
    # ''' M.2 SSD NVMe '''
    problem += m2_count >= min_m2, "Minimum_M2"
    problem += m2_count <= lpSum(mobo_vars[i] * mobosf.iloc[i]['NVMe Slot'] for i in lenMOBO)

    for i in lenM2:
        problem += m2_count_selected[i] <= m2_count
        problem += m2_count_selected[i] <= mobosf['NVMe Slot'].max() * m2_vars[i]

    problem += lpSum(m2_count_selected[i] for i in lenM2) == m2_count

    # ''' SATA SSD '''
    if use_sata:
        problem += sata_count >= min_sata, "Minimum_SATA"
        problem += sata_count <= lpSum(mobo_vars[i] * mobosf.iloc[i]['SATA Slot'] for i in lenMOBO)

        for i in lenSATA:
            problem += sata_count_selected[i] <= sata_count
            problem += sata_count_selected[i] <= mobosf['SATA Slot'].max() * sata_vars[i]

        problem += lpSum(sata_count_selected[i] for i in lenSATA) == sata_count
    else:
        for i in lenSATA:
            problem += sata_vars[i] == 0
            problem += sata_count_selected[i] == 0

    # ''' ---Motherboard Constraints--- '''
    # Socket 
    for i in lenCPU:
        problem += (
            cpu_vars[i] <= lpSum(mobo_cpu_compatibility_dict[(i,j)] * mobo_vars[j] for j in lenMOBO),
            f"CPU_Socket_Compatibility_{add_leading_zeros(i)}"
        )

    # Case
    for i in lenCase:
        problem += (
            case_vars[i] <= lpSum(mobo_case_compatibility_dict[(i,j)] * mobo_vars[j] for j in lenMOBO),
            f"Case_Size_Compatibility_{add_leading_zeros(i)}"
        )

    # DDR
    for i in lenRAM:
        problem += (
            ram_vars[i] <= lpSum(mobo_ram_compatibility_dict[(i,j)] * mobo_vars[j] for j in lenMOBO),
            f"RAM_DDR_Compatibility_{add_leading_zeros(i)}"
        )

    # RAM Count
    problem += ram_count >= 1, "At_Least_One_RAM"
    problem += ram_count <= lpSum(mobo_vars[i] * mobosf.iloc[i]["RAM Slot"] for i in lenMOBO)

    # ''' ---RAM Selection Algorithm--- '''
    # For each RAM Model, link count to total count
    for i in lenRAM:
        problem += ram_count_selected[i] <= ram_count
        problem += ram_count_selected[i] <= 4 * ram_vars[i]
        problem += ram_count_selected[i] >= ram_count - (1 - ram_vars[i]) * 4
        problem += ram_count_selected[i] >= 0

    # ''' ---Game Constraints--- '''
    problem += (
        lpSum(cpu_vars[i] * cpusf.iloc[i]["Score"] for i in lenCPU) >= game_data["CPU"],
        "Game_CPU_Constraint"
    )

    problem += (
        lpSum(gpu_vars[i] * gpusf.iloc[i]["Score"] for i in lenGPU) >= game_data["GPU"],
        "Game_GPU_Constraint"
    )

    memory_requirement = int(game_data["memory"])

    if min_ram_capacity > memory_requirement:
        memory_requirement = min_ram_capacity

    problem += (
        lpSum(ram_count_selected[i] * ramsf.iloc[i]['Capacity (GB)'] for i in lenRAM) >= memory_requirement,
        "Game_Memory_Constraint"
    )

    # ''' ---Cost Function--- '''
    total_cost = (
        lpSum(cpu_vars[i] * cpusf.iloc[i]["Price"] for i in lenCPU) +
        lpSum(gpu_vars[i] * gpusf.iloc[i]["Price"] for i in lenGPU) +
        lpSum(ram_count_selected[i] * ramsf.iloc[i]["Price"] for i in lenRAM) +
        lpSum(psu_vars[i] * psusf.iloc[i]["Price"] for i in lenPSU) +
        lpSum(mobo_vars[i] * mobosf.iloc[i]["Price"] for i in lenMOBO) +
        lpSum(case_vars[i] * casesf.iloc[i]["Price"] for i in lenCase) +
        lpSum(m2_count_selected[i] * m2sf.iloc[i]["Price"] for i in lenM2) +
        lpSum(sata_count_selected[i] * satasf.iloc[i]["Price"] for i in lenSATA)
    )

    # Budget constraint applies to both modes
    if budget > 0:
        problem += total_cost <= budget, "Budget_Constraint"
    
    # ''' ---Performance Function---'''
    # Parameters (same for both optimization modes)
    cpu_weight = 0.3
    gpu_weight = 0.695
    ram_weight = 0.005
    dual_channel_bonus_value = 0.02

    # Extra variables
    dual_channel_bonus_var = LpVariable("dual_channel_bonus_var", cat="Binary")

    # Performance function (always calculated)
    total_performance = (
        cpu_weight * lpSum(cpu_vars[i] * cpusf.iloc[i]["Score_Normalized"] for i in lenCPU) +
        gpu_weight * lpSum(gpu_vars[i] * gpusf.iloc[i]["Score_Normalized"] for i in lenGPU) +
        ram_weight * lpSum(ram_count_selected[i] * ramsf.iloc[i]["Capacity (GB)_Normalized"] for i in lenRAM) +
        dual_channel_bonus_value * dual_channel_bonus_var
    )

    # Dual-Channel Activation: activate bonus if RAM count >= 2
    problem += ram_count_selected >= 2 * dual_channel_bonus_var, "Dual-Channel Activation Condition"
    
    # Set objective based on optimization mode
    if maximize_performance:
        # Objective: Maximize performance (within budget constraint)
        problem += total_performance, "Objective"
    else:
        # Objective: Minimize cost (while meeting game requirements)
        problem += total_cost, "Objective"

    result_status = problem.solve()
    
    return {
        "status": LpStatus[result_status],
        "cpu": [cpusf.iloc[i] for i in lenCPU if cpu_vars[i].varValue == 1],
        "gpu": [gpusf.iloc[i] for i in lenGPU if gpu_vars[i].varValue == 1],
        "ram": [(ramsf.iloc[i], int(ram_count_selected[i].varValue or 0)) for i in lenRAM if ram_count_selected[i].varValue and ram_count_selected[i].varValue > 0],
        "mobo": [mobosf.iloc[i] for i in lenMOBO if mobo_vars[i].varValue == 1],
        "psu": [psusf.iloc[i] for i in lenPSU if psu_vars[i].varValue == 1],
        "case": [casesf.iloc[i] for i in lenCase if case_vars[i].varValue == 1],
        "m2": [(m2sf.iloc[i], int(m2_count_selected[i].varValue or 0)) for i in lenM2 if m2_count_selected[i].varValue and m2_count_selected[i].varValue > 0],
        "sata": [(satasf.iloc[i], int(sata_count_selected[i].varValue or 0)) for i in lenSATA if sata_count_selected[i].varValue and sata_count_selected[i].varValue > 0]
    }

# -----------------------------
# STEP 3: Streamlit UI
# -----------------------------
st.title("🖥️ PC Build Recommender")
st.sidebar.title("Optimizer Options")

# Optimization Objective

# Define default filter values
DEFAULT_FILTERS = {
    "cpu_brands": [],
    "gpu_brands": [],
    "ram_brands": [],
    "mobo_brands": [],
    "mobo_sockets": [],
    "psu_brands": [],
    "psu_ratings": [],
    "case_brands": [],
    "case_colours": [],
    "storage_brands": [],
    "optimization_mode": "Maximize Performance",
    "selected_games": ['Counter-Strike 2'],
    "budget": 1300,
    "storage_required": 512,
    "ram16_toggle": True,
    "use_sata": False,
    "min_m2": 1,
    "min_sata": 1,
    "min_ram_capacity": 8,
    "min_ram_speed": 2400,
    "min_storage_read": 500,
    "min_storage_write": 500,
    "mobo_wifi_only": False
}

# Reset to default button
if st.sidebar.button("🔄 Reset to Default", type="secondary"):
    for key, default_value in DEFAULT_FILTERS.items():
        st.session_state[key] = default_value
    st.rerun()

st.sidebar.subheader("🎯 Objective")
optimization_mode = st.sidebar.radio(
    "Choose optimization goal:",
    options=["Maximize Performance", "Minimize Cost"],
    index=0,
    key= "optimization_mode",
    help="Maximize will find the best performance, while minimize will find the bare minimum"
)
with st.sidebar.expander("ℹ️ About Optimization Modes"):
    st.write("""
    **Maximize Performance:**
    - Finds the highest performing components within your budget
    - Prioritizes CPU and GPU performance scores
    - May use most or all of your budget
    
    **Minimize Cost:**
    - Finds the cheapest build that meets game requirements
    - Focuses on meeting minimum specs efficiently
    - Will likely cost much less than your budget
    """)

# Game Selection
st.sidebar.subheader("🎮 Game Selection")
selected_games = st.sidebar.multiselect(
    "Choose Games",
    options=games['name'].unique(),
    default=['Counter-Strike 2'], key = "selected_games"
)

# Budget and Storage
st.sidebar.subheader("💰 Budget & Storage")
budget = st.sidebar.number_input("Max Budget (RM)", min_value=0, value=1300, step=100, key="budget")
storage_required = st.sidebar.number_input("Storage Needed (GB)", min_value=128, value=512, step=128, key="storage_required")

# Extra Options
st.sidebar.subheader("⚙ Extra Options")
ram16_toggle = st.sidebar.checkbox("Min. 16GB Memory", value=True, help="Build includes minimum 16GB Memory", key="ram16_toggle")
use_sata = st.sidebar.checkbox("Incl. SATA Storage", value=False, help="Cheaper & more storage, slower than M.2", key="use_sata")

# Storage/Memory Options
st.sidebar.subheader("💾 Storage/Memory Options")

# Default values - will be updated based on filtering
max_nvme_slots = mobos['NVMe Slot'].max() if len(mobos) > 0 else 4
max_sata_slots = mobos['SATA Slot'].max() if len(mobos) > 0 else 4

# RAM Capacity filter - based on Min. 16GB Memory toggle
if not ram16_toggle:
    min_ram_capacity = st.sidebar.number_input(
        "Minimum RAM Capacity (GB)",
        min_value=4,
        max_value=128,
        value=4,
        step=4,
        key="min_ram_capacity"
    )
else:
    min_ram_capacity = 16

min_m2 = st.sidebar.number_input("Minimum M.2 SSDs", min_value=1, max_value=max_nvme_slots, value=1, key="min_m2")
if use_sata:
    min_sata = st.sidebar.number_input("Minimum SATA SSDs", min_value=1, max_value=max_sata_slots, value=1, key="min_sata")
else:
    min_sata = 0

st.sidebar.caption(f"Max slots available: NVMe: {max_nvme_slots}, SATA: {max_sata_slots}")

# Brand Filtering
st.sidebar.subheader("📁 Filtering")
with st.sidebar.expander("CPU"):
    cpu_brands = st.multiselect("Select CPU Brands", options=sorted(cpus['Brand'].unique()), key="cpu_brands")

with st.sidebar.expander("GPU"):
    gpu_brands = st.multiselect("Select GPU Brands", options=sorted(gpus['Brand'].unique()), key="gpu_brands")

with st.sidebar.expander("RAM"):
    ram_brands = st.multiselect("Select RAM Brands", options=sorted(rams['Brand'].unique()), key="ram_brands")
    min_ram_speed = st.number_input("Minimum RAM Speed (MHz)", min_value=1600, max_value=6000, value=2400, step=100, key="min_ram_speed")

with st.sidebar.expander("Motherboard"):
    mobo_brands = st.multiselect("Select Motherboard Brands", options=sorted(mobos['Brand'].unique()), key="mobo_brands")
    mobo_sockets = st.multiselect("Select Motherboard Sockets", options=sorted(mobos['Socket'].unique()), key="mobo_sockets")
    mobo_wifi_only = st.checkbox("WiFi Required", value=False, help="Only show motherboards with WiFi", key="mobo_wifi_only")

# Custom sort order for PSU 80+ Rating
psu_rating_order = ['White', 'Bronze', 'Silver', 'Gold', 'Platinum']
psu_ratings_unique = [r for r in psu_rating_order if r in psus['80+ Rating'].dropna().unique()]

with st.sidebar.expander("PSU"):
    psu_brands = st.multiselect("Select PSU Brands", options=sorted(psus['Brand'].unique()), key="psu_brands")
    psu_ratings = st.multiselect("Select 80+ Ratings", options=psu_ratings_unique, key="psu_ratings")

with st.sidebar.expander("Case"):
    case_brands = st.multiselect("Select Case Brands", options=sorted(cases['Brand'].unique()), key="case_brands")
    case_colours = st.multiselect("Select Case Colours", options=sorted(cases['Colour'].unique()), key="case_colours")


with st.sidebar.expander("Storage"):
    storage_brands = st.multiselect("Select Storage Brands", options=sorted(pd.concat([m2s['Brand'], satas['Brand']]).unique()), key="storage_brands")
    # Storage Speed filters
    min_storage_read = st.number_input("Minimum Read Speed (MB/s)", min_value=100, max_value=7000, value=500, step=100, key="min_storage_read")
    min_storage_write = st.number_input("Minimum Write Speed (MB/s)", min_value=100, max_value=7000, value=500, step=100, key="min_storage_write")

# Main content area
if selected_games:
    selected_games_data = games[games['name'].isin(selected_games)]
    combined_requirements = calculate_combined_game_requirements(selected_games_data)
    
    st.subheader("🎯 Selected Games & Requirements")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Selected Games:**")
        for game in selected_games:
            st.write(f"• {game}")
    
    with col2:
        st.write("**Combined Requirements:**")
        st.write(f"• CPU Score: {combined_requirements['CPU']}")
        st.write(f"• GPU Score: {combined_requirements['GPU']}")
        st.write(f"• Memory: {combined_requirements['memory']} GB")

if st.button("🔍 Find Optimal Build", type="primary"):
    if not selected_games:
        st.error("❌ Please select at least one game!")
    else:
        # Filter components by selected brands
        filtered_cpus, filtered_gpus, filtered_rams, filtered_mobos, filtered_psus, filtered_cases, filtered_m2s, filtered_satas = filter_components_by_brands(
            cpus, gpus, rams, mobos, psus, cases, m2s, satas,
            cpu_brands, gpu_brands, ram_brands, mobo_brands, mobo_sockets, psu_brands, psu_ratings, case_brands, case_colours, storage_brands,
            min_ram_speed, min_storage_read, min_storage_write, mobo_wifi_only
        )
        
        # ✅ CRITICAL FIX: Regenerate compatibility dictionaries with filtered data
        filtered_mobo_cpu_compat, filtered_mobo_ram_compat, filtered_mobo_case_compat = generate_compatibility_dicts(
            filtered_cpus, filtered_mobos, filtered_rams, filtered_cases
        )
        
        # Dynamic validation of storage counts based on filtered motherboards
        max_nvme_filtered = filtered_mobos['NVMe Slot'].max() if len(filtered_mobos) > 0 else 0
        max_sata_filtered = filtered_mobos['SATA Slot'].max() if len(filtered_mobos) > 0 else 0
        
        # Validate storage count settings against filtered motherboards
        storage_errors = []
        if min_m2 > max_nvme_filtered:
            storage_errors.append(f"Minimum M.2 SSDs ({min_m2}) exceeds maximum NVMe slots ({max_nvme_filtered}) in filtered motherboards")
        if use_sata and min_sata > max_sata_filtered:
            storage_errors.append(f"Minimum SATA SSDs ({min_sata}) exceeds maximum SATA slots ({max_sata_filtered}) in filtered motherboards")
        
        if storage_errors:
            st.error("❌ Storage Configuration Error:")
            for error in storage_errors:
                st.error(f"• {error}")
            st.info("💡 Adjust your storage requirements or motherboard brand filters.")
            st.stop()
        
        # Check if any component category is empty after filtering
        component_counts = {
            "CPU": len(filtered_cpus),
            "GPU": len(filtered_gpus), 
            "RAM": len(filtered_rams),
            "Motherboard": len(filtered_mobos),
            "PSU": len(filtered_psus),
            "Case": len(filtered_cases),
            "M.2 Storage": len(filtered_m2s),
            "SATA Storage": len(filtered_satas) if use_sata else 1  # Don't check SATA if not used
        }
        
        empty_components = [comp for comp, count in component_counts.items() if count == 0]
        
        if empty_components:
            st.error(f"❌ No components found for: {', '.join(empty_components)}. Please adjust your brand filters.")
        else:
            # Calculate combined game requirements
            selected_games_data = games[games['name'].isin(selected_games)]
            game_data = calculate_combined_game_requirements(selected_games_data)
            
            # Determine optimization mode
            maximize_performance = optimization_mode == "Maximize Performance"
            
            with st.spinner("🔍 Finding optimal build..."):
                result = build_and_solve_ilp(
                    filtered_cpus, filtered_gpus, filtered_rams, filtered_mobos, 
                    filtered_psus, filtered_cases, filtered_m2s, filtered_satas,
                    budget, storage_required, min_ram_capacity, use_sata, min_m2, min_sata, game_data,
                    filtered_mobo_cpu_compat, filtered_mobo_ram_compat, filtered_mobo_case_compat,
                    maximize_performance
                )

            if result["status"] != "Optimal":
                st.error(f"❌ No feasible build found: {result['status']}")
                st.info("💡 Try increasing your budget, reducing requirements, or adjusting brand filters.")
            else:
                if maximize_performance:
                    st.success("✅ Optimal High-Performance Build Found!")
                    st.info(f"🎯 **Goal**: Maximum performance within RM {budget:,} budget")
                else:
                    st.success("✅ Optimal Cost-Efficient Build Found!")
                    st.info(f"🎯 **Goal**: Minimum cost while meeting game requirements")
                
                # Calculate total cost
                total_cost = 0
                
                st.subheader("🛠️ Your Optimized Build")
                
                # Display components with clean, compact design
                
                # CPU
                if result["cpu"]:
                    comp = result["cpu"][0]
                    total_cpu_count = len(cpus)
                    
                    st.markdown(f"### 🖥️ **CPU**: {comp['Name']}")
                    cols = st.columns([2, 1, 1])
                    with cols[0]:
                        st.caption(f"**Brand:** {comp['Brand']} | **Socket:** {comp['Socket']}")
                    with cols[1]:
                        st.metric("Price", f"RM {comp['Price']:,.0f}")
                    with cols[2]:
                        st.metric("Score", f"{comp['Score']:,.0f}")
                        st.caption(f"Top {(comp['Rank']/total_cpu_count)*100:.2f}%")
                    
                    total_cost += comp['Price']
                    st.divider()
                
                # GPU
                if result["gpu"]:
                    comp = result["gpu"][0]
                    total_gpu_count = len(gpus)
                    
                    st.markdown(f"### 🎮 **GPU**: {comp['Name']}")
                    cols = st.columns([2, 1, 1])
                    with cols[0]:
                        st.caption(f"**Brand:** {comp['Brand']}")
                    with cols[1]:
                        st.metric("Price", f"RM {comp['Price']:,.0f}")
                    with cols[2]:
                        st.metric("Score", f"{comp['Score']:,.0f}")
                        st.caption(f"Top {(comp['Rank']/total_gpu_count)*100:.2f}%")
                    
                    with st.expander("🔍 Additional Details", expanded=False):
                        detail_cols = st.columns(1)
                        with detail_cols[0]:
                            if 'Series' in comp.index:
                                st.caption(f"**Series:** {comp['Series']}")
                            if 'Power Consumption' in comp.index:
                                st.caption(f"**Power Consumption:** {comp['Power Consumption']} W")
                            if 'Recommended Power' in comp.index:
                                st.caption(f"**Recommended PSU:** {comp['Recommended Power']} W")
                    
                    total_cost += comp['Price']
                    st.divider()
                
                # Motherboard
                if result["mobo"]:
                    comp = result["mobo"][0]
                    wifi_status = "Yes" if comp.get('Wifi') == 1 else "No"
                    
                    st.markdown(f"### 🔌 **Motherboard**: {comp['Name']}")
                    cols = st.columns([2, 1, 1])
                    with cols[0]:
                        st.caption(f"**Brand:** {comp['Brand']} | **Socket:** {comp['Socket']}")
                    with cols[1]:
                        st.metric("Price", f"RM {comp['Price']:,.0f}")
                    with cols[2]:
                        st.caption(f"**RAM:** {comp['RAM Slot']} slots")
                        st.caption(f"**NVMe:** {comp['NVMe Slot']} | **SATA:** {comp['SATA Slot']}")
                    
                    with st.expander("🔍 Additional Details", expanded=False):
                        detail_cols = st.columns(1)
                        with detail_cols[0]:
                            st.caption(f"**Size:** {comp['Size']}")
                            st.caption(f"**WiFi:** {wifi_status}")
                    
                    total_cost += comp['Price']
                    st.divider()
                
                # PSU
                if result["psu"]:
                    comp = result["psu"][0]
                    
                    st.markdown(f"### ⚡ **PSU**: {comp['Name']}")
                    cols = st.columns([2, 1, 1])
                    with cols[0]:
                        st.caption(f"**Brand:** {comp['Brand']} | **80+ Rating:** {comp['80+ Rating']}")
                    with cols[1]:
                        st.metric("Price", f"RM {comp['Price']:,.0f}")
                    with cols[2]:
                        st.metric("Wattage", f"{comp['Wattage']} W")
                    
                    total_cost += comp['Price']
                    st.divider()
                
                # Case
                if result["case"]:
                    comp = result["case"][0]
                    
                    st.markdown(f"### 📦 **Case**: {comp['Name']}")
                    cols = st.columns([2, 1, 1])
                    with cols[0]:
                        st.caption(f"**Brand:** {comp['Brand']} | **Size:** {comp['Size']}")
                    with cols[1]:
                        st.metric("Price", f"RM {comp['Price']:,.0f}")
                    with cols[2]:
                        st.metric("Size", comp['Size'])
                    
                    with st.expander("🔍 Additional Details", expanded=False):
                        detail_cols = st.columns(1)
                        with detail_cols[0]:
                            if 'Colour' in comp.index:
                                st.caption(f"**Colour:** {comp['Colour']}")
                    
                    total_cost += comp['Price']
                    st.divider()
                
                # RAM
                if result["ram"]:
                    ram_item, count = result["ram"][0]
                    ram_total = ram_item['Price'] * count
                    total_capacity = ram_item['Capacity (GB)'] * count
                    
                    st.markdown(f"### 🧠 **RAM**: {ram_item['Name']} ×{count}")
                    cols = st.columns([2, 1, 1])
                    with cols[0]:
                        st.caption(f"**Brand:** {ram_item['Brand']} | **Total:** {total_capacity} GB")
                    with cols[1]:
                        st.metric("Total Price", f"RM {ram_total:,.0f}")
                        st.caption(f"Unit: RM {ram_item['Price']:,.0f}")
                    with cols[2]:
                        st.metric("Capacity", f"{total_capacity} GB")
                        st.caption(f"Per stick: {ram_item['Capacity (GB)']} GB")
                    
                    with st.expander("🔍 Additional Details", expanded=False):
                        detail_cols = st.columns(1)
                        with detail_cols[0]:
                            if 'DDR' in ram_item.index:
                                st.caption(f"**DDR Type:** {ram_item['DDR']}")
                            if 'Speed (MHz)' in ram_item.index:
                                st.caption(f"**Speed (MHz):** {ram_item['Speed (MHz)']}")
                    
                    total_cost += ram_total
                    st.divider()
                
                # M.2 SSDs
                if result["m2"]:
                    for storage_item, count in result["m2"]:
                        storage_total = storage_item['Price'] * count
                        total_capacity = storage_item['Capacity (GB)'] * count
                        st.markdown(f"### 💽 **M.2 SSD**: {storage_item['Name']} ×{count}")
                        cols = st.columns([2, 1, 1])
                        with cols[0]:
                            st.caption(f"**Brand:** {storage_item['Brand']} | **Total:** {total_capacity} GB")
                        with cols[1]:
                            st.metric("Total Price", f"RM {storage_total:,.0f}")
                            st.caption(f"Unit: RM {storage_item['Price']:,.0f}")
                        with cols[2]:
                            st.metric("Storage", f"{total_capacity} GB")
                            st.caption(f"Per drive: {storage_item['Capacity (GB)']} GB")
                        with st.expander("🔍 Additional Details", expanded=False):
                            detail_cols = st.columns(1)
                            with detail_cols[0]:
                                if 'Read (MB)' in storage_item.index:
                                    st.caption(f"**Read Speed:** {storage_item['Read (MB)']} MB/s")
                                if 'Write (MB)' in storage_item.index:
                                    st.caption(f"**Write Speed:** {storage_item['Write (MB)']} MB/s")
                        total_cost += storage_total
                        st.divider()
                # SATA SSDs
                if result["sata"]:
                    for storage_item, count in result["sata"]:
                        storage_total = storage_item['Price'] * count
                        total_capacity = storage_item['Capacity (GB)'] * count
                        st.markdown(f"### 💿 **SATA SSD**: {storage_item['Name']} ×{count}")
                        cols = st.columns([2, 1, 1])
                        with cols[0]:
                            st.caption(f"**Brand:** {storage_item['Brand']} | **Total:** {total_capacity} GB")
                        with cols[1]:
                            st.metric("Total Price", f"RM {storage_total:,.0f}")
                            st.caption(f"Unit: RM {storage_item['Price']:,.0f}")
                        with cols[2]:
                            st.metric("Storage", f"{total_capacity} GB")
                            st.caption(f"Per drive: {storage_item['Capacity (GB)']} GB")
                        with st.expander("🔍 Additional Details", expanded=False):
                            detail_cols = st.columns(1)
                            with detail_cols[0]:
                                if 'Read (MB)' in storage_item.index:
                                    st.caption(f"**Read Speed:** {storage_item['Read (MB)']} MB/s")
                                if 'Write (MB)' in storage_item.index:
                                    st.caption(f"**Write Speed:** {storage_item['Write (MB)']} MB/s")
                        total_cost += storage_total
                        st.divider()
                
                # Summary
                st.divider()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("💸 Total Cost", f"RM {total_cost:,.0f}")
                with col2:
                    st.metric("💰 Budget Used", f"{(total_cost/budget)*100:.2f}%")
                with col3:
                    if budget > 0:
                        remaining = budget - total_cost
                        st.metric("💵 Remaining", f"RM {remaining:,.0f}")
                # Show optimization info
                if maximize_performance:
                    st.caption("🔥 This build maximizes performance within your budget constraints.")
                else:
                    st.caption("💰 This build minimizes cost while meeting all game requirements.")
else:
    st.info("👈 Configure your build requirements in the sidebar and select games to get started!")