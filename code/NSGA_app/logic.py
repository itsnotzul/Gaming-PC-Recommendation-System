import pandas as pd
import numpy as np
from pymoo.core.problem import ElementwiseProblem

path = "../../data/"

# Load datasets
games_df = pd.read_csv(path + "games/top300.csv")
cpu_df = pd.read_csv(path + "parts/CPU_Data.csv")
gpu_df = pd.read_csv(path + "parts/GPU_Data.csv")
ram_df = pd.read_csv(path + "parts/RAM_Data.csv")
mobo_df = pd.read_csv(path + "parts/MOBO_Data.csv")
psu_df = pd.read_csv(path + "parts/PSU_Data.csv")
case_df = pd.read_csv(path + "parts/Case_Data.csv")

ram_stick_options = [1, 2, 4]

# Preprocess
def safe_to_numeric(value):
    try:
        return pd.to_numeric(value)
    except Exception:
        return value

for df in [games_df, cpu_df, gpu_df, ram_df, mobo_df, psu_df, case_df]:
    df.applymap(safe_to_numeric)

valid_cpu_mobo_pairs = cpu_df.merge(mobo_df, on="Socket", suffixes=("_cpu", "_mobo")).reset_index(drop=True)

# Maximize budget optimizer
class PCBuildProblem(ElementwiseProblem):
    def __init__(self, budget, required_cpu_score, required_gpu_score, required_ram_gb):
        self.budget = budget
        self.required_cpu_score = required_cpu_score
        self.required_gpu_score = required_gpu_score
        self.required_ram_gb = max(required_ram_gb, 8)  # Enforce minimum 8 GB RAM

        super().__init__(
            n_var=6,  # [cpu_mobo, gpu, ram, ram_stick_count, psu, case]
            n_obj=3,  # Maximize CPU score, GPU score, dual channel
            n_constr=8,  # Now includes CPU and GPU score requirements
            xl=np.array([0, 0, 0, 0, 0, 0]),
            xu=np.array([
                len(valid_cpu_mobo_pairs) - 1,
                len(gpu_df) - 1,
                len(ram_df) - 1,
                len(ram_stick_options) - 1,
                len(psu_df) - 1,
                len(case_df) - 1
            ])
        )

    def _evaluate(self, x, out, *args, **kwargs):
        cpu_mobo = valid_cpu_mobo_pairs.iloc[int(x[0])]
        gpu = gpu_df.iloc[int(x[1])]
        ram = ram_df.iloc[int(x[2])]
        sticks = ram_stick_options[int(x[3])]
        psu = psu_df.iloc[int(x[4])]
        case = case_df.iloc[int(x[5])]

        total_ram = ram['Capacity (GB)'] * sticks
        total_price = (
            cpu_mobo['Price_cpu'] + cpu_mobo['Price_mobo'] +
            gpu['Price'] + psu['Price'] + case['Price'] +
            ram['Price'] * sticks
        )

        cpu_score = cpu_mobo['Score']
        gpu_score = gpu['Score']

        # Constraint checks
        wattage_ok = psu['Wattage'] >= gpu['Recommended Power']
        case_ok = str(cpu_mobo['Size']) in str(case['Size'])
        ram_slot_ok = sticks <= cpu_mobo['RAM Slot']
        ddr_ok = str(ram['DDR']) == str(cpu_mobo['DDR'])
        min_ram_ok = total_ram >= self.required_ram_gb
        budget_ok = total_price <= self.budget
        cpu_ok = cpu_score >= self.required_cpu_score
        gpu_ok = gpu_score >= self.required_gpu_score

        # Dual-channel preference
        dual_channel_score = 1 if sticks in [2, 4] else 0

        # Objectives (maximize → negate)
        out["F"] = [
            -cpu_score,
            -gpu_score,
            -dual_channel_score
        ]

        # Constraints: must be <= 0 to be feasible
        out["G"] = [
            0 if wattage_ok else 1,
            0 if case_ok else 1,
            0 if ram_slot_ok else 1,
            0 if ddr_ok else 1,
            0 if min_ram_ok else 1,
            0 if budget_ok else 1,
            0 if cpu_ok else 1,
            0 if gpu_ok else 1 
        ]

# Minimize cost optimizer
class PCBuildCostMinProblem(ElementwiseProblem):
    def __init__(self, budget, required_cpu_score, required_gpu_score, required_ram_gb):
        self.budget = budget
        self.required_cpu_score = required_cpu_score
        self.required_gpu_score = required_gpu_score
        self.required_ram_gb = max(required_ram_gb, 8)  # Enforce minimum 8GB RAM

        super().__init__(
            n_var=6,  # [cpu_mobo_index, gpu_index, ram_index, ram_sticks, psu_index, case_index]
            n_obj=1,  # minimize total price
            n_constr=6,  # CPU, GPU, PSU, RAM, case, and budget
            xl=np.array([0, 0, 0, 0, 0, 0]),
            xu=np.array([
                len(valid_cpu_mobo_pairs) - 1,
                len(gpu_df) - 1,
                len(ram_df) - 1,
                len(ram_stick_options) - 1,
                len(psu_df) - 1,
                len(case_df) - 1
            ])
        )

    def _evaluate(self, x, out, *args, **kwargs):
        cpu_mobo = valid_cpu_mobo_pairs.iloc[int(x[0])]
        gpu = gpu_df.iloc[int(x[1])]
        ram = ram_df.iloc[int(x[2])]
        sticks = ram_stick_options[int(x[3])]
        psu = psu_df.iloc[int(x[4])]
        case = case_df.iloc[int(x[5])]

        total_ram = ram['Capacity (GB)'] * sticks
        total_price = (
            cpu_mobo['Price_cpu'] + cpu_mobo['Price_mobo'] +
            gpu['Price'] + psu['Price'] + case['Price'] +
            ram['Price'] * sticks
        )

        cpu_score = cpu_mobo['Score']
        gpu_score = gpu['Score']

        # Constraints
        wattage_ok = psu['Wattage'] >= gpu['Recommended Power']
        case_ok = str(cpu_mobo['Size']) in str(case['Size'])
        ram_slot_ok = sticks <= cpu_mobo['RAM Slot']
        ddr_ok = str(ram['DDR']) == str(cpu_mobo['DDR'])
        min_ram_ok = total_ram >= self.required_ram_gb
        cpu_ok = cpu_score >= self.required_cpu_score
        gpu_ok = gpu_score >= self.required_gpu_score
        budget_ok = total_price <= self.budget

        out["F"] = [total_price]  # pure cost minimization
        out["G"] = [
            0 if cpu_ok else 1,
            0 if gpu_ok else 1,
            0 if wattage_ok else 1,
            0 if case_ok else 1,
            0 if ram_slot_ok and ddr_ok and min_ram_ok else 1,
            0 if budget_ok else 1  # <-- HARD budget constraint
        ]

# Main function used by frontend
def buildPC_NSGA(game_names: list[str], budget: float, mode: str = "Maximize Performance"):
    game_names_lower = [name.lower() for name in game_names]
    matched_games = games_df[games_df["name"].str.lower().isin(game_names_lower)]

    if matched_games.empty:
        raise ValueError("None of the specified games were found in the dataset.")

    required_cpu_score = matched_games["CPU"].astype(float).max()
    required_gpu_score = matched_games["GPU"].astype(float).max()
    required_ram_gb = matched_games["memory"].astype(int).max()

    if mode == "Maximize Performance":
        return PCBuildProblem(budget, required_cpu_score, required_gpu_score, required_ram_gb)
    elif mode == "Minimize Cost":
        return PCBuildCostMinProblem(budget, required_cpu_score, required_gpu_score, required_ram_gb)
    else:
        raise ValueError(f"Unknown mode: {mode}")
