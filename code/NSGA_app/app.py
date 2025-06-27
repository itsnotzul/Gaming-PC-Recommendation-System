import streamlit as st
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination

from logic import buildPC_NSGA, valid_cpu_mobo_pairs, gpu_df, ram_df, psu_df, case_df, ram_stick_options, games_df

st.title("PC Build Optimizer")

game_names = st.multiselect("Choose Games", games_df["name"].tolist())
budget = st.number_input("Budget (RM)", min_value=1000, max_value=10000, value=3000)
mode = st.selectbox("Optimization Mode", ["Maximize Performance", "Minimize Cost"])

if st.button("Optimize"):
    try:
        problem = buildPC_NSGA(game_names, budget, mode)
        res = minimize(problem, NSGA2(pop_size=100), get_termination("n_gen", 50), seed=1, save_history=True, verbose=False)

        sol = res.X[0]
        cpu_mobo = valid_cpu_mobo_pairs.iloc[int(sol[0])]
        gpu = gpu_df.iloc[int(sol[1])]
        ram = ram_df.iloc[int(sol[2])]
        sticks = ram_stick_options[int(sol[3])]
        psu = psu_df.iloc[int(sol[4])]
        case = case_df.iloc[int(sol[5])]

        total_price = cpu_mobo['Price_cpu'] + cpu_mobo['Price_mobo'] + gpu['Price'] + ram['Price'] * sticks + psu['Price'] + case['Price']

        st.success(f"Total Cost: RM{total_price:.2f}  |  Unspent Cash: RM{budget - total_price:.2f}")

        labels = ["CPU", "Motherboard", "GPU", "RAM", "PSU", "Case"]
        names = [
            cpu_mobo['Name_cpu'],
            cpu_mobo['Name_mobo'],
            gpu['Name'],
            f"{sticks}x {ram['Name']}",
            psu['Name'],
            case['Name']
        ]
        prices = [
            cpu_mobo['Price_cpu'],
            cpu_mobo['Price_mobo'],
            gpu['Price'],
            ram['Price'] * sticks,
            psu['Price'],
            case['Price']
        ]

        col1, col2 = st.columns(2)

        # Equal-height rows using HTML block with fixed height
        with col1:
            for label in labels:
                st.markdown(
                    f"""<div style="height: 60px; display: flex; align-items: center;">
                            <strong>{label}</strong>
                        </div>""",
                    unsafe_allow_html=True
                )

        with col2:
            for name, price in zip(names, prices):
                st.markdown(
                    f"""<div style="height: 60px;">
                            <div style="font-weight: 500;">{name}</div>
                            <div style="font-family: monospace; color: gray;">RM{price:.2f}</div>
                        </div>""",
                    unsafe_allow_html=True
                )

    except Exception as e:
        st.error(str(e))
