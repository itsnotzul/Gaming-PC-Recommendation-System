import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pandas as pd
import os
from pathlib import Path
import threading
from queue import Queue, Empty

class PCBuildOptimizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PC Build Optimizer")
        self.root.geometry("1200x800")
        self.queue = Queue()
        self.update_thread = None
        self._compatibility_cache = {}
        
        # Data storage
        self.data_loaded = False
        self.games_df = None
        self.cpus_df = None
        self.gpus_df = None
        self.rams_df = None
        self.mobos_df = None
        self.psus_df = None
        self.cases_df = None
        self.storage_df = None
        
        # Create main notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.setup_tab = ttk.Frame(self.notebook)
        self.filters_tab = ttk.Frame(self.notebook)
        self.results_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.setup_tab, text="Game & Budget")
        self.notebook.add(self.filters_tab, text="Component Filters")
        self.notebook.add(self.results_tab, text="Build Results")  
        
        self.setup_main_tab()
        self.setup_filters_tab()
        self.setup_results_tab()
        
        # Try to load data on startup
        self.load_data()
    
    def setup_main_tab(self):
        """Setup the main game selection and budget tab"""
        main_frame = ttk.Frame(self.setup_tab)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Data path section
        ttk.Label(main_frame, text="Data Path:", font=('Arial', 12, 'bold')).pack(anchor='w', pady=(0, 5))
        path_frame = ttk.Frame(main_frame)
        path_frame.pack(fill='x', pady=(0, 20))
        
        self.data_path_var = tk.StringVar(value="data/")
        ttk.Entry(path_frame, textvariable=self.data_path_var, width=50).pack(side='left', padx=(0, 10))
        ttk.Button(path_frame, text="Load Data", command=self.load_data).pack(side='left')
        
        self.data_status_label = ttk.Label(main_frame, text="Data not loaded", foreground="red")
        self.data_status_label.pack(anchor='w', pady=(0, 20))
        
        # Game selection
        ttk.Label(main_frame, text="Select Game(s):", font=('Arial', 12, 'bold')).pack(anchor='w', pady=(0, 5))
        
        # Game listbox with scrollbar
        game_frame = ttk.Frame(main_frame)
        game_frame.pack(fill='x', pady=(0, 20))
        
        self.game_listbox = tk.Listbox(game_frame, selectmode='multiple', height=8)
        game_scrollbar = ttk.Scrollbar(game_frame, orient='vertical', command=self.game_listbox.yview)
        self.game_listbox.configure(yscrollcommand=game_scrollbar.set)
        
        self.game_listbox.pack(side='left', fill='both', expand=True)
        game_scrollbar.pack(side='right', fill='y')
        
        # Budget section
        ttk.Label(main_frame, text="Budget (RM):", font=('Arial', 12, 'bold')).pack(anchor='w', pady=(0, 5))
        self.budget_var = tk.StringVar(value="1600")
        budget_entry = ttk.Entry(main_frame, textvariable=self.budget_var, width=20)
        budget_entry.pack(anchor='w', pady=(0, 20))
        
        # Storage requirements
        storage_frame = ttk.LabelFrame(main_frame, text="Storage Requirements", padding=10)
        storage_frame.pack(fill='x', pady=(0, 20))
        
        ttk.Label(storage_frame, text="Minimum Total Storage (GB):").grid(row=0, column=0, sticky='w', padx=(0, 10))
        self.min_storage_var = tk.StringVar(value="1024")
        ttk.Entry(storage_frame, textvariable=self.min_storage_var, width=10).grid(row=0, column=1, sticky='w')
        
        self.use_sata_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(storage_frame, text="Include SATA SSD", variable=self.use_sata_var).grid(row=1, column=0, sticky='w', pady=5)
        
        self.ram_16_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(storage_frame, text="Minimum 16GB RAM", variable=self.ram_16_var).grid(row=1, column=1, sticky='w', pady=5)
    
    def setup_filters_tab(self):
        """Setup the component filters tab"""
        # Create instance variables for widgets
        self.brand_filters = {}  # Initialize the dictionary first

        # Modify canvas to fill window
        canvas = tk.Canvas(self.filters_tab)
        scrollbar = ttk.Scrollbar(self.filters_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        # Configure scrollable_frame to expand
        self.filters_tab.grid_rowconfigure(0, weight=1)
        self.filters_tab.grid_columnconfigure(0, weight=1)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Use grid instead of pack for better expansion
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        # Modify on_select to only affect current listbox
        def on_select(event):
            """Highlight selected items in listbox"""
            widget = event.widget
            selected = widget.curselection()

            # Only clear the current listbox selections
            widget.select_clear(0, tk.END)
            for idx in selected:
                widget.selection_set(idx)
                widget.itemconfig(idx, {'bg': '#e6e6e6'})

            # Clear highlighting for unselected items in current listbox only
            all_indices = range(widget.size())
            unselected = set(all_indices) - set(selected)
            for idx in unselected:
                widget.itemconfig(idx, {'bg': 'white'})

        # Brand filters frame
        brand_frame = ttk.LabelFrame(scrollable_frame, text="Brand Filters", padding=10)
        brand_frame.pack(fill='x', padx=20, pady=10)

        # Create brand filter variables and listboxes
        self.brand_filters = {}
        brand_categories = [
            ("CPU Brands", "cpu_brands", self.cpus_df),
            ("GPU Brands", "gpu_brands", self.gpus_df),
            ("RAM Brands", "ram_brands", self.rams_df),
            ("Motherboard Brands", "mobo_brands", self.mobos_df),
            ("PSU Brands", "psu_brands", self.psus_df),
            ("Case Brands", "case_brands", self.cases_df),
            ("Storage Brands", "storage_brands", self.storage_df)
        ]

        for i, (label, key, df) in enumerate(brand_categories):
            frame = ttk.Frame(brand_frame)
            frame.grid(row=i, column=0, sticky='ew', pady=2)
            frame.grid_columnconfigure(1, weight=1)

            ttk.Label(frame, text=f"{label}:").grid(row=0, column=0, sticky='w', padx=(0, 10))

            list_frame = ttk.Frame(frame)
            list_frame.grid(row=0, column=1, sticky='e')

            listbox = tk.Listbox(list_frame, selectmode='multiple', height=3, width=40)
            scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=listbox.yview)
            listbox.configure(yscrollcommand=scrollbar.set)

            listbox.pack(side='left', fill='x', expand=True)
            scrollbar.pack(side='right', fill='y')

            # Add selection binding to listbox
            listbox.bind('<<ListboxSelect>>', on_select)
            listbox.configure(selectmode='multiple', 
                         activestyle='none',  # Remove dotted line around selected item
                         selectbackground='#0078D7',  # Windows blue
                         selectforeground='white')
            
            self.brand_filters[key] = listbox

        # Motherboard specifications
        mobo_frame = ttk.LabelFrame(scrollable_frame, text="Motherboard Specifications", padding=10)
        mobo_frame.pack(fill='x', padx=20, pady=10)

        # Socket and DDR dropdowns
        ttk.Label(mobo_frame, text="Socket:").grid(row=0, column=0, sticky='w', padx=(0, 10))
        self.mobo_socket_var = tk.StringVar()
        self.socket_combo = ttk.Combobox(mobo_frame, textvariable=self.mobo_socket_var)
        self.socket_combo.grid(row=0, column=1, sticky='w')

        ttk.Label(mobo_frame, text="DDR Version:").grid(row=0, column=2, sticky='w', padx=(20, 10))
        self.mobo_ddr_var = tk.StringVar()
        self.ddr_combo = ttk.Combobox(mobo_frame, textvariable=self.mobo_ddr_var)
        self.ddr_combo.grid(row=0, column=3, sticky='w')

        # NVMe slider (integer steps)
        nvme_max = int(self.mobos_df['NVMe Slot'].max()) if self.mobos_df is not None else 4
        ttk.Label(mobo_frame, text="Min NVMe Slots:").grid(row=1, column=0, sticky='w', padx=(0, 10))
        self.mobo_nvme_var = tk.IntVar(value=0)
        self.nvme_slider = ttk.Scale(
            mobo_frame, 
            from_=1, 
            to=nvme_max, 
            variable=self.mobo_nvme_var,
            orient='horizontal',
            command=lambda x: self.mobo_nvme_var.set(int(float(x)))  # Force integer values
        )
        self.nvme_slider.grid(row=1, column=1, sticky='ew')
        ttk.Label(mobo_frame, textvariable=self.mobo_nvme_var).grid(row=1, column=2)

        # SATA slider (integer steps)
        sata_max = int(self.mobos_df['SATA Slot'].max()) if self.mobos_df is not None else 8
        ttk.Label(mobo_frame, text="Min SATA Slots:").grid(row=2, column=0, sticky='w', padx=(0, 10))
        self.mobo_sata_var = tk.IntVar(value=0)
        self.sata_slider = ttk.Scale(
            mobo_frame,
            from_=1,
            to=sata_max,
            variable=self.mobo_sata_var,
            orient='horizontal',
            command=lambda x: self.mobo_sata_var.set(int(float(x)))  # Force integer values
        )
        self.sata_slider.grid(row=2, column=1, sticky='ew')
        ttk.Label(mobo_frame, textvariable=self.mobo_sata_var).grid(row=2, column=2)

        # PSU specifications
        psu_frame = ttk.LabelFrame(scrollable_frame, text="PSU Specifications", padding=10)
        psu_frame.pack(fill='x', padx=20, pady=10)

        # Rating dropdown
        ttk.Label(psu_frame, text="80+ Rating:").grid(row=0, column=0, sticky='w', padx=(0, 10))
        self.psu_rating_var = tk.StringVar()
        self.rating_combo = ttk.Combobox(psu_frame, textvariable=self.psu_rating_var)
        self.rating_combo.grid(row=0, column=1, sticky='w')

        # PSU slider (steps of 10)
        ttk.Label(psu_frame, text="Min Wattage:").grid(row=1, column=0, sticky='w', padx=(0, 10))
        self.psu_wattage_var = tk.IntVar(value=0)
        self.watt_slider = ttk.Scale(
            psu_frame,
            from_=0,
            to=1500,
            variable=self.psu_wattage_var,
            orient='horizontal',
            command=lambda x: self.psu_wattage_var.set(int(float(x) // 10 * 10))  # Round to nearest 10
        )
        self.watt_slider.grid(row=1, column=1, sticky='ew')
        ttk.Label(psu_frame, textvariable=self.psu_wattage_var).grid(row=1, column=2)
    
        # Filter action buttons
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill='x', padx=20, pady=20)

        ttk.Button(button_frame, text="Clear All Filters", command=self.clear_filters).pack(side='left', padx=(0, 10))
        ttk.Button(button_frame, text="Apply Filters & Optimize", command=self.optimize_build).pack(side='left')

        # Progress bar
        self.progress_var = tk.StringVar()
        self.progress_label = ttk.Label(scrollable_frame, textvariable=self.progress_var)
        self.progress_label.pack(pady=5)
        self.progress_bar = ttk.Progressbar(scrollable_frame, mode='indeterminate')
        self.progress_bar.pack(fill='x', padx=20, pady=5)
    
    def setup_results_tab(self):
        """Setup the results display tab"""
        self.results_text = scrolledtext.ScrolledText(self.results_tab, wrap=tk.WORD, font=('Consolas', 10))
        self.results_text.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Export button
        export_frame = ttk.Frame(self.results_tab)
        export_frame.pack(fill='x', padx=20, pady=(0, 20))
        ttk.Button(export_frame, text="Export Results", command=self.export_results).pack(side='right')
    
    def load_sample_data(self):
        """Try to load sample data if available"""
        try:
            # This would be called with actual data paths
            # For demo purposes, we'll create sample data
            self.create_sample_data()
            self.data_loaded = True
            self.data_status_label.config(text="Sample data loaded successfully", foreground="green")
            self.populate_game_list()
        except Exception as e:
            self.data_status_label.config(text=f"Could not load sample data: {str(e)}", foreground="orange")
    
    def load_data(self):
        """Load data from specified path"""
        try:
            data_path = self.data_path_var.get()

            # Load all required datasets
            parts_path = os.path.join(data_path, "parts")
            games_path = os.path.join(data_path, "games")

            self.cpus_df = pd.read_csv(os.path.join(parts_path, "CPU_Data.csv"))
            self.gpus_df = pd.read_csv(os.path.join(parts_path, "GPU_Data.csv"))
            self.rams_df = pd.read_csv(os.path.join(parts_path, "RAM_Data.csv"))
            self.mobos_df = pd.read_csv(os.path.join(parts_path, "MOBO_Data.csv"))
            self.psus_df = pd.read_csv(os.path.join(parts_path, "PSU_Data.csv"))
            self.cases_df = pd.read_csv(os.path.join(parts_path, "Case_Data.csv"))
            self.storage_df = pd.read_csv(os.path.join(parts_path, "Storage_Data.csv"))
            self.games_df = pd.read_csv(os.path.join(games_path, "top300.csv"))

            # Convert numeric columns
            self.preprocess_data()

            # Update UI elements with loaded data
            self.update_filters_with_data()
            self.data_loaded = True
            self.data_status_label.config(text="Data loaded successfully", foreground="green")
            self.populate_game_list()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            self.data_status_label.config(text="Data loading failed", foreground="red")

    def custom_rating_sort(self, ratings):
        """Sort PSU ratings in specific order"""
        rating_order = {'White': 1, 'Bronze': 2, 'Silver': 3, 'Gold': 4, 'Platinum': 5}
        return sorted(ratings, key=lambda x: rating_order.get(x, 0) if x else -1)
    
    def update_filters_with_data(self):
        """Update all filters with loaded data"""
        if not hasattr(self, 'brand_filters'):
            return

        if not all([self.cpus_df is not None, self.gpus_df is not None, 
                    self.rams_df is not None, self.mobos_df is not None,
                    self.psus_df is not None, self.cases_df is not None,
                    self.storage_df is not None]):
            return

        # Update brand filters
        brand_data = {
            'cpu_brands': self.cpus_df['Brand'].unique() if self.cpus_df is not None else [],
            'gpu_brands': self.gpus_df['Brand'].unique() if self.gpus_df is not None else [],
            'ram_brands': self.rams_df['Brand'].unique() if self.rams_df is not None else [],
            'mobo_brands': self.mobos_df['Brand'].unique() if self.mobos_df is not None else [],
            'psu_brands': self.psus_df['Brand'].unique() if self.psus_df is not None else [],
            'case_brands': self.cases_df['Brand'].unique() if self.cases_df is not None else [],
            'storage_brands': self.storage_df['Brand'].unique() if self.storage_df is not None else []
        }

        for key, brands in brand_data.items():
            if key in self.brand_filters:
                listbox = self.brand_filters[key]
                listbox.delete(0, tk.END)  # Clear existing items
                for brand in sorted(brands):
                    listbox.insert(tk.END, brand)

        # Update sliders with data-based ranges
        if self.mobos_df is not None:
            nvme_max = int(self.mobos_df['NVMe Slot'].max())
            sata_max = int(self.mobos_df['SATA Slot'].max())

            if hasattr(self, 'nvme_slider'):
                self.nvme_slider.configure(to=nvme_max)

            if hasattr(self, 'sata_slider'):
                self.sata_slider.configure(to=sata_max)

        # Update other combo boxes
        if hasattr(self, 'socket_combo') and self.mobos_df is not None:
            sockets = sorted(self.mobos_df['Socket'].unique())
            self.socket_combo['values'] = [''] + sockets

        if hasattr(self, 'ddr_combo') and self.mobos_df is not None:
            ddr_versions = sorted(self.mobos_df['DDR'].unique())
            self.ddr_combo['values'] = [''] + ddr_versions

        if hasattr(self, 'rating_combo') and self.psus_df is not None:
            ratings = self.psus_df['80+ Rating'].unique()
            sorted_ratings = self.custom_rating_sort(ratings)
            self.rating_combo['values'] = [''] + sorted_ratings

    def preprocess_data(self):
        """Preprocess the loaded data"""
        def safe_to_numeric(val):
            try:
                return pd.to_numeric(val)
            except (ValueError, TypeError):
                return val
        
        # Apply numeric conversion to all dataframes
        for df_name in ['cpus_df', 'gpus_df', 'rams_df', 'mobos_df', 'psus_df', 'cases_df', 'storage_df', 'games_df']:
            df = getattr(self, df_name)
            if df is not None:
                setattr(self, df_name, df.apply(safe_to_numeric))
        
        # Normalize scores if they exist
        if 'Score' in self.cpus_df.columns:
            self.normalize_column(self.cpus_df, 'Score')
        if 'Score' in self.gpus_df.columns:
            self.normalize_column(self.gpus_df, 'Score')
        
        # Normalize RAM capacity if it exists
        if 'Capacity (GB)' in self.rams_df.columns:
            rams_min = self.rams_df["Capacity (GB)"].min()
            rams_max = self.rams_df["Capacity (GB)"].max()
            self.rams_df["Capacity (GB)_Normalized"] = 0.01 + 0.99 * (self.rams_df["Capacity (GB)"] - rams_min) / (rams_max - rams_min)
    
    def normalize_column(self, df, column_name):
        """Normalize a column to 0-1 range"""
        min_val = df[column_name].min()
        max_val = df[column_name].max()
        df[column_name + "_Normalized"] = (df[column_name] - min_val) / (max_val - min_val)
        return df
    
    def populate_game_list(self):
        """Populate the game selection listbox"""
        if self.games_df is not None:
            self.game_listbox.delete(0, tk.END)
            for game in self.games_df['name'].tolist():
                self.game_listbox.insert(tk.END, game)
    
    def clear_filters(self):
        """Clear all filter inputs"""
        # Clear brand selections
        for listbox in self.brand_filters.values():
            listbox.selection_clear(0, tk.END)
        
        # Clear other filters
        self.mobo_socket_var.set('')
        self.mobo_ddr_var.set('')
        self.mobo_nvme_var.set(int(self.mobos_df['NVMe Slot'].min()) if self.mobos_df is not None else 0)
        self.mobo_sata_var.set(int(self.mobos_df['SATA Slot'].min()) if self.mobos_df is not None else 0)
        self.psu_rating_var.set('')
        self.psu_wattage_var.set(int(self.psus_df['Wattage'].min()) if self.psus_df is not None else 0)
    
    def apply_filters(self):
        """Apply filters to the datasets"""
        # Start with original data
        cpus_filtered = self.cpus_df.copy()
        gpus_filtered = self.gpus_df.copy()
        rams_filtered = self.rams_df.copy()
        mobos_filtered = self.mobos_df.copy()
        psus_filtered = self.psus_df.copy()
        cases_filtered = self.cases_df.copy()
        storage_filtered = self.storage_df.copy()
        
        # Apply brand filters
        for key, listbox in self.brand_filters.items():
            selected_indices = listbox.curselection()
            if selected_indices:
                selected_brands = [listbox.get(i) for i in selected_indices]
                if key == 'cpu_brands':
                    cpus_filtered = cpus_filtered[cpus_filtered['Brand'].isin(selected_brands)]
                elif key == 'gpu_brands':
                    gpus_filtered = gpus_filtered[gpus_filtered['Brand'].isin(selected_brands)]
                # ... similar for other components
        
        # Apply motherboard filters
        if self.mobo_socket_var.get():
            mobos_filtered = mobos_filtered[mobos_filtered['Socket'] == self.mobo_socket_var.get()]
        
        if self.mobo_ddr_var.get():
            mobos_filtered = mobos_filtered[mobos_filtered['DDR'] == self.mobo_ddr_var.get()]
        
        mobos_filtered = mobos_filtered[
            (mobos_filtered['NVMe Slot'] >= self.mobo_nvme_var.get()) &
            (mobos_filtered['SATA Slot'] >= self.mobo_sata_var.get())
        ]
        
        # Apply PSU filters
        if self.psu_rating_var.get():
            psus_filtered = psus_filtered[psus_filtered['80+ Rating'] == self.psu_rating_var.get()]
        
        psus_filtered = psus_filtered[psus_filtered['Wattage'] >= self.psu_wattage_var.get()]
        
        return (cpus_filtered, gpus_filtered, rams_filtered, mobos_filtered, 
                psus_filtered, cases_filtered, storage_filtered)
    
    def optimize_build(self):
        """Run optimization in separate thread"""
        if not self.data_loaded:
            messagebox.showerror("Error", "Please load data first")
            return
            
        selected_indices = self.game_listbox.curselection()
        if not selected_indices:
            messagebox.showerror("Error", "Please select at least one game")
            return
            
        try:
            budget = float(self.budget_var.get())
            min_storage = int(self.min_storage_var.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid budget and storage values")
            return

        # Start progress bar
        self.progress_bar.start(10)
        self.progress_var.set("Optimizing build...")
        
        # Run optimization in thread
        self.update_thread = threading.Thread(
            target=self._optimize_thread,
            args=(selected_indices, budget, min_storage)
        )
        self.update_thread.daemon = True
        self.update_thread.start()
        
        # Check queue periodically
        self.root.after(100, self._check_optimization)

    def _optimize_thread(self, selected_indices, budget, min_storage):
        """Thread worker for optimization"""
        try:
            # Apply filters
            filtered_components = self.apply_filters()
            
            # Get selected games
            selected_games = [self.game_listbox.get(i) for i in selected_indices]
            
            results = []
            for game_name in selected_games:
                game_data = self.games_df[self.games_df['name'] == game_name].iloc[0]
                result = self.run_optimization(game_data, budget, min_storage, *filtered_components)
                results.append((game_name, result))
            
            # Put results in queue
            self.queue.put(('success', results))
        except Exception as e:
            self.queue.put(('error', str(e)))

    def _check_optimization(self):
        """Check if optimization is complete"""
        try:
            msg_type, data = self.queue.get_nowait()

            # Stop progress bar
            self.progress_bar.stop()
            self.progress_var.set("")

            if msg_type == 'error':
                messagebox.showerror("Error", f"Optimization failed: {data}")
            else:
                self.display_results(data)
                self.notebook.select(self.results_tab)
        except Empty:
            # Continue checking
            self.root.after(100, self._check_optimization)
    
    def get_compatibility_dict(self, type_key, df1, df2, col1, col2):
        """Get cached compatibility dictionary or create new one"""
        cache_key = f"{type_key}_{id(df1)}_{id(df2)}"

        if cache_key not in self._compatibility_cache:
            self._compatibility_cache[cache_key] = {
                (i, j): int(df1.iloc[i][col1] == df2.iloc[j][col2])
                for i in range(len(df1)) 
                for j in range(len(df2))
            }

        return self._compatibility_cache[cache_key]
    
    def run_optimization(self, game_data, budget, min_storage, cpusf, gpusf, ramsf, mobosf, psusf, casesf, storage_df):
        """Run optimization with PuLP"""
        try:
            # Filter storage by type
            m2sf = storage_df[storage_df['Storage Type'] == 'M.2 SSD'].reset_index(drop=True)
            satasf = storage_df[storage_df['Storage Type'] == 'SATA SSD'].reset_index(drop=True)

            # Create ranges for indices
            lenCPU = range(len(cpusf))
            lenGPU = range(len(gpusf))
            lenRAM = range(len(ramsf))
            lenPSU = range(len(psusf))
            lenMOBO = range(len(mobosf))
            lenCase = range(len(casesf))
            lenM2 = range(len(m2sf))
            lenSATA = range(len(satasf))

            # Use cached compatibility dictionaries
            mobo_cpu_compatibility_dict = self.get_compatibility_dict(
                'cpu_mobo', cpusf, mobosf, 'Socket', 'Socket'
            )
            mobo_case_compatibility_dict = self.get_compatibility_dict(
                'case_mobo', casesf, mobosf, 'Size', 'Size'
            )
            mobo_ram_compatibility_dict = self.get_compatibility_dict(
                'ram_mobo', ramsf, mobosf, 'DDR', 'DDR'
            )

            # Configuration
            min_m2 = 1
            min_sata = 1
            use_sata = self.use_sata_var.get()
            ram_16 = self.ram_16_var.get()

            def add_leading_zeros(i, width=4):
                return str(i).zfill(width)

            # Create the LP problem
            from pulp import LpProblem, LpMaximize, LpVariable, lpSum, value, LpStatus
            problem = LpProblem("Desktop_Optimization", LpMaximize)

            # Define variables
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
            sata_vars_test = [LpVariable(f"sata_{i}", cat="Binary") for i in lenSATA]
            sata_count = LpVariable("sata_count", lowBound=1, upBound=mobosf['SATA Slot'].max(), cat="Integer")
            sata_count_selected = [LpVariable(f"sata_count_selected_{add_leading_zeros(i)}", lowBound=0, upBound=mobosf['SATA Slot'].max(), cat="Integer") for i in lenSATA] #Auxiliary Variable


            ''' ---Part Count Constraint--- '''
            # Only select exactly one of each major part
            problem += lpSum(cpu_vars) == 1, "Select_One_CPU"
            problem += lpSum(gpu_vars) == 1, "Select_One_GPU"
            problem += lpSum(ram_vars) == 1, "Select_One_RAM"
            problem += lpSum(psu_vars) == 1, "Select_One_PSU"
            problem += lpSum(mobo_vars) == 1, "Select_One_MOBO"
            problem += lpSum(case_vars) == 1, "Select_One_Case"

            ''' ---GPU Wattage Constraints--- '''
            # GPU x Power Supply
            problem += (
                lpSum(gpu_vars[i] * gpusf.iloc[i]["Recommended Power"] for i in lenGPU) <=
                lpSum(psu_vars[i] * psusf.iloc[i]["Wattage"] for i in lenPSU),
                "PSU_Power_Constraint",
            )

            ''' Storage Constraints '''
            problem += (
                lpSum(m2_count_selected[i] * m2sf.iloc[i]["Capacity (GB)"] for i in lenM2) +
                lpSum(sata_count_selected[i] * satasf.iloc[i]["Capacity (GB)"] for i in lenSATA)
                >= min_storage,
                "Minimum_Total_Storage"
            )

            ''' Default: 1 M.2 SSD 512GB Storage '''
            ''' M.2 SSD NVMe '''
            problem += m2_count >= min_m2, "Minimum_M2"
            problem += m2_count <= lpSum(mobo_vars[i] * mobosf.iloc[i]['NVMe Slot'] for i in lenMOBO)

            for i in lenM2:
                problem += m2_count_selected[i] <= m2_count
                problem += m2_count_selected[i] <= mobosf['NVMe Slot'].max() * m2_vars[i]

            problem += lpSum(m2_count_selected[i] for i in lenM2) == m2_count

            ''' SATA SSD '''
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

            ''' ---Motherboard Constraints--- '''
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

            ''' ---RAM Selection Algorithm--- '''
            # For each RAM Model, link count to total count
            for i in lenRAM:
                problem += ram_count_selected[i] <= ram_count
                problem += ram_count_selected[i] <= 4 * ram_vars[i]
                problem += ram_count_selected[i] >= ram_count - (1 - ram_vars[i]) * 4
                problem += ram_count_selected[i] >= 0

            ''' ---Game Constraints--- '''
            problem += (
                lpSum(cpu_vars[i] * cpusf.iloc[i]["Score"] for i in lenCPU) >= game_data["CPU"],
                "Game_CPU_Constraint"
            )

            problem += (
                lpSum(gpu_vars[i] * gpusf.iloc[i]["Score"] for i in lenGPU) >= game_data["GPU"],
                "Game_GPU_Constraint"
            )

            memory_requirement = int(game_data["memory"])
            if (ram_16 and memory_requirement <= 16): # If minimum RAM is set to 16GB
                memory_requirement = 16

            problem += (
                lpSum(ram_count_selected[i] * ramsf.iloc[i]['Capacity (GB)'] for i in lenRAM) >= memory_requirement,
                "Game_Memory_Constraint"
            )

            ''' ---Cost Function--- '''
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

            problem += total_cost <= budget, "Budget_Constraint"

            ''' ---Performance Function---'''
            # Parameters
            cpu_weight = 0.1
            gpu_weight = 0.895
            ram_weight = 0.005
            dual_channel_bonus_value = 0.02

            # Extra variables
            dual_channel_bonus_var = LpVariable("dual_channel_bonus_var", cat="Binary")

            # Function
            total_performance = (
                cpu_weight * lpSum(cpu_vars[i] * cpusf.iloc[i]["Score_Normalized"] for i in lenCPU) +
                gpu_weight * lpSum(gpu_vars[i] * gpusf.iloc[i]["Score_Normalized"] for i in lenGPU) +
                ram_weight * lpSum(ram_count_selected[i] * ramsf.iloc[i]["Capacity (GB)_Normalized"] for i in lenRAM) +
                dual_channel_bonus_value * dual_channel_bonus_var
            )

            '''
            Dual-Channel Activation TLDR:
            Solver has to toggle on-off the binary "dual_channel_bonus_var" to figure out
            optimal ram selection based on budget & game.
            '''
            problem += total_performance
            problem += ram_count_selected >= 2 * dual_channel_bonus_var, "Dual-Channel Activation Condition"

            # Solve the problem
            status = problem.solve()

            if LpStatus[status] != "Optimal":
                return "No feasible solution found"

            # Get selected components
            selected_cpu_idx = next(i for i in lenCPU if value(cpu_vars[i]) == 1)
            selected_gpu_idx = next(i for i in lenGPU if value(gpu_vars[i]) == 1)
            selected_psu_idx = next(i for i in lenPSU if value(psu_vars[i]) == 1)
            selected_mobo_idx = next(i for i in lenMOBO if value(mobo_vars[i]) == 1)
            selected_case_idx = next(i for i in lenCase if value(case_vars[i]) == 1)
            selected_ram_idx = next(i for i in lenRAM if value(ram_vars[i]) == 1)
            selected_ram_count = int(value(ram_count_selected[selected_ram_idx]))

            # Get selected storage
            selected_m2 = [(i, m2sf.iloc[i], int(value(m2_count_selected[i]))) 
                          for i in lenM2 if value(m2_count_selected[i]) > 0]
            selected_sata = [(i, satasf.iloc[i], int(value(sata_count_selected[i]))) 
                            for i in lenSATA if value(sata_count_selected[i]) > 0]

            # Return results dictionary
            result = {
                'cpu': {
                    'name': cpusf.iloc[selected_cpu_idx]['Name'],
                    'brand': cpusf.iloc[selected_cpu_idx]['Brand'],
                    'price': cpusf.iloc[selected_cpu_idx]['Price'],
                    'score': cpusf.iloc[selected_cpu_idx]['Score']
                },
                'gpu': {
                    'name': gpusf.iloc[selected_gpu_idx]['Name'],
                    'brand': gpusf.iloc[selected_gpu_idx]['Brand'],
                    'price': gpusf.iloc[selected_gpu_idx]['Price'],
                    'score': gpusf.iloc[selected_gpu_idx]['Score']
                },
                'ram': {
                    'name': ramsf.iloc[selected_ram_idx]['Name'],
                    'brand': ramsf.iloc[selected_ram_idx]['Brand'],
                    'price': ramsf.iloc[selected_ram_idx]['Price'] * selected_ram_count,
                    'capacity': ramsf.iloc[selected_ram_idx]['Capacity (GB)'] * selected_ram_count,
                    'modules': selected_ram_count
                },
                'psu': {
                    'name': psusf.iloc[selected_psu_idx]['Name'],
                    'brand': psusf.iloc[selected_psu_idx]['Brand'],
                    'price': psusf.iloc[selected_psu_idx]['Price'],
                    'wattage': psusf.iloc[selected_psu_idx].get('Wattage', ''),
                    'rating': psusf.iloc[selected_psu_idx].get('80+ Rating', '')
                },
                'mobo': {
                    'name': mobosf.iloc[selected_mobo_idx]['Name'],
                    'brand': mobosf.iloc[selected_mobo_idx]['Brand'],
                    'price': mobosf.iloc[selected_mobo_idx]['Price'],
                    'socket': mobosf.iloc[selected_mobo_idx].get('Socket', ''),
                    'ddr': mobosf.iloc[selected_mobo_idx].get('DDR', '')
                },
                'case': {
                    'name': casesf.iloc[selected_case_idx]['Name'],
                    'brand': casesf.iloc[selected_case_idx]['Brand'],
                    'price': casesf.iloc[selected_case_idx]['Price']
                },
                'storage': {
                    'm2': [{'brand': row['Brand'], 
                           'name': row['Name'],
                           'capacity': row['Capacity (GB)'],
                           'price': row['Price'] * count} 
                          for _, row, count in selected_m2],
                    'sata': [{'brand': row['Brand'],
                             'name': row['Name'],
                             'capacity': row['Capacity (GB)'],
                             'price': row['Price'] * count}
                            for _, row, count in selected_sata]
                },
                'total_cost': value(total_cost)
            }

            return result

        except Exception as e:
            return f"Optimization error: {str(e)}"

    def select_best_component(self, df, min_score, score_column, max_price):
        """Select best component within constraints"""
        if df.empty:
            return {'name': 'None available', 'price': 0, 'score': 0}
        
        # Filter by minimum requirements and price
        suitable = df[(df[score_column] >= min_score) & (df['Price'] <= max_price)]
        
        if suitable.empty:
            # Fallback to cheapest that meets requirements
            suitable = df[df[score_column] >= min_score]
            if suitable.empty:
                return {'name': 'Requirements too high', 'price': 0, 'score': 0}
        
        # Select best value (highest score per price ratio)
        suitable['value_ratio'] = suitable[score_column] / suitable['Price']
        best = suitable.loc[suitable['value_ratio'].idxmax()]
        
        return {
            'name': best['Name'],
            'brand': best['Brand'],
            'price': best['Price'],
            'score': best[score_column]
        }
    
    def select_best_ram(self, df, min_capacity, max_price):
        """Select best RAM configuration"""
        if df.empty:
            return {'name': 'None available', 'price': 0, 'capacity': 0}
        
        # Ensure minimum 16GB if option is selected
        if self.ram_16_var.get() and min_capacity < 16:
            min_capacity = 16
        
        # Find suitable RAM
        suitable = df[df['Price'] <= max_price]
        if suitable.empty:
            suitable = df
        
        # For simplicity, select based on capacity and price efficiency
        suitable['capacity_per_price'] = suitable['Capacity (GB)'] / suitable['Price']
        best = suitable.loc[suitable['capacity_per_price'].idxmax()]
        
        # Calculate how many modules needed
        modules_needed = max(1, int(min_capacity / best['Capacity (GB)']))
        if modules_needed > 4:  # Assume max 4 RAM slots
            modules_needed = 4
        
        return {
            'name': best['Name'],
            'brand': best['Brand'],
            'price': best['Price'] * modules_needed,
            'capacity': best['Capacity (GB)'] * modules_needed,
            'modules': modules_needed
        }
    
    def select_compatible_mobo(self, df, max_price):
        """Select compatible motherboard"""
        if df.empty:
            return {'name': 'None available', 'price': 0}
        
        suitable = df[df['Price'] <= max_price]
        if suitable.empty:
            suitable = df
        
        # Select cheapest suitable motherboard
        best = suitable.loc[suitable['Price'].idxmin()]
        
        return {
            'name': best['Name'],
            'brand': best['Brand'],
            'price': best['Price'],
            'socket': best.get('Socket', 'Unknown'),
            'ddr': best.get('DDR', 'Unknown')
        }
    
    def select_adequate_psu(self, df, max_price):
        """Select adequate PSU"""
        if df.empty:
            return {'name': 'None available', 'price': 0}
        
        suitable = df[df['Price'] <= max_price]
        if suitable.empty:
            suitable = df
        
        # Select based on efficiency and price
        best = suitable.loc[suitable['Price'].idxmin()]
        
        return {
            'name': best['Name'],
            'brand': best['Brand'],
            'price': best['Price'],
            'wattage': best.get('Wattage', 'Unknown'),
            'rating': best.get('80+ Rating', 'Unknown')
        }
    
    def select_cheapest_compatible_case(self, df, max_price):
        """Select cheapest compatible case"""
        if df.empty:
            return {'name': 'None available', 'price': 0}
        
        suitable = df[df['Price'] <= max_price]
        if suitable.empty:
            suitable = df
        
        best = suitable.loc[suitable['Price'].idxmin()]
        
        return {
            'name': best['Name'],
            'brand': best['Brand'],
            'price': best['Price']
        }
    
    def display_results(self, results):
        """Display optimization results"""
        self.results_text.delete(1.0, tk.END)

        for game_name, result in results:
            self.results_text.insert(tk.END, f"🎮 Recommended PC Build for: {game_name}\n")
            self.results_text.insert(tk.END, "-" * 50 + "\n")

            if isinstance(result, str):
                self.results_text.insert(tk.END, f"Error: {result}\n\n")
                continue

            # Show requirements if available
            cpu_req = result.get('cpu', {}).get('score', None)
            gpu_req = result.get('gpu', {}).get('score', None)
            ram_req = result.get('ram', {}).get('capacity', None)
            if cpu_req is not None:
                self.results_text.insert(tk.END, f"🧠 Required CPU Benchmark Score : {cpu_req}\n")
            if gpu_req is not None:
                self.results_text.insert(tk.END, f"🎮 Required GPU Benchmark Score : {gpu_req}\n")
            if ram_req is not None:
                self.results_text.insert(tk.END, f"🧠 Required Memory              : {ram_req} GB\n")
            self.results_text.insert(tk.END, "-" * 50 + "\n")

            # CPU
            cpu = result.get('cpu', {})
            self.results_text.insert(tk.END, f"🧩 CPU: {cpu.get('brand', '')} {cpu.get('name', '')}\n")
            self.results_text.insert(tk.END, f"   └─ Price  : RM {cpu.get('price', 0):.2f}\n")
            self.results_text.insert(tk.END, f"   └─ Score  : {cpu.get('score', 0)}\n\n")

            # GPU
            gpu = result.get('gpu', {})
            self.results_text.insert(tk.END, f"🖼️ GPU: {gpu.get('brand', '')} {gpu.get('name', '')}\n")
            self.results_text.insert(tk.END, f"   └─ Price  : RM {gpu.get('price', 0):.2f}\n")
            self.results_text.insert(tk.END, f"   └─ Score  : {gpu.get('score', 0)}\n\n")

            # RAM
            ram = result.get('ram', {})
            self.results_text.insert(tk.END, f"🧠 RAM: {ram.get('brand', '')} {ram.get('name', '')}\n")
            self.results_text.insert(tk.END, f"   └─ Modules      : {ram.get('modules', 1)}\n")
            self.results_text.insert(tk.END, f"   └─ Capacity     : {ram.get('capacity', 0)} GB total\n")
            self.results_text.insert(tk.END, f"   └─ Total Price  : RM {ram.get('price', 0):.2f}\n\n")

            # PSU
            psu = result.get('psu', {})
            self.results_text.insert(tk.END, f"🔌 PSU: {psu.get('brand', '')} {psu.get('name', '')}\n")
            self.results_text.insert(tk.END, f"   └─ Price: RM {psu.get('price', 0):.2f}\n")
            if 'wattage' in psu:
                self.results_text.insert(tk.END, f"   └─ Wattage: {psu.get('wattage', '')}\n")
            if 'rating' in psu:
                self.results_text.insert(tk.END, f"   └─ 80+ Rating: {psu.get('rating', '')}\n")
            self.results_text.insert(tk.END, "\n")

            # Motherboard
            mobo = result.get('mobo', {})
            self.results_text.insert(tk.END, f"🧩 Motherboard: {mobo.get('brand', '')} {mobo.get('name', '')}\n")
            self.results_text.insert(tk.END, f"   └─ Price: RM {mobo.get('price', 0):.2f}\n")
            if 'socket' in mobo:
                self.results_text.insert(tk.END, f"   └─ Socket: {mobo.get('socket', '')}\n")
            if 'ddr' in mobo:
                self.results_text.insert(tk.END, f"   └─ DDR: {mobo.get('ddr', '')}\n")
            self.results_text.insert(tk.END, "\n")

            # Case
            case = result.get('case', {})
            self.results_text.insert(tk.END, f"🖥️ Case: {case.get('brand', '')} {case.get('name', '')}\n")
            self.results_text.insert(tk.END, f"   └─ Price: RM {case.get('price', 0):.2f}\n\n")

            # Storage
            storage = result.get('storage', {})

            # M.2 SSDs
            self.results_text.insert(tk.END, "💾 M.2 SSDs:\n")
            m2_drives = storage.get('m2', [])
            if m2_drives:
                total_m2_price = 0
                for drive in m2_drives:
                    self.results_text.insert(tk.END, 
                        f"   └─ {drive['brand']} {drive['name']} "
                        f"({drive['capacity']}GB, RM {drive['price']:.2f})\n")
                    total_m2_price += drive['price']
                self.results_text.insert(tk.END, f"   └─ Total M.2 Price: RM {total_m2_price:.2f}\n")
            else:
                self.results_text.insert(tk.END, "   └─ None selected\n")
            self.results_text.insert(tk.END, "\n")

            # SATA SSDs
            self.results_text.insert(tk.END, "💽 SATA SSDs:\n")
            sata_drives = storage.get('sata', [])
            if sata_drives:
                total_sata_price = 0
                for drive in sata_drives:
                    self.results_text.insert(tk.END,
                        f"   └─ {drive['brand']} {drive['name']} "
                        f"({drive['capacity']}GB, RM {drive['price']:.2f})\n")
                    total_sata_price += drive['price']
                self.results_text.insert(tk.END, f"   └─ Total SATA Price: RM {total_sata_price:.2f}\n")
            else:
                self.results_text.insert(tk.END, "   └─ None selected\n")
            self.results_text.insert(tk.END, "\n")

            # Summary
            self.results_text.insert(tk.END, "=" * 50 + "\n")
            total_cost = result.get('total_cost', 0)
            self.results_text.insert(tk.END, f"💸 Total Build Cost   : RM {total_cost:.2f}\n")
            self.results_text.insert(tk.END, "=" * 50 + "\n")
            self.results_text.insert(tk.END, "❗ Prices may vary depending on market fluctuations ❗\n\n")
    
    def export_results(self):
        """Export results to a text file"""
        results_content = self.results_text.get(1.0, tk.END)
        if not results_content.strip():
            messagebox.showinfo("Export", "No results to export.")
            return
        try:
            file_path = tk.filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if file_path:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(results_content)
                messagebox.showinfo("Export", f"Results exported to {file_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export results: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PCBuildOptimizerGUI(root)
    root.mainloop()