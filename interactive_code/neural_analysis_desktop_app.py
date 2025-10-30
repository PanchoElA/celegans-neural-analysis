"""
🧠 C. elegans Neural Analysis Desktop Application
Interactive Desktop App for Neural Dynamics Visualization

Esta aplicación permite explorar de forma interactiva:
- Visualizaciones PCA (1D, 2D, 3D)
- Análisis de derivadas del firing rate
- Filtrado por comportamientos específicos
- Preprocesamiento de datos
- Visualizaciones interactivas con zoom y rotación
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from tkinter import font
import os
import sys
import webbrowser
import tempfile
from threading import Thread
import time

# Importar dependencias científicas
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.stats import zscore
import matplotlib.pyplot as plt
try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except ImportError:
    FigureCanvasTkAgg = None
import matplotlib.colors as mcolors

class NeuralAnalysisApp:
    """Aplicación principal para análisis neural interactivo"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 C. elegans Neural Analysis - Interactive Desktop App")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Variables de datos
        self.data = None
        self.neural_data = None
        self.time_points = None
        self.neuron_names = None
        self.preprocessed_data = None
        self.pca_results = None
        self.derivatives = None
        self.behavioral_data = None
        
        # Variables de estado
        self.current_plot_type = tk.StringVar(value="PCA_3D")
        self.current_preprocessing = tk.StringVar(value="StandardScaler")
        self.current_behavior = tk.StringVar(value="All")
        self.show_derivatives = tk.BooleanVar(value=False)
        self.color_scheme = tk.StringVar(value="viridis")
        
        # Variables PCA
        self.pc_x = tk.IntVar(value=1)
        self.pc_y = tk.IntVar(value=2)
        self.pc_z = tk.IntVar(value=3)
        self.n_components = tk.IntVar(value=10)
        
        # Configurar interfaz
        self.setup_ui()
        self.load_default_data()
        
    def setup_ui(self):
        """Configurar la interfaz de usuario"""
        
        # Estilo
        style = ttk.Style()
        style.theme_use('clam')
        
        # Frame principal - división izquierda/derecha
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Panel de controles (izquierda)
        self.create_control_panel(main_frame)
        
        # Panel de visualización (derecha)
        self.create_visualization_panel(main_frame)
        
        # Barra de estado
        self.create_status_bar()
        
    def create_control_panel(self, parent):
        """Crear panel de controles"""
        
        # Frame de controles con scroll
        control_frame = ttk.Frame(parent, width=400)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)
        
        # Canvas y scrollbar para hacer scroll
        canvas = tk.Canvas(control_frame, bg='#f8f9fa')
        scrollbar = ttk.Scrollbar(control_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Título del panel
        title_label = ttk.Label(scrollable_frame, text="🎛️ Panel de Controles", 
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=(10, 20))
        
        # Sección 1: Carga de datos
        self.create_data_section(scrollable_frame)
        
        # Sección 2: Preprocesamiento
        self.create_preprocessing_section(scrollable_frame)
        
        # Sección 3: Tipo de visualización
        self.create_visualization_section(scrollable_frame)
        
        # Sección 4: Configuración PCA
        self.create_pca_section(scrollable_frame)
        
        # Sección 5: Comportamientos
        self.create_behavior_section(scrollable_frame)
        
        # Sección 6: Colores y estilo
        self.create_style_section(scrollable_frame)
        
        # Sección 7: Acciones
        self.create_action_section(scrollable_frame)
        
    def create_data_section(self, parent):
        """Sección de carga de datos"""
        
        # Frame de sección
        section_frame = ttk.LabelFrame(parent, text="📊 Datos Neurales", padding=10)
        section_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Botón cargar datos
        load_btn = ttk.Button(section_frame, text="📁 Cargar Archivo CSV", 
                             command=self.load_data)
        load_btn.pack(fill=tk.X, pady=(0, 5))
        
        # Info del dataset
        self.data_info_label = ttk.Label(section_frame, text="No hay datos cargados", 
                                        foreground="gray")
        self.data_info_label.pack(anchor=tk.W)
        
        # Botón usar datos por defecto
        default_btn = ttk.Button(section_frame, text="🔄 Usar Datos por Defecto", 
                                command=self.load_default_data)
        default_btn.pack(fill=tk.X, pady=(5, 0))
        
    def create_preprocessing_section(self, parent):
        """Sección de preprocesamiento"""
        
        section_frame = ttk.LabelFrame(parent, text="⚙️ Preprocesamiento", padding=10)
        section_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Método de escalado
        ttk.Label(section_frame, text="Método de Escalado:").pack(anchor=tk.W)
        scaling_combo = ttk.Combobox(section_frame, textvariable=self.current_preprocessing,
                                    values=["StandardScaler", "MinMaxScaler", "RobustScaler", "Z-Score", "None"])
        scaling_combo.pack(fill=tk.X, pady=(0, 10))
        scaling_combo.bind('<<ComboboxSelected>>', lambda e: self.update_preprocessing())
        
        # Opciones de filtrado
        self.apply_filter = tk.BooleanVar(value=False)
        filter_check = ttk.Checkbutton(section_frame, text="Aplicar Filtro Temporal", 
                                      variable=self.apply_filter,
                                      command=self.update_preprocessing)
        filter_check.pack(anchor=tk.W)
        
        # Frame para parámetros de filtro
        filter_frame = ttk.Frame(section_frame)
        filter_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(filter_frame, text="Ventana:").pack(side=tk.LEFT)
        self.filter_window = tk.IntVar(value=5)
        filter_spin = ttk.Spinbox(filter_frame, from_=3, to=21, width=5, 
                                 textvariable=self.filter_window,
                                 command=self.update_preprocessing)
        filter_spin.pack(side=tk.LEFT, padx=(5, 0))
        
    def create_visualization_section(self, parent):
        """Sección de tipo de visualización"""
        
        section_frame = ttk.LabelFrame(parent, text="📈 Tipo de Visualización", padding=10)
        section_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Radio buttons para tipo de plot
        plot_types = [
            ("📊 PC vs Tiempo", "PC_vs_Time"),
            ("📈 PC vs PC (2D)", "PC_vs_PC"), 
            ("🎆 PCA 3D", "PCA_3D"),
            ("🌊 Derivadas vs Tiempo", "Derivatives_vs_Time"),
            ("🔄 Derivadas 2D", "Derivatives_2D"),
            ("🎯 Derivadas 3D", "Derivatives_3D")
        ]
        
        for text, value in plot_types:
            radio = ttk.Radiobutton(section_frame, text=text, 
                                   variable=self.current_plot_type, value=value,
                                   command=self.update_visualization)
            radio.pack(anchor=tk.W, pady=2)
            
        # Checkbox para mostrar derivadas
        deriv_check = ttk.Checkbutton(section_frame, text="🔢 Mostrar Derivadas del FR", 
                                     variable=self.show_derivatives,
                                     command=self.update_visualization)
        deriv_check.pack(anchor=tk.W, pady=(10, 0))
        
    def create_pca_section(self, parent):
        """Sección de configuración PCA"""
        
        section_frame = ttk.LabelFrame(parent, text="🎯 Configuración PCA", padding=10)
        section_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Número de componentes
        comp_frame = ttk.Frame(section_frame)
        comp_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(comp_frame, text="Componentes:").pack(side=tk.LEFT)
        comp_spin = ttk.Spinbox(comp_frame, from_=3, to=50, width=5,
                               textvariable=self.n_components,
                               command=self.update_pca)
        comp_spin.pack(side=tk.LEFT, padx=(5, 0))
        
        # Selección de PCs para visualización
        pc_frame = ttk.LabelFrame(section_frame, text="Componentes a Visualizar", padding=5)
        pc_frame.pack(fill=tk.X, pady=(0, 10))
        
        # PC X
        x_frame = ttk.Frame(pc_frame)
        x_frame.pack(fill=tk.X, pady=2)
        ttk.Label(x_frame, text="PC X:").pack(side=tk.LEFT)
        x_spin = ttk.Spinbox(x_frame, from_=1, to=10, width=5, textvariable=self.pc_x,
                            command=self.update_visualization)
        x_spin.pack(side=tk.LEFT, padx=(5, 0))
        
        # PC Y
        y_frame = ttk.Frame(pc_frame)
        y_frame.pack(fill=tk.X, pady=2)
        ttk.Label(y_frame, text="PC Y:").pack(side=tk.LEFT)
        y_spin = ttk.Spinbox(y_frame, from_=1, to=10, width=5, textvariable=self.pc_y,
                            command=self.update_visualization)
        y_spin.pack(side=tk.LEFT, padx=(5, 0))
        
        # PC Z
        z_frame = ttk.Frame(pc_frame)
        z_frame.pack(fill=tk.X, pady=2)
        ttk.Label(z_frame, text="PC Z:").pack(side=tk.LEFT)
        z_spin = ttk.Spinbox(z_frame, from_=1, to=10, width=5, textvariable=self.pc_z,
                            command=self.update_visualization)
        z_spin.pack(side=tk.LEFT, padx=(5, 0))
        
    def create_behavior_section(self, parent):
        """Sección de comportamientos"""
        
        section_frame = ttk.LabelFrame(parent, text="🎭 Análisis por Comportamiento", padding=10)
        section_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Combo para seleccionar comportamiento
        ttk.Label(section_frame, text="Filtrar por Comportamiento:").pack(anchor=tk.W)
        behavior_combo = ttk.Combobox(section_frame, textvariable=self.current_behavior,
                                     values=["All", "Forward", "Reverse", "Pause", "Turn"])
        behavior_combo.pack(fill=tk.X, pady=(0, 10))
        behavior_combo.bind('<<ComboboxSelected>>', lambda e: self.update_visualization())
        
        # Botones para identificar comportamientos
        id_btn = ttk.Button(section_frame, text="🔍 Identificar Comportamientos",
                           command=self.identify_behaviors)
        id_btn.pack(fill=tk.X, pady=(0, 5))
        
        # Info de comportamientos
        self.behavior_info_label = ttk.Label(section_frame, text="Comportamientos no identificados", 
                                            foreground="gray")
        self.behavior_info_label.pack(anchor=tk.W)
        
    def create_style_section(self, parent):
        """Sección de colores y estilo"""
        
        section_frame = ttk.LabelFrame(parent, text="🎨 Colores y Estilo", padding=10)
        section_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Esquema de colores
        ttk.Label(section_frame, text="Esquema de Colores:").pack(anchor=tk.W)
        color_combo = ttk.Combobox(section_frame, textvariable=self.color_scheme,
                                  values=["viridis", "plasma", "inferno", "magma", "cividis", 
                                         "turbo", "rainbow", "jet", "hot", "cool"])
        color_combo.pack(fill=tk.X, pady=(0, 10))
        color_combo.bind('<<ComboboxSelected>>', lambda e: self.update_visualization())
        
        # Opciones de marcadores
        self.show_trajectory = tk.BooleanVar(value=True)
        traj_check = ttk.Checkbutton(section_frame, text="Mostrar Trayectoria", 
                                    variable=self.show_trajectory,
                                    command=self.update_visualization)
        traj_check.pack(anchor=tk.W)
        
        self.show_points = tk.BooleanVar(value=True)
        points_check = ttk.Checkbutton(section_frame, text="Mostrar Puntos", 
                                      variable=self.show_points,
                                      command=self.update_visualization)
        points_check.pack(anchor=tk.W)
        
    def create_action_section(self, parent):
        """Sección de acciones"""
        
        section_frame = ttk.LabelFrame(parent, text="🚀 Acciones", padding=10)
        section_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Botón actualizar
        update_btn = ttk.Button(section_frame, text="🔄 Actualizar Visualización",
                               command=self.update_visualization)
        update_btn.pack(fill=tk.X, pady=(0, 5))
        
        # Botón exportar
        export_btn = ttk.Button(section_frame, text="💾 Exportar Gráfico",
                               command=self.export_plot)
        export_btn.pack(fill=tk.X, pady=(0, 5))
        
        # Botón resetear
        reset_btn = ttk.Button(section_frame, text="🔄 Resetear Vista",
                              command=self.reset_view)
        reset_btn.pack(fill=tk.X)
        
    def create_visualization_panel(self, parent):
        """Crear panel de visualización"""
        
        # Frame de visualización
        viz_frame = ttk.Frame(parent)
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Título del panel
        title_label = ttk.Label(viz_frame, text="📊 Visualización Interactiva", 
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=(10, 10))
        
        # Frame para el plot (se llenará con plotly)
        self.plot_frame = ttk.Frame(viz_frame, relief=tk.SUNKEN, borderwidth=2)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Label placeholder
        self.plot_placeholder = ttk.Label(self.plot_frame, 
                                         text="🧠 Cargar datos para comenzar la visualización",
                                         font=('Arial', 12),
                                         foreground="gray")
        self.plot_placeholder.pack(expand=True)
        
    def create_status_bar(self):
        """Crear barra de estado"""
        
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_label = ttk.Label(self.status_bar, text="Listo", relief=tk.SUNKEN)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2, pady=2)
        
        # Info adicional
        self.info_label = ttk.Label(self.status_bar, text="C. elegans Neural Analysis v1.0", 
                                   relief=tk.SUNKEN)
        self.info_label.pack(side=tk.RIGHT, padx=2, pady=2)
        
    def update_status(self, message):
        """Actualizar mensaje de estado"""
        self.status_label.config(text=message)
        self.root.update()
        
    def load_data(self):
        """Cargar datos desde archivo CSV"""
        
        file_path = filedialog.askopenfilename(
            title="Seleccionar archivo de datos neurales",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.update_status("Cargando datos...")
                self.data = pd.read_csv(file_path)
                self.process_loaded_data()
                self.update_status("Datos cargados exitosamente")
                messagebox.showinfo("Éxito", "Datos cargados correctamente")
            except Exception as e:
                messagebox.showerror("Error", f"Error cargando datos: {str(e)}")
                self.update_status("Error cargando datos")
                
    def load_default_data(self):
        """Cargar datos por defecto"""
        
        try:
            self.update_status("Cargando datos por defecto...")
            
            # Intentar cargar el archivo por defecto
            default_file = 'neural_data_dataframe.csv'
            if os.path.exists(default_file):
                self.data = pd.read_csv(default_file)
            else:
                # Generar datos sintéticos si no existe el archivo
                self.generate_synthetic_data()
                
            self.process_loaded_data()
            self.update_status("Datos por defecto cargados")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error cargando datos por defecto: {str(e)}")
            self.update_status("Error cargando datos por defecto")
            
    def generate_synthetic_data(self):
        """Generar datos sintéticos para demostración"""
        
        np.random.seed(42)
        n_timepoints = 1000
        n_neurons = 50
        
        # Generar actividad neural sintética con patrones
        time = np.linspace(0, 100, n_timepoints)
        neural_data = np.zeros((n_timepoints, n_neurons))
        
        for i in range(n_neurons):
            # Diferentes patrones para diferentes neuronas
            frequency = 0.1 + i * 0.05
            phase = i * np.pi / n_neurons
            noise_level = 0.3
            
            signal = np.sin(2 * np.pi * frequency * time + phase)
            noise = np.random.normal(0, noise_level, n_timepoints)
            neural_data[:, i] = signal + noise
            
        # Crear DataFrame
        columns = ['Time_minutes'] + [f'Neuron_{i+1:03d}' for i in range(n_neurons)]
        data_dict = {'Time_minutes': time}
        
        for i in range(n_neurons):
            data_dict[f'Neuron_{i+1:03d}'] = neural_data[:, i]
            
        self.data = pd.DataFrame(data_dict)
        
    def process_loaded_data(self):
        """Procesar datos cargados"""
        
        # Identificar columnas de neuronas
        neuron_cols = [col for col in self.data.columns if 'Neuron' in col or 'neuron' in col]
        
        if not neuron_cols:
            raise ValueError("No se encontraron columnas de neuronas en el archivo")
            
        self.neural_data = self.data[neuron_cols].values
        self.neuron_names = neuron_cols
        
        # Tiempo
        if 'Time_minutes' in self.data.columns:
            self.time_points = self.data['Time_minutes'].values
        else:
            self.time_points = np.arange(len(self.neural_data))
            
        # Actualizar info
        info_text = f"Datos: {self.neural_data.shape[0]} timepoints × {self.neural_data.shape[1]} neuronas"
        self.data_info_label.config(text=info_text, foreground="black")
        
        # Aplicar preprocesamiento inicial
        self.update_preprocessing()
        
    def update_preprocessing(self):
        """Actualizar preprocesamiento de datos"""
        
        if self.neural_data is None:
            return
            
        try:
            self.update_status("Aplicando preprocesamiento...")
            
            data_to_process = self.neural_data.copy()
            
            # Aplicar filtro temporal si está seleccionado
            if self.apply_filter.get():
                window = self.filter_window.get()
                if window % 2 == 0:
                    window += 1  # Asegurar que sea impar
                    
                for i in range(data_to_process.shape[1]):
                    data_to_process[:, i] = savgol_filter(data_to_process[:, i], window, 3)
            
            # Aplicar escalado
            scaling_method = self.current_preprocessing.get()
            
            if scaling_method == "StandardScaler":
                scaler = StandardScaler()
                self.preprocessed_data = scaler.fit_transform(data_to_process)
            elif scaling_method == "MinMaxScaler":
                scaler = MinMaxScaler()
                self.preprocessed_data = scaler.fit_transform(data_to_process)
            elif scaling_method == "RobustScaler":
                scaler = RobustScaler()
                self.preprocessed_data = scaler.fit_transform(data_to_process)
            elif scaling_method == "Z-Score":
                self.preprocessed_data = zscore(data_to_process, axis=0)
            else:  # None
                self.preprocessed_data = data_to_process
                
            # Recalcular PCA y derivadas
            self.update_pca()
            self.calculate_derivatives()
            
            self.update_status("Preprocesamiento completado")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en preprocesamiento: {str(e)}")
            
    def update_pca(self):
        """Actualizar análisis PCA"""
        
        if self.preprocessed_data is None:
            return
            
        try:
            n_comp = min(self.n_components.get(), self.preprocessed_data.shape[1], self.preprocessed_data.shape[0])
            
            pca = PCA(n_components=n_comp)
            pca_scores = pca.fit_transform(self.preprocessed_data)
            
            self.pca_results = {
                'scores': pca_scores,
                'loadings': pca.components_,
                'explained_variance': pca.explained_variance_ratio_,
                'pca_object': pca
            }
            
        except Exception as e:
            messagebox.showerror("Error", f"Error calculando PCA: {str(e)}")
            
    def calculate_derivatives(self):
        """Calcular derivadas del firing rate"""
        
        if self.preprocessed_data is None:
            return
            
        try:
            # Calcular derivadas usando Savitzky-Golay
            window = 5
            if len(self.preprocessed_data) > window:
                self.derivatives = np.array([
                    savgol_filter(self.preprocessed_data[:, i], window, 3, deriv=1)
                    for i in range(self.preprocessed_data.shape[1])
                ]).T
            else:
                # Usar gradiente simple si hay pocos datos
                self.derivatives = np.gradient(self.preprocessed_data, axis=0)
                
        except Exception as e:
            messagebox.showerror("Error", f"Error calculando derivadas: {str(e)}")
            
    def identify_behaviors(self):
        """Identificar comportamientos usando clustering"""
        
        if self.pca_results is None:
            messagebox.showwarning("Advertencia", "Primero debe cargar y procesar los datos")
            return
            
        try:
            self.update_status("Identificando comportamientos...")
            
            # Usar los primeros 3 PCs para clustering
            pca_data = self.pca_results['scores'][:, :3]
            
            # K-means con 4 clusters (forward, reverse, pause, turn)
            kmeans = KMeans(n_clusters=4, random_state=42)
            behavior_labels = kmeans.fit_predict(pca_data)
            
            # Asignar nombres a los clusters
            behavior_names = {0: "Forward", 1: "Reverse", 2: "Pause", 3: "Turn"}
            
            self.behavioral_data = {
                'labels': behavior_labels,
                'names': behavior_names,
                'centroids': kmeans.cluster_centers_
            }
            
            # Actualizar info
            unique_behaviors = np.unique(behavior_labels)
            info_text = f"Identificados {len(unique_behaviors)} tipos de comportamiento"
            self.behavior_info_label.config(text=info_text, foreground="black")
            
            self.update_status("Comportamientos identificados")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error identificando comportamientos: {str(e)}")
            
    def update_visualization(self):
        """Actualizar visualización principal"""
        
        if self.preprocessed_data is None or self.pca_results is None:
            return
            
        try:
            self.update_status("Actualizando visualización...")
            
            # Ocultar placeholder
            self.plot_placeholder.pack_forget()
            
            # Crear gráfico según el tipo seleccionado
            plot_type = self.current_plot_type.get()
            
            if plot_type == "PC_vs_Time":
                self.create_pc_vs_time_plot()
            elif plot_type == "PC_vs_PC":
                self.create_pc_vs_pc_plot()
            elif plot_type == "PCA_3D":
                self.create_pca_3d_plot()
            elif plot_type == "Derivatives_vs_Time":
                self.create_derivatives_vs_time_plot()
            elif plot_type == "Derivatives_2D":
                self.create_derivatives_2d_plot()
            elif plot_type == "Derivatives_3D":
                self.create_derivatives_3d_plot()
                
            self.update_status("Visualización actualizada")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error actualizando visualización: {str(e)}")
            self.update_status("Error en visualización")
            
    def create_pc_vs_time_plot(self):
        """Crear gráfico PC vs Tiempo"""
        
        pc_idx = self.pc_x.get() - 1
        pca_scores = self.pca_results['scores']
        explained_var = self.pca_results['explained_variance']
        
        fig = go.Figure()
        
        # Datos a mostrar
        y_data = pca_scores[:, pc_idx]
        if self.show_derivatives.get() and self.derivatives is not None:
            # Calcular PCA de derivadas
            pca_deriv = PCA(n_components=self.n_components.get())
            deriv_scores = pca_deriv.fit_transform(self.derivatives)
            y_data = deriv_scores[:, pc_idx]
            title_suffix = "(Derivadas)"
        else:
            title_suffix = "(Firing Rate)"
            
        # Filtrar por comportamiento si está seleccionado
        mask = self.get_behavior_mask()
        x_data = self.time_points[mask]
        y_data = y_data[mask]
        
        # Crear trace
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='lines+markers' if self.show_points.get() else 'lines',
            line=dict(color=self.get_color_for_behavior(), width=2),
            marker=dict(size=4, opacity=0.7),
            name=f'PC{pc_idx+1} {title_suffix}'
        ))
        
        fig.update_layout(
            title=f'PC{pc_idx+1} vs Tiempo {title_suffix}<br>' + 
                  f'<sub>Varianza explicada: {explained_var[pc_idx]:.2%}</sub>',
            xaxis_title='Tiempo',
            yaxis_title=f'PC{pc_idx+1}',
            template='plotly_white',
            hovermode='x unified'
        )
        
        self.display_plotly_figure(fig)
        
    def create_pc_vs_pc_plot(self):
        """Crear gráfico PC vs PC (2D)"""
        
        pc_x_idx = self.pc_x.get() - 1
        pc_y_idx = self.pc_y.get() - 1
        
        pca_scores = self.pca_results['scores']
        explained_var = self.pca_results['explained_variance']
        
        fig = go.Figure()
        
        # Datos a mostrar
        if self.show_derivatives.get() and self.derivatives is not None:
            pca_deriv = PCA(n_components=self.n_components.get())
            scores = pca_deriv.fit_transform(self.derivatives)
            title_suffix = "(Derivadas)"
        else:
            scores = pca_scores
            title_suffix = "(Firing Rate)"
            
        # Filtrar por comportamiento
        mask = self.get_behavior_mask()
        x_data = scores[mask, pc_x_idx]
        y_data = scores[mask, pc_y_idx]
        
        # Color por tiempo o comportamiento
        if self.behavioral_data is not None and self.current_behavior.get() == "All":
            color_data = self.behavioral_data['labels'][mask]
            colorscale = 'Set1'
        else:
            color_data = self.time_points[mask]
            colorscale = self.color_scheme.get()
            
        # Crear trace
        mode = 'lines+markers' if self.show_trajectory.get() and self.show_points.get() else \
               'lines' if self.show_trajectory.get() else 'markers'
               
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode=mode,
            marker=dict(
                size=6,
                color=color_data,
                colorscale=colorscale,
                opacity=0.8,
                colorbar=dict(title="Tiempo" if self.current_behavior.get() != "All" else "Comportamiento")
            ),
            line=dict(width=2, color='rgba(100,100,100,0.5)'),
            name='Trayectoria Neural'
        ))
        
        fig.update_layout(
            title=f'PC{pc_x_idx+1} vs PC{pc_y_idx+1} {title_suffix}<br>' + 
                  f'<sub>Varianza: {explained_var[pc_x_idx]:.2%} × {explained_var[pc_y_idx]:.2%}</sub>',
            xaxis_title=f'PC{pc_x_idx+1} ({explained_var[pc_x_idx]:.2%})',
            yaxis_title=f'PC{pc_y_idx+1} ({explained_var[pc_y_idx]:.2%})',
            template='plotly_white',
            hovermode='closest'
        )
        
        self.display_plotly_figure(fig)
        
    def create_pca_3d_plot(self):
        """Crear gráfico PCA 3D"""
        
        pc_x_idx = self.pc_x.get() - 1
        pc_y_idx = self.pc_y.get() - 1
        pc_z_idx = self.pc_z.get() - 1
        
        pca_scores = self.pca_results['scores']
        explained_var = self.pca_results['explained_variance']
        
        fig = go.Figure()
        
        # Datos a mostrar
        if self.show_derivatives.get() and self.derivatives is not None:
            pca_deriv = PCA(n_components=self.n_components.get())
            scores = pca_deriv.fit_transform(self.derivatives)
            title_suffix = "(Derivadas)"
        else:
            scores = pca_scores
            title_suffix = "(Firing Rate)"
            
        # Filtrar por comportamiento
        mask = self.get_behavior_mask()
        x_data = scores[mask, pc_x_idx]
        y_data = scores[mask, pc_y_idx]
        z_data = scores[mask, pc_z_idx]
        
        # Color por tiempo o comportamiento
        if self.behavioral_data is not None and self.current_behavior.get() == "All":
            color_data = self.behavioral_data['labels'][mask]
            colorscale = 'Set1'
        else:
            color_data = self.time_points[mask]
            colorscale = self.color_scheme.get()
            
        # Crear trace
        mode = 'lines+markers' if self.show_trajectory.get() and self.show_points.get() else \
               'lines' if self.show_trajectory.get() else 'markers'
               
        fig.add_trace(go.Scatter3d(
            x=x_data,
            y=y_data,
            z=z_data,
            mode=mode,
            marker=dict(
                size=4,
                color=color_data,
                colorscale=colorscale,
                opacity=0.8,
                colorbar=dict(title="Tiempo" if self.current_behavior.get() != "All" else "Comportamiento")
            ),
            line=dict(width=4, color='rgba(100,100,100,0.5)'),
            name='Trayectoria Neural 3D'
        ))
        
        fig.update_layout(
            title=f'PCA 3D {title_suffix}<br>' + 
                  f'<sub>PC{pc_x_idx+1} ({explained_var[pc_x_idx]:.2%}) × ' +
                  f'PC{pc_y_idx+1} ({explained_var[pc_y_idx]:.2%}) × ' + 
                  f'PC{pc_z_idx+1} ({explained_var[pc_z_idx]:.2%})</sub>',
            scene=dict(
                xaxis_title=f'PC{pc_x_idx+1}',
                yaxis_title=f'PC{pc_y_idx+1}',
                zaxis_title=f'PC{pc_z_idx+1}',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            template='plotly_white'
        )
        
        self.display_plotly_figure(fig)
        
    def create_derivatives_vs_time_plot(self):
        """Crear gráfico Derivadas vs Tiempo"""
        
        if self.derivatives is None:
            return
            
        # Usar PCA para reducir dimensionalidad de derivadas
        pca_deriv = PCA(n_components=min(10, self.derivatives.shape[1]))
        deriv_scores = pca_deriv.fit_transform(self.derivatives)
        
        pc_idx = self.pc_x.get() - 1
        
        fig = go.Figure()
        
        # Filtrar por comportamiento
        mask = self.get_behavior_mask()
        x_data = self.time_points[mask]
        y_data = deriv_scores[mask, pc_idx]
        
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='lines+markers' if self.show_points.get() else 'lines',
            line=dict(color=self.get_color_for_behavior(), width=2),
            marker=dict(size=4, opacity=0.7),
            name=f'Derivada PC{pc_idx+1}'
        ))
        
        fig.update_layout(
            title=f'Derivadas PC{pc_idx+1} vs Tiempo<br>' + 
                  f'<sub>Velocidad de Cambio Neural</sub>',
            xaxis_title='Tiempo',
            yaxis_title=f'Derivada PC{pc_idx+1}',
            template='plotly_white',
            hovermode='x unified'
        )
        
        self.display_plotly_figure(fig)
        
    def create_derivatives_2d_plot(self):
        """Crear gráfico Derivadas 2D"""
        
        if self.derivatives is None:
            return
            
        pca_deriv = PCA(n_components=min(10, self.derivatives.shape[1]))
        deriv_scores = pca_deriv.fit_transform(self.derivatives)
        
        pc_x_idx = self.pc_x.get() - 1
        pc_y_idx = self.pc_y.get() - 1
        
        fig = go.Figure()
        
        # Filtrar por comportamiento
        mask = self.get_behavior_mask()
        x_data = deriv_scores[mask, pc_x_idx]
        y_data = deriv_scores[mask, pc_y_idx]
        
        # Color por tiempo o comportamiento
        if self.behavioral_data is not None and self.current_behavior.get() == "All":
            color_data = self.behavioral_data['labels'][mask]
            colorscale = 'Set1'
        else:
            color_data = self.time_points[mask]
            colorscale = self.color_scheme.get()
            
        mode = 'lines+markers' if self.show_trajectory.get() and self.show_points.get() else \
               'lines' if self.show_trajectory.get() else 'markers'
               
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode=mode,
            marker=dict(
                size=6,
                color=color_data,
                colorscale=colorscale,
                opacity=0.8,
                colorbar=dict(title="Tiempo" if self.current_behavior.get() != "All" else "Comportamiento")
            ),
            line=dict(width=2, color='rgba(100,100,100,0.5)'),
            name='Trayectoria Derivadas'
        ))
        
        fig.update_layout(
            title=f'Derivadas PC{pc_x_idx+1} vs PC{pc_y_idx+1}<br>' + 
                  f'<sub>Espacio de Velocidades de Cambio</sub>',
            xaxis_title=f'Derivada PC{pc_x_idx+1}',
            yaxis_title=f'Derivada PC{pc_y_idx+1}',
            template='plotly_white',
            hovermode='closest'
        )
        
        self.display_plotly_figure(fig)
        
    def create_derivatives_3d_plot(self):
        """Crear gráfico Derivadas 3D"""
        
        if self.derivatives is None:
            return
            
        pca_deriv = PCA(n_components=min(10, self.derivatives.shape[1]))
        deriv_scores = pca_deriv.fit_transform(self.derivatives)
        
        pc_x_idx = self.pc_x.get() - 1
        pc_y_idx = self.pc_y.get() - 1
        pc_z_idx = self.pc_z.get() - 1
        
        fig = go.Figure()
        
        # Filtrar por comportamiento
        mask = self.get_behavior_mask()
        x_data = deriv_scores[mask, pc_x_idx]
        y_data = deriv_scores[mask, pc_y_idx]
        z_data = deriv_scores[mask, pc_z_idx]
        
        # Color por tiempo o comportamiento
        if self.behavioral_data is not None and self.current_behavior.get() == "All":
            color_data = self.behavioral_data['labels'][mask]
            colorscale = 'Set1'
        else:
            color_data = self.time_points[mask]
            colorscale = self.color_scheme.get()
            
        mode = 'lines+markers' if self.show_trajectory.get() and self.show_points.get() else \
               'lines' if self.show_trajectory.get() else 'markers'
               
        fig.add_trace(go.Scatter3d(
            x=x_data,
            y=y_data,
            z=z_data,
            mode=mode,
            marker=dict(
                size=4,
                color=color_data,
                colorscale=colorscale,
                opacity=0.8,
                colorbar=dict(title="Tiempo" if self.current_behavior.get() != "All" else "Comportamiento")
            ),
            line=dict(width=4, color='rgba(100,100,100,0.5)'),
            name='Trayectoria Derivadas 3D'
        ))
        
        fig.update_layout(
            title=f'Derivadas 3D<br>' + 
                  f'<sub>Espacio de Velocidades de Cambio Neural</sub>',
            scene=dict(
                xaxis_title=f'Derivada PC{pc_x_idx+1}',
                yaxis_title=f'Derivada PC{pc_y_idx+1}',
                zaxis_title=f'Derivada PC{pc_z_idx+1}',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            template='plotly_white'
        )
        
        self.display_plotly_figure(fig)
        
    def get_behavior_mask(self):
        """Obtener máscara para filtrar por comportamiento"""
        
        if self.current_behavior.get() == "All" or self.behavioral_data is None:
            return np.ones(len(self.time_points), dtype=bool)
            
        behavior_mapping = {"Forward": 0, "Reverse": 1, "Pause": 2, "Turn": 3}
        behavior_idx = behavior_mapping.get(self.current_behavior.get(), 0)
        
        return self.behavioral_data['labels'] == behavior_idx
        
    def get_color_for_behavior(self):
        """Obtener color para comportamiento específico"""
        
        color_mapping = {
            "All": "blue",
            "Forward": "green", 
            "Reverse": "red",
            "Pause": "orange",
            "Turn": "purple"
        }
        
        return color_mapping.get(self.current_behavior.get(), "blue")
        
    def display_plotly_figure(self, fig):
        """Mostrar figura de plotly en el panel"""
        
        # Crear archivo temporal HTML
        temp_file = os.path.join(tempfile.gettempdir(), "neural_plot.html")
        fig.write_html(temp_file)
        
        # Abrir en navegador (solución simple)
        webbrowser.open(f'file://{temp_file}')
        
        # Actualizar label en el frame
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
            
        info_label = ttk.Label(self.plot_frame, 
                              text="🌐 Gráfico abierto en navegador\n\n" +
                                   "El gráfico interactivo se ha abierto en tu navegador.\n" +
                                   "Usa los controles de plotly para interactuar:\n\n" +
                                   "🖱️ Arrastra para rotar (3D)\n" +
                                   "🔍 Scroll para zoom\n" +
                                   "📊 Hover para detalles\n" +
                                   "🎛️ Toolbar para más opciones",
                              font=('Arial', 11),
                              justify=tk.CENTER)
        info_label.pack(expand=True)
        
    def export_plot(self):
        """Exportar gráfico actual"""
        
        messagebox.showinfo("Exportar", "La función de exportación se abrirá en el navegador.\n" +
                           "Usa la toolbar de plotly para descargar el gráfico.")
                           
    def reset_view(self):
        """Resetear vista a valores por defecto"""
        
        self.pc_x.set(1)
        self.pc_y.set(2) 
        self.pc_z.set(3)
        self.current_behavior.set("All")
        self.show_derivatives.set(False)
        self.color_scheme.set("viridis")
        self.show_trajectory.set(True)
        self.show_points.set(True)
        
        self.update_visualization()


def main():
    """Función principal"""
    
    # Crear ventana principal
    root = tk.Tk()
    
    # Configurar icono y estilo
    try:
        root.iconbitmap('icon.ico')  # Si tienes un icono
    except:
        pass
        
    # Crear aplicación
    app = NeuralAnalysisApp(root)
    
    # Manejar cierre de ventana
    def on_closing():
        if messagebox.askokcancel("Salir", "¿Desea cerrar la aplicación?"):
            root.destroy()
            
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Ejecutar aplicación
    root.mainloop()


if __name__ == "__main__":
    main()